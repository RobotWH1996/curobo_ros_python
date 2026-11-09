"""OPT-backed ROS actions with scoped contacts and per-request payload geometry."""
from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace as NS
import math
import threading
import time
import numpy as np
import rclpy
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.logging import LoggingSeverity
from curobo_ros_interfaces.action import Pick, Place, PickPlace
from curobo_ros_python.planning.manipulation import ManipulationWorkflow, TaskCancelled
from curobo_ros_python.planning.geom import quat_mul_wxyz, transform_pose_wxyz
from curobo_ros_python.planning.payload import Payload, pose_matrix, outside_box, resolve_object_geometry
from curobo_ros_python.planning.pick_collision import set_payload
from curobo_ros_python.planning.convert import stamped_to_pose, numpy_traj_to_msg, group_goal_state, joint_map_to_curobo
from curobo_ros_python.planning.offline_trajectory import (
    blend_through, continuous_runs, cpu_dynamic_limits, gripper_segment, concatenate_segments,
    host_state_arrays, overlay_joint_motion, insert_stationary_hold, sample_trajectory, ros_duration)
from curobo.trajectory_optimizer import expand_joint_trajectory
from curobo_ros_python.planning.groups import GROUP_JOINTS
from curobo_ros_python.planning.trajectory import select_trajectory_joints


DEFAULTS={
    'pick_place_default_object_size':[0.0,0.0,0.0],
    'pick_place_timeout':180.,
    'gripper_wait_time':1.0,
    'keep_tool_z_vertical':False,
    'pick_place_joint_state_max_age':1.,
    'pick_place_cloud_max_age':2.,
    'pick_place_path_tolerance':.01,
    'pick_place_max_placed_objects':16,
}


def gripper_wait_time(value):
    value=float(value)
    if not math.isfinite(value) or value<0:
        raise ValueError('gripper_wait_time 必须是有限非负秒数')
    return value


def xyz(vector):
    return np.array([vector.x,vector.y,vector.z],dtype=np.float64)


def matrix(pose):
    return pose_matrix(xyz(pose.position),[pose.orientation.w,pose.orientation.x,pose.orientation.y,pose.orientation.z])


def relative_wxyz(pose):
    """Normalize a relative pose. An all-zero quaternion means identity."""
    translation=np.array([pose.position.x,pose.position.y,pose.position.z],dtype=np.float64)
    quaternion=np.array([pose.orientation.w,pose.orientation.x,pose.orientation.y,pose.orientation.z],dtype=np.float64)
    if not np.isfinite(translation).all() or not np.isfinite(quaternion).all():
        raise ValueError('相对目标位姿含无效数值')
    if float(np.linalg.norm(translation))>.5:
        raise ValueError('相对目标位姿的平移不能超过 0.5 m')
    if float(np.linalg.norm(quaternion))<1e-8:
        quaternion=np.array([1.,0.,0.,0.])
    else:
        quaternion/=np.linalg.norm(quaternion)
    return translation,quaternion


def compose_pose(target, relative, in_base_frame=False):
    """Apply relative at the reference pose. Base adds it in base_link; otherwise T_target * T_relative."""
    translation,quaternion=relative_wxyz(relative)
    position=target.pose.position
    orientation=target.pose.orientation
    reference_t=(position.x,position.y,position.z)
    reference_q=(orientation.w,orientation.x,orientation.y,orientation.z)
    if in_base_frame:
        result_t=tuple(float(reference_t[i])+float(translation[i]) for i in range(3))
        result_q=quat_mul_wxyz(quaternion,reference_q)
    else:
        result_t,result_q=transform_pose_wxyz(reference_t,reference_q,translation,quaternion)
    pose=copy.deepcopy(target)
    pose.pose.position.x,pose.pose.position.y,pose.pose.position.z=result_t
    pose.pose.orientation.w,pose.pose.orientation.x,pose.pose.orientation.y,pose.pose.orientation.z=result_q
    return pose


def validate_common(task):
    if task.group_name not in ('left_arm','right_arm'):
        raise ValueError('首版抓放Action只支持 left_arm/right_arm；原双臂多目标规划服务保留')
    if not math.isfinite(float(task.open_position)): raise ValueError('open_position 必须是有限关节位置')


class PickPlaceServer:
    def __init__(self,node):
        self.node=node
        gripper_wait_time(node._p('gripper_wait_time'))
        for name in ('pick_place_timeout','pick_place_joint_state_max_age',
                     'pick_place_cloud_max_age','pick_place_path_tolerance'):
            value=float(node._p(name))
            if not math.isfinite(value) or value<=0: raise ValueError(f'{name} 必须是有限正数')
        if not 1<=int(node._p('pick_place_max_placed_objects'))<=16:
            raise ValueError('pick_place_max_placed_objects 必须在1到16之间')
        self.lock=threading.Lock()
        self.busy=False
        self._active_backend=None
        self.held=None
        self.placed={}
        self.placed_seq=0
        self.preview_new_payload=False
        self.preview_exclusions=[]
        self.contact_exclusions=[]
        self.servers=[]
        for name,action in (('pick',Pick),('place',Place),('pick_place',PickPlace)):
            self.servers.append(ActionServer(node,action,name,
                goal_callback=lambda request,k=name:self.accept(k,request),
                cancel_callback=self.cancel,
                execute_callback=lambda handle,k=name,a=action:self.execute(k,a,handle),
                callback_group=node._action_group))

    def cancel(self, handle):
        backend=self._active_backend
        if backend is not None and bytes(backend.handle.goal_id.uuid)==bytes(handle.goal_id.uuid):
            backend.cancel_stream()
        return CancelResponse.ACCEPT

    def accept(self,kind,request):
        try:
            gripper_wait_time(self.node._p('gripper_wait_time'))
            timeout=float(request.timeout)
            if not math.isfinite(timeout) or timeout<0: raise ValueError('timeout必须为有限非负数')
            tasks=(request.pick,request.place) if kind=='pick_place' else (request.task,)
            for task in tasks: validate_common(task)
            if kind=='pick_place' and request.pick.group_name!=request.place.group_name:
                raise ValueError('PickPlace 的抓取和放置必须使用同一手臂')
            if self.node.execute_trajectory and self.node.executor_helper is None:
                raise ValueError('当前启动配置启用了执行，但轨迹执行器尚未初始化')
            if self.node._mpc_following: raise RuntimeError('MPC跟踪中，请先取消')
            if self.node.executor_helper is not None and self.node.executor_helper.execution_pending:
                raise RuntimeError('上一条控制轨迹尚未结束')
            if self.node.pointcloud_collision!='mesh': raise ValueError('首版抓放Action需要 pointcloud_collision=mesh')
            with self.lock:
                if self.busy: raise RuntimeError('抓放任务正忙')
                self.busy=True
            return GoalResponse.ACCEPT
        except Exception as exc:
            self.node.get_logger().warn(f'[抓放] action={kind} event=拒绝 reason={str(exc)!r}')
            return GoalResponse.REJECT

    def invalidate_world(self):
        p=self.node.planner
        p._cloud_loaded={}
        p._cloud_fcl_sig=None

    def filter_cloud(self,points):
        for transform,size in self.preview_exclusions+self.contact_exclusions:
            points=points[outside_box(points,transform,size)]
        payload=self.held
        if payload is None: return points
        keep=np.ones(len(points),dtype=bool)
        # Old source occupancy is removed only from frames available at attach.
        # Never permanently carve a hole at an object's previous location.
        if self.preview_new_payload or self.node._pc_seq<=getattr(payload,'cloud_seq_at_attach',-1):
            keep &= outside_box(points,payload.origin,payload.size)
        if not self.preview_new_payload:
            checker=self.node.planner._mesh_checker
            values=dict(self.node._joint_map)
            tool=checker._transforms(values,[payload.tool])[0]
            keep &= outside_box(points,tool@payload.in_tool,payload.size)
        return points[keep]

    def execute(self,kind,action,handle):
        result=action.Result()
        result.execution_requested=self.node.execute_trajectory
        started=time.monotonic()
        task=handle.request.pick if kind=='pick_place' else handle.request.task
        context=(f'[抓放] task={bytes(handle.goal_id.uuid).hex()} action={kind} '
                 f'group={task.group_name} mode={"execute" if self.node.execute_trajectory else "publish"}')
        self.node.get_logger().info(f'{context} event=开始')
        held_gpu=False
        backend=None
        try:
            if not self.node._gpu_lock.acquire(blocking=False): raise RuntimeError('规划器正忙')
            held_gpu=True
            backend=TaskBackend(self,handle,action)
            backend.log_prefix=context
            backend.started=started
            self._active_backend=backend
            runner=ManipulationWorkflow(backend)
            if kind=='pick': runner.pick(handle.request.task)
            elif kind=='place': runner.place(handle.request.task)
            else: runner.pick_place(handle.request.pick,handle.request.place)
            backend.check()
            backend.flush(task.group_name, final=True)
            backend.check()
            result.trajectory=backend.full_trajectory()
            result.success=True
            result.message=('完整轨迹规划成功；已按停顿点分块提交，未等待执行结果'
                            if self.node.execute_trajectory else '完整轨迹规划成功；已分块发布，未下发控制器')
            result.execution_completed=False
        except TaskCancelled as exc:
            result.message=str(exc)
        except Exception as exc:
            result.message=f'{type(exc).__name__}: {exc}'
        finally:
            if backend is not None:
                if not result.success:
                    if handle.is_cancel_requested or backend._stream_cancelled or not rclpy.ok():
                        backend.cancel_stream()
                    elif backend._stream is not None:
                        # Close planning while the validated prefix finishes at its own stop.
                        backend._stream.finish()
                        if not backend._stream.error:
                            result.message+='；保留已提交轨迹执行至原定停止点'
                    if backend._completed:
                        result.trajectory=backend.full_trajectory()
                result.stage=backend.stage
                result.trajectories=backend._completed
                result.trajectory_stages=backend._completed_stages
                try:
                    backend.cleanup()
                except Exception as exc:
                    result.success=False
                    result.message+=f'；恢复规划状态失败: {exc}'
                    backend.cancel_stream()
            self._active_backend=None
            if held_gpu: self.node._gpu_lock.release()
            with self.lock: self.busy=False
        status='成功' if result.success else '取消' if handle.is_cancel_requested else '失败'
        message=(f'{context} event=结束 status={status} stage={result.stage or "validate"} '
            f'trajectories={len(result.trajectories)} '
            f'elapsed_ms={(time.monotonic()-started)*1000:.1f} '
            f'{backend.timing_summary() if backend is not None else ""} reason={result.message!r}')
        if result.success: self.node.get_logger().info(message)
        elif handle.is_cancel_requested: self.node.get_logger().warn(message)
        else: self.node.get_logger().error(message)
        if result.success:
            handle.succeed()
        elif handle.is_cancel_requested:
            handle.canceled()
        else:
            handle.abort()
        return result


class TaskBackend:
    def __init__(self,server,handle,action):
        self.server=server;self.node=server.node;self.p=self.node.planner
        self.handle=handle;self.action=action
        self.p.keep_tool_z_vertical=bool(self.node._p('keep_tool_z_vertical'))
        timeout=float(handle.request.timeout) or float(self.node._p('pick_place_timeout'))
        self.deadline=time.monotonic()+timeout
        self.gripper_wait_time=gripper_wait_time(self.node._p('gripper_wait_time'))
        self.stage='validate'
        self.released_key=None
        self.trajectories=[];self.trajectory_stages=[]
        self.saved_held=server.held
        self.saved_placed=dict(server.placed)
        self.saved_preview=(server.preview_new_payload,list(server.preview_exclusions))
        self.saved_worlds={g:m.scene_collision_checker.scene_model for g,m in self.p._planners.items()}
        self.saved_mesh_scenes=dict(self.p._mesh_scenes)
        self.saved_fcl_environment=self.p._mesh_checker.environment
        self.virtual_state=None
        self._feedback_checked=False
        self._world_checked=False
        self._prepared_cloud=None
        self._return_segments={}
        self._open_along={}
        self._pending_open=None
        self._segment_checks=[]
        self._open_collision_joint=None
        self._changed_groups=set()
        self.virtual_joints=dict(self.node._joint_map)
        self.joint_names=[j.get('name') for j in self.p._mesh_checker.ordered
                          if j.get('type')!='fixed' and j.find('mimic') is None]
        self.released_box=None
        self.last_gripper_position=0.
        self.started=time.monotonic()
        self._completed=[];self._completed_stages=[]
        self._stream=None;self._stream_cancelled=False;self._stream_stamp=None
        self._blocks=0;self._first_ready_ms=None;self._first_submit_ms=None

    def check(self):
        if self._stream is not None and self._stream.error: raise RuntimeError(self._stream.error)
        if self.handle.is_cancel_requested or self._stream_cancelled: raise TaskCancelled('抓放任务已取消')
        if not rclpy.ok(): raise TaskCancelled('节点关闭')
        if time.monotonic()>self.deadline: raise TimeoutError('抓放任务总超时')

    def feedback(self,stage,message=''):
        self.stage=stage
        fb=self.action.Feedback();fb.stage=stage;fb.message=message or stage
        self.handle.publish_feedback(fb)
        self.check()
        self.log('阶段开始',message)

    def log(self,event,details='',level='info'):
        message=f'{self.log_prefix} stage={self.stage} event={event} {details}'.rstrip()
        # Humble caches severity per call site; keep INFO and DEBUG separate.
        if level=='debug': self.node.get_logger().debug(message)
        else: self.node.get_logger().info(message)

    def cancel_stream(self):
        self._stream_cancelled=True
        if self._stream is not None:
            self._stream.cancel()

    def timing_summary(self):
        ready='none' if self._first_ready_ms is None else f'{self._first_ready_ms:.1f}'
        submit='none' if self._first_submit_ms is None else f'{self._first_submit_ms:.1f}'
        return f'blocks={self._blocks} first_ready_ms={ready} first_submit_ms={submit}'

    def full_trajectory(self):
        trajectory=concatenate_segments(self._completed)
        trajectory.header.frame_id=self.node.base_frame
        if self._stream_stamp is not None:
            trajectory.header.stamp=copy.deepcopy(self._stream_stamp)
        return trajectory

    def flush(self, group, final=False):
        """Commit only the current stopped block; future planning cannot modify it."""
        self.check()
        if self.trajectories:
            started=time.monotonic()
            block=self.merge_continuous_motions(group)
            if self._completed:
                previous=self._completed[-1].points[-1];following=block.points[0]
                for field,tolerance in (('positions',1e-6),('velocities',1e-5),('accelerations',1e-4)):
                    if not np.allclose(getattr(previous,field),getattr(following,field),atol=tolerance,rtol=0.):
                        raise RuntimeError(f'轨迹块接缝 {field} 不连续')
            combined=block
            if self._completed:
                combined=concatenate_segments(self._completed+self.trajectories)
                self._validated_samples(group,combined)
            ready=time.monotonic()
            if self._first_ready_ms is None:
                self._first_ready_ms=(ready-self.started)*1000.
            self.check()
            if self.node.execute_trajectory:
                joint=group.split('_')[0]+'_gripper'
                commands=[(group,select_trajectory_joints(combined,GROUP_JOINTS[group])),
                          (joint,select_trajectory_joints(combined,[joint]))]
                if self._stream is None:
                    self._stream=self.node.executor_helper.start_stream(commands)
                    self._stream_stamp=copy.deepcopy(self._stream.stamp)
                    self._first_submit_ms=(time.monotonic()-self.started)*1000.
                else:
                    self._stream.append(commands)
            elif self._stream_stamp is None:
                self._stream_stamp=self.node.get_clock().now().to_msg()
            self._completed.extend(self.trajectories)
            self._completed_stages.extend(self.trajectory_stages)
            self.trajectories=[];self.trajectory_stages=[];self._segment_checks=[]
            self._pending_open=None;self._blocks+=1
            self.node._traj_pub.publish(self.full_trajectory())
            self.log('轨迹块提交',f'block={self._blocks} labels={",".join(self._completed_stages)} '
                f'block_ready_ms={(ready-started)*1000:.1f} ready_ms={(ready-self.started)*1000:.1f} '
                f'prefix_duration_s={combined.points[-1].time_from_start.sec+combined.points[-1].time_from_start.nanosec*1e-9:.3f} '
                f'controller={str(self.node.execute_trajectory).lower()}')
        if final and self._stream is not None:
            self._stream.finish()

    def _preflight(self,task):
        validate_common(task)
        self.check()
        from curobo_ros_python.planning.pick_collision import contact_links,models
        if not set(contact_links(task.group_name)).issubset(self.p._mesh_checker.names):
            raise ValueError('机器人模型缺少此手臂的夹指碰撞几何')
        payload_link='pick_place_'+task.group_name+'_payload'
        if any(payload_link not in kin.link_name_to_idx_map for kin in models(self.p)):
            raise ValueError('机器人模型未预留此手臂携物球，请确认机器人配置')
        self._validate_joint_feedback()
        self._validate_gripper_target(task,task.open_position)

    def _validate_joint_feedback(self):
        if self._feedback_checked: return
        # Require every observed joint used by full-body FCL, not defaults.
        names=[j.get('name') for j in self.p._mesh_checker.ordered
               if j.get('type')!='fixed' and j.find('mimic') is None]
        missing=[name for name in names if name not in self.virtual_joints
                 or not math.isfinite(float(self.virtual_joints[name]))]
        if missing: raise RuntimeError(f'缺少必要关节反馈: {missing}')
        # Virtual positions are predicted, while stream health remains live.
        # Keep the same freshness limit and check it at every task preflight.
        now=time.monotonic();ages=self.node._joint_received
        stale=[name for name in names if now-ages.get(name,0)>self.node._p('pick_place_joint_state_max_age')]
        if stale: raise RuntimeError(f'初始关节反馈过期: {stale}')
        self._feedback_checked=True

    def validate_pair(self,pick,place):
        if pick.group_name!=place.group_name:
            raise ValueError('抓放必须使用相同手臂')
        self._validate_place_fields(place)

    def validate_pick(self,task):
        self._preflight(task)
        self._validate_gripper_target(task,task.close_position)
        if abs(task.close_position-task.open_position)<1e-6:
            raise ValueError('抓取的开、闭关节位置必须不同')
        self.base_target=self.node._pose_in_base(task.tool_pose)
        matrix(self.base_target.pose)
        in_base=bool(task.in_base_frame)
        self.base_pick=compose_pose(self.base_target,task.pick_pose,in_base)
        self.base_prepick=compose_pose(self.base_target,task.prepick_in_target,in_base)
        self.base_lift=compose_pose(self.base_target,task.lift_in_target,in_base)
        self._require_separation(self.base_pick,self.base_prepick,'预抓取')
        self._require_separation(self.base_pick,self.base_lift,'抬升')
        self.object_size,self.object_in_tool=self._geometry(task.object_size,self.base_target,self.base_pick)
        self.pick_via=self._resolve_route(task)
        self.pick_end=self._resolve_motion(task.end_pose,task.group_name,allow_empty=True)

    def _validate_place_fields(self,task):
        validate_common(task)
        reference=self.node._pose_in_base(task.tool_pose)
        matrix(reference.pose)
        self.place_reference=reference
        in_base=bool(task.in_base_frame)
        self.base_place=compose_pose(reference,task.place_pose,in_base)
        self.base_preplace=compose_pose(reference,task.preplace_in_target,in_base)
        self.base_retreat=compose_pose(reference,task.retreat_in_target,in_base)
        self._require_separation(self.base_place,self.base_preplace,'预放置')
        self._require_separation(self.base_place,self.base_retreat,'撤退')
        self.place_via=self._resolve_route(task)
        self.place_end=self._resolve_motion(task.end_pose,task.group_name,allow_empty=True)

    def _resolve_motion(self, waypoint, group, allow_empty=False):
        self.check()
        kind=waypoint.type.strip().lower()
        if not kind and allow_empty:
            if waypoint.joint_positions or waypoint.pose.header.frame_id or not np.allclose(matrix(waypoint.pose.pose),np.eye(4)):
                raise ValueError('end_pose 已填写目标时必须指定 type=pose/joint')
            return None
        if kind=='joint':
            values=np.asarray(waypoint.joint_positions,dtype=np.float64)
            if len(values)!=len(GROUP_JOINTS[group]) or not np.isfinite(values).all():
                raise ValueError(f'{group} 关节路点需要{len(GROUP_JOINTS[group])}个有限弧度值')
            if waypoint.pose.header.frame_id or not np.allclose(matrix(waypoint.pose.pose),np.eye(4)):
                raise ValueError('joint 路点不能同时填写 pose')
            return NS(kind='joint',positions=values.tolist(),pose=None)
        if kind=='pose':
            if waypoint.joint_positions: raise ValueError('pose 路点不能同时填写 joint_positions')
            pose=self.node._pose_in_base(waypoint.pose);matrix(pose.pose)
            return NS(kind='pose',positions=None,pose=pose)
        raise ValueError('路点 type 必须是 pose 或 joint；只有空 end_pose 可以省略 type')

    def _resolve_route(self,task):
        return [self._resolve_motion(w,task.group_name) for w in task.waypoints]

    def _route_for_stage(self,stage):
        return self.pick_via if stage=='prepick' else self.place_via if stage=='preplace' else []

    def end(self,task):
        target=self.pick_end if hasattr(task,'pick_pose') else self.place_end
        if target is not None: self._move_segment(task,'end',waypoint=target)

    def enable_payload(self,task):
        self.feedback('attach')
        set_payload(self.p,self.server.held)
        self.log('携物碰撞启用','gpu=on fcl=on')

    @contextmanager
    def _released_object_scope(self,task,stage):
        if stage!='retreat' or self.released_box is None:
            yield
            return
        group=task.group_name
        old_placed=dict(self.server.placed)
        old_exclusions=list(self.server.contact_exclusions)
        motion=self.p._planners[group]
        old_scene=motion.scene_collision_checker.scene_model
        old_fcl=self.p._mesh_scenes[group]
        try:
            self.server.contact_exclusions.append(self.released_box)
            if self.released_key is not None: self.server.placed.pop(self.released_key,None)
            self.p._placed_objects=dict(self.server.placed)
            self.server.invalidate_world()
            yield
        finally:
            self.server.placed=old_placed
            self.server.contact_exclusions=old_exclusions
            self.p._placed_objects=dict(old_placed)
            from curobo._src.geom.types import SceneCfg
            motion.update_world(old_scene if old_scene is not None else SceneCfg())
            self.p._mesh_scenes[group]=old_fcl
            self.server.invalidate_world()

    def _validate_gripper_target(self,task,position):
        name=task.group_name.split('_')[0]+'_gripper'
        limits=self.p._gripper_state._limits
        if name not in limits: raise ValueError(f'模型没有夹爪关节 {name}')
        lo,hi=limits[name]
        if not math.isfinite(position) or not lo<=position<=hi:
            raise ValueError(f'{name} 目标须在模型关节限位 [{lo}, {hi}] 内')
        return name

    def _geometry(self,size,reference,grasp):
        in_tool=np.linalg.inv(matrix(grasp.pose))@matrix(reference.pose)
        return resolve_object_geometry(xyz(size),in_tool,False,
            self.node._p('pick_place_default_object_size'),[0,0,0,0,0,0,1])

    def validate_place(self,task):
        self._preflight(task);self._validate_place_fields(task)
        payload=self.server.held
        if payload is None:
            self.object_size,self.object_in_tool=self._geometry(task.object_size,self.place_reference,self.base_place)
            current=self._current()
            self.p._sync_locked_arm(task.group_name,current)
            transform=self._tool_matrix(current,task.group_name+'_ee_link')
            payload=Payload(task.group_name,self.object_size,self.object_in_tool,transform@self.object_in_tool)
            payload.cloud_seq_at_attach=self.node._pc_seq
            self.server.held=payload
            self.server.preview_new_payload=True
            set_payload(self.p,payload)
            self.server.invalidate_world()
        if payload.group!=task.group_name: raise ValueError('放置手臂与本次携物模型不匹配')
        name=task.group_name.split('_')[0]+'_gripper'
        if abs(task.open_position-self.virtual_joints[name])<1e-6:
            raise ValueError('放置释放位置与初始夹持位置相同')
        if len(self.server.placed)>=self.node._p('pick_place_max_placed_objects'):
            raise RuntimeError('已放置工件数量达到场景容量')

    def _current(self):
        started=time.monotonic();rebuilt=self.virtual_state is None
        gripper_before=getattr(self.p._gripper_state,'_previous',None)
        fixed_before=getattr(self.p._fixed_state,'positions',None)
        gripper_signature=getattr(self.p._gripper_state,'_model_signature',None)
        fixed_signature=getattr(self.p._fixed_state,'_model_signature',None)
        if self.virtual_state is None:
            self.virtual_state,_=joint_map_to_curobo(self.p.joint_names,self.virtual_joints,self.p.device,
                                                    default_position=self.p.default_joint_position)
        built=time.monotonic()
        self.p.update_gripper_state(self._collision_joints())
        updated=time.monotonic()
        state=self.virtual_state.clone()
        finished=time.monotonic()
        after_signature=getattr(self.p._gripper_state,'_model_signature',None)
        self._last_current_timing=dict(rebuilt=rebuilt,
            build_ms=(built-started)*1000.,model_update_ms=(updated-built)*1000.,
            clone_ms=(finished-updated)*1000.,total_ms=(finished-started)*1000.,
            gripper_pose_changed=gripper_before!=getattr(self.p._gripper_state,'_previous',None),
            fixed_pose_changed=fixed_before!=getattr(self.p._fixed_state,'positions',None),
            fixed_changed_joints=','.join(name for name,value in self.p._fixed_state.positions.items()
                if value!=(fixed_before or {}).get(name,value)),
            fixed_max_delta_rad=max((abs(value-(fixed_before or {}).get(name,value))
                for name,value in self.p._fixed_state.positions.items()),default=0.),
            gripper_signature_changed=gripper_signature!=after_signature,
            fixed_signature_changed=fixed_signature!=getattr(self.p._fixed_state,'_model_signature',None),
            model_count=len(after_signature or ()),
            model_entries_changed=sum(a!=b for a,b in zip(gripper_signature or (),after_signature or ()))+
                abs(len(gripper_signature or ())-len(after_signature or ())))
        return state

    def _remember_joints(self, updates):
        """Keep the cached GPU state aligned with virtual_joints. Missing names stay dict-only."""
        state=self.virtual_state
        if state is None or not state.joint_names:
            return
        index={name:i for i,name in enumerate(state.joint_names)}
        position=state.position.detach().reshape(-1).cpu()
        for name,value in updates:
            slot=index.get(name)
            if slot is not None:
                position[slot]=float(value)
        flat=state.position.reshape(-1)
        flat.copy_(position.to(device=flat.device,dtype=flat.dtype))

    def _collision_joints(self):
        values=dict(self.virtual_joints)
        if self._open_collision_joint is not None:
            joint=self._open_collision_joint
            values[joint]=self.p._gripper_state._limits[joint][1]
        return values

    def _world(self,group):
        if self.node.use_pointcloud:
            if self.node._pc_msg is None: raise RuntimeError('抓放任务要求收到有效环境点云')
            age=time.monotonic()-getattr(self.node,'_pc_received',0)
            if not self._world_checked and age>self.node._p('pick_place_cloud_max_age'): raise RuntimeError('初始环境点云过期')
        previous=self.node.require_pointcloud
        self._changed_groups.add(group)
        try:
            self.node.require_pointcloud=True
            if self._prepared_cloud is None:
                error=self.node._sync_world_from_pointcloud(group)
                self._prepared_cloud=getattr(self.node,'_last_prepared_cloud',None)
            else:
                error=self.node._sync_world_from_pointcloud(group,prepared_cloud=self._prepared_cloud)
        finally:
            self.node.require_pointcloud=previous
        if error: raise RuntimeError(error)
        self._world_checked=True
        if not self.node.use_pointcloud:
            self._set_placed(group)
        self.check()

    def _require_separation(self,start,goal,name):
        distance=float(np.linalg.norm(xyz(goal.pose.position)-xyz(start.pose.position)))
        if not .001<=distance<=.5:
            raise ValueError(f'{name}与目标位姿的距离须在1mm至0.5m之间')

    def _target(self,stage):
        name={'prepick':'base_prepick','pick':'base_pick','lift':'base_lift',
              'preplace':'base_preplace','place':'base_place','retreat':'base_retreat'}[stage]
        return getattr(self,name)

    def move(self,task,stage,contact=False):
        for index,waypoint in enumerate(self._route_for_stage(stage)):
            self._move_segment(task,f'{stage}_via_{index+1}',waypoint=waypoint)
        with self._released_object_scope(task,stage):
            self._move_segment(task,stage,contact)

    def _move_segment(self,task,stage,contact=False,waypoint=None):
        if contact:
            return self._interpolate_segment(task,stage)
        self.feedback(stage,f'target={waypoint.kind if waypoint is not None else "pose"} method=opt fcl=keypoints gpu_collision=on')
        started=time.monotonic();group=task.group_name
        current=self._current();state_ready=time.monotonic()
        self._world(group);world_ready=time.monotonic()
        if waypoint is None and stage in ('prepick','preplace'):
            planned=self._plan_preset(task,stage,current)
        elif waypoint is not None and waypoint.kind=='joint':
            goal=group_goal_state(self.p.joint_names,group,waypoint.positions,current)
            planned=self.p.plan_joint(group,current,[goal],max_attempts=3)
        else:
            target=waypoint.pose if waypoint is not None else self._target(stage)
            pose=stamped_to_pose(target,self.p.device_cfg)
            planned=self.p.plan_pose(group,current,[{group+'_ee_link':pose}],max_attempts=3)
        solved=time.monotonic();self.check()
        timing=self.p.last_timing
        self.log('求解计时',
            f"ik_ms={timing.get('ik',0.)*1000:.1f} prm_ms={timing.get('prm',0.)*1000:.1f} "
            f"trajopt_ms={timing.get('trajopt',0.)*1000:.1f} check_ms={timing.get('check',0.)*1000:.1f} "
            f"attempts={int(timing.get('attempts',0))}")
        if not planned.success or planned.trajectory is None: raise RuntimeError(planned.message)
        traj=self._record_motion(planned.trajectory,stage,group)
        self.log('规划完成',f'method=opt points={len(traj.points)} '
            f'state_ms={(state_ready-started)*1000:.1f} world_ms={(world_ready-state_ready)*1000:.1f} '
            f'solve_ms={(solved-world_ready)*1000:.1f} segment_ms={(time.monotonic()-started)*1000:.1f}')

    def _plan_preset(self,task,stage,current):
        pose=stamped_to_pose(self._target(stage),self.p.device_cfg)
        return self.p.plan_pose(task.group_name,current,[{task.group_name+'_ee_link':pose}],max_attempts=3)

    def _interpolate_segment(self,task,stage):
        from curobo_ros_python.planning.cartesian_interpolation import interpolate_cartesian
        self.feedback(stage,'method=cartesian_interpolation ik=on joint_limits=on self_collision=off world_collision=off fcl=off')
        started=time.monotonic();current=self._current()
        state_ready=time.monotonic()
        motion=self.p._sync_locked_arm(task.group_name,current)
        sync_ready=time.monotonic()
        trajectory=self._return_segments.pop(stage,None)
        reused=trajectory is not None
        if trajectory is None:
            trajectory,details=interpolate_cartesian(self.p,motion,current,self._target(stage),
                task.group_name+'_ee_link',self.p.interpolation_dt,self.node.trajopt_velocity_scale,self.check)
        else:
            from curobo_ros_python.planning.cartesian_interpolation import validate_path
            actual=self.p.fk_tool_path(trajectory,task.group_name+'_ee_link')
            goal=self._target(stage).pose;v=goal.position;q=goal.orientation
            validate_path(actual,actual[0],np.array([v.x,v.y,v.z,q.w,q.x,q.y,q.z]),
                          self.p._position_tolerance,self.p._orientation_tolerance)
            details={'ik_samples':0,'ik_calls':0,'lm_calls':0,'refine_calls':0,'ik_calls_total':0}
        path_ready=time.monotonic()
        if stage in ('pick','place'):
            first,last=('prepick','lift') if stage=='pick' else ('preplace','retreat')
            if np.allclose(matrix(self._target(first).pose),matrix(self._target(last).pose),atol=1e-7,rtol=0):
                reverse=trajectory.clone()
                for field,sign in [('position',1),('velocity',-1),('acceleration',1),('jerk',-1)]:
                    values=getattr(reverse,field,None)
                    if values is not None: setattr(reverse,field,values.flip(0)*sign)
                self._return_segments[last]=reverse
        reverse_ready=time.monotonic()
        self.p.last_timing={'interpolation':time.monotonic()-started}
        traj=self._record_motion(trajectory,stage,task.group_name)
        finished=time.monotonic()
        self.log('插补完成',f'ik_samples={details["ik_samples"]} ik_calls={details["ik_calls"]} points={len(traj.points)} '
            f'elapsed_ms={(finished-started)*1000:.1f} collision=off '
            f'lm_calls={details.get("lm_calls",details["ik_calls"])} refine_calls={details.get("refine_calls",0)} '
            f'ik_calls_total={details.get("ik_calls_total",details["ik_calls"])} path_reused={str(reused).lower()}')

        self.log('插补状态分项','clock=host_wall '+ ' '.join(
            f'{name}={str(value).lower() if isinstance(value,bool) else format(value,".3e" if name=="fixed_max_delta_rad" else ".2f") if isinstance(value,float) else value}'
            for name,value in self._last_current_timing.items()))
        self.log('插补分项',f'clock=host_wall state_ms={(state_ready-started)*1000:.2f} '
            f'sync_ms={(sync_ready-state_ready)*1000:.2f} path_ms={(path_ready-sync_ready)*1000:.2f} '
            f'reverse_ms={(reverse_ready-path_ready)*1000:.2f} pack_ms={(finished-reverse_ready)*1000:.2f} '
            f'total_ms={(finished-started)*1000:.2f}')
        timings=details.get('timings_ms')
        if timings is not None:
            self.log('插补计算分项','clock=host_wall '+ ' '.join(f'{name}={value:.2f}' for name,value in timings.items())+
                f' endpoint_cache_hit={str(details["endpoint_cache_hit"]).lower()}'
                f' batch_cache_hit={str(details["batch_cache_hit"]).lower()}')
        refine=details.get('refine_stats',{})
        self.log('备用IK分项',f'calls={refine.get("calls",0)} endpoint_calls={refine.get("endpoint_calls",0)} '
            f'continuation_calls={refine.get("continuation_calls",0)} cache_misses={refine.get("cache_misses",0)} '+
            ' '.join(f'{name}={refine.get(name,0.):.2f}' for name in
                ('setup_ms','sync_ms','solve_ms','select_ms','other_ms','total_ms'))+
            ' included_in=path_ms')

    def merge_continuous_motions(self,group):
        """Keep source timing; open promptly and plan a hold only if needed."""
        started=time.monotonic();arm=GROUP_JOINTS[group]
        merged=[];names=[];pending=self._pending_open;self._open_along={}
        for kind,start,end in continuous_runs(self.trajectory_stages):
            labels=self.trajectory_stages[start:end];chunk=self.trajectories[start:end]
            contexts=list(self._segment_checks[start:end])
            if kind=='gripper':
                merged.extend(chunk);names.extend(labels);continue
            label='+'.join(labels);stops=[]
            prepick_index=labels.index('prepick')+1 if 'prepick' in labels else None
            if len(chunk)>1:
                trajectory,boundaries=self._blend_arm(chunk,arm,group,stops=stops,contacts=[name in ('pick','lift','place','retreat') for name in labels])
            else:
                trajectory=copy.deepcopy(chunk[0])
                boundaries=np.array([0.,trajectory.points[-1].time_from_start.sec+trajectory.points[-1].time_from_start.nanosec*1e-9])
            if pending is not None and prepick_index is not None:
                until=pending['duration'];arrival=float(boundaries[prepick_index]);wait=max(0.,until-arrival)
                if wait>1e-8:
                    trajectory=insert_stationary_hold(trajectory,0.,wait)
                    boundaries[1:]+=wait
                    arrival+=wait
                    boundaries=np.insert(boundaries,1,wait)
                    hold=dict(contexts[0]);hold['stage']='initial_open_wait'
                    contexts.insert(0,hold)
                if until>0.:
                    trajectory=overlay_joint_motion(trajectory,pending['joint'],pending['start'],pending['target'],until)
                    self._open_along[label]=until
                self.log('夹爪自动张开',f'open_s={until:.3f} prepick_arrival_s={arrival:.3f} initial_wait_s={wait:.3f} feedback_wait=false')
                pending=None
            through=[float(boundaries[index]) for index,context in enumerate(contexts[:-1],1)
                     if context['stage'] in ('prepick','preplace','lift','retreat')]
            if through:
                _,_,velocity,_,_=sample_trajectory(trajectory,sample_times=through)
                columns=[trajectory.joint_names.index(name) for name in arm]
                if np.any(np.linalg.norm(velocity[:,columns],axis=1)<1e-5):
                    raise RuntimeError('预置点或提升/撤退位无法连续通过')
            self._validate_merged(group,trajectory,contexts,boundaries)
            merged.append(trajectory);names.append(label)
        if pending is not None: raise RuntimeError('张开轨迹没有对应的预抓取阶段')
        combined=concatenate_segments(merged);self._validated_samples(group,combined)
        self.trajectories=merged;self.trajectory_stages=names
        self.log('合并验收完成',f'segments={len(merged)} points={len(combined.points)} elapsed_ms={(time.monotonic()-started)*1000:.1f}')
        return combined

    def _dynamic_limits(self,group):
        return cpu_dynamic_limits(self.p._planners[group].trajopt_solver.transition_model)

    def _blend_arm(self,chunk,arm,group,stops=(),contacts=None):
        paths=[];clocks=[];velocities=[];accelerations=[]
        for traj in chunk:
            index=[traj.joint_names.index(name) for name in arm]
            paths.append(np.array([[point.positions[i] for i in index] for point in traj.points],dtype=np.float64))
            clocks.append(np.array([point.time_from_start.sec+point.time_from_start.nanosec*1e-9 for point in traj.points]))
            velocities.append(np.array([[point.velocities[i] if point.velocities else 0. for i in index] for point in traj.points]))
            accelerations.append(np.array([[point.accelerations[i] if point.accelerations else 0. for i in index] for point in traj.points]))
        # Segment messages already include trajopt_velocity_scale. Do not apply it again.
        piece,times,boundaries=blend_through(paths,arm,self.p.interpolation_dt,self._dynamic_limits(group),1.,timings=clocks,velocities=velocities,accelerations=accelerations,stops=stops,contacts=contacts,metadata=True)
        full_names=list(chunk[0].joint_names)
        held=np.asarray(chunk[0].points[0].positions,dtype=np.float64)
        arm_index=[full_names.index(name) for name in arm]
        position=np.repeat(held.reshape(1,-1),len(piece.position),axis=0)
        velocity=np.zeros_like(position);acceleration=np.zeros_like(position)
        host=host_state_arrays(piece.position,piece.velocity,piece.acceleration)
        position[:,arm_index]=host[0]
        velocity[:,arm_index]=host[1]
        acceleration[:,arm_index]=host[2]
        merged=numpy_traj_to_msg(full_names,position,velocity,dt=self.p.interpolation_dt,velocity_scale=1.)
        for point,row,t in zip(merged.points,acceleration,times):
            point.accelerations=row.tolist();point.time_from_start=ros_duration(round(float(t)*1e9))
        for index,boundary in enumerate(boundaries[1:-1],1):
            nearest=int(np.argmin(np.abs(times-boundary)))
            speed=float(np.linalg.norm(velocity[nearest,arm_index]))
            self.log('路点通过速度',f'boundary={index} time_s={boundary:.3f} joint_speed_norm={speed:.6f}rad/s planned_stop={str(index in stops).lower()}')
        merged.header.frame_id=self.node.base_frame
        return merged,boundaries

    def _validated_samples(self,group,trajectory):
        """Check the exported controller curve, including close/release joins."""
        times,q,v,a,j=sample_trajectory(trajectory)
        columns=[trajectory.joint_names.index(name) for name in GROUP_JOINTS[group]]
        for values,limit,order in zip((v,a,j),self._dynamic_limits(group),(1,2,3)):
            if np.any(np.abs(values[:,columns])>limit*1.015+1e-5):
                raise RuntimeError(f'合并轨迹超过 {order} 阶关节动态限位')
        for joint,limit in self.p._gripper_state._velocity_limits.items():
            if joint in trajectory.joint_names:
                column=trajectory.joint_names.index(joint)
                low,high=self.p._gripper_state._limits[joint]
                if np.max(np.abs(v[:,column]))>limit*1.015 or np.min(q[:,column])<low-1e-6 or np.max(q[:,column])>high+1e-6:
                    raise RuntimeError(f'合并轨迹夹爪 {joint} 超限')
        return times,q

    def _validate_merged(self,group,trajectory,contexts,boundaries):
        """Recheck the actual controller spline against each original phase's scene."""
        import torch
        from curobo.types import JointState
        from curobo_ros_python.planning.cartesian_interpolation import validate_path
        from curobo._src.geom.types import SceneCfg
        if len(contexts)!=len(boundaries)-1 or any(item is None for item in contexts):
            raise RuntimeError('合并轨迹缺少原始阶段验收信息')
        times,q=self._validated_samples(group,trajectory)
        columns=[trajectory.joint_names.index(name) for name in GROUP_JOINTS[group]]
        motion=self.p._planners[group];checker=self.p._mesh_checker
        saved_world=motion.scene_collision_checker.scene_model
        saved_scene=self.p._mesh_scenes[group];saved_environment=checker.environment
        saved_payload=checker.payload
        bounds=motion.kinematics.get_joint_limits().position.detach().cpu().numpy()
        try:
            for index,context in enumerate(contexts):
                self.check()
                mask=(times>=boundaries[index]-1e-8)&(times<=boundaries[index+1]+1e-8)
                positions=q[mask][:,columns]
                state=JointState.from_position(torch.as_tensor(positions,device=self.p.device,dtype=torch.float32),joint_names=GROUP_JOINTS[group])
                values=context['joints']
                self.p.update_gripper_state(values)
                current,_=joint_map_to_curobo(self.p.joint_names,values,self.p.device,default_position=self.p.default_joint_position)
                self.p._sync_locked_arm(group,current)
                if np.any(positions<bounds[0]-1e-6) or np.any(positions>bounds[1]+1e-6):
                    raise RuntimeError(f"{context['stage']} 合并后超过关节位置限位")
                actual=None
                keep_vertical=getattr(self.p,'keep_tool_z_vertical',False)
                vertical_debug=keep_vertical and self.node.get_logger().is_enabled_for(LoggingSeverity.DEBUG)
                if context['contact'] or vertical_debug:
                    actual=self.p.fk_tool_path(state,group+'_ee_link')
                if vertical_debug:
                    quaternion=actual[:,3:]
                    z=1.-2.*(quaternion[:,1]**2+quaternion[:,2]**2)
                    tilt=np.degrees(np.arccos(np.clip(np.abs(z),0.,1.)))
                    self.log('工具Z竖直检查',f"source_stage={context['stage']} mean_tilt_deg={tilt.mean():.2f} max_tilt_deg={tilt.max():.2f} mode=soft yaw=free",level='debug')
                if context['contact']:
                    validate_path(actual,context['line_start'],context['line_end'],self.p._position_tolerance,self.p._orientation_tolerance)
                else:
                    if checker.payload is not context['payload']: set_payload(self.p,context['payload'])
                    # Captured phase worlds are read-only during this GPU-locked check.
                    # Joint geometry and payload are synchronized independently above.
                    if motion.scene_collision_checker.scene_model is not context['world']:
                        motion.update_world(context['world'] if context['world'] is not None else SceneCfg())
                    checker.environment=context['scene']
                    if not motion._joint_traj_is_collision_free(state):
                        raise RuntimeError(f"{context['stage']} 合并后 GPU 碰撞验收失败")
                    for row in (positions[0],positions[-1]):
                        keypoint=dict(values);keypoint.update(zip(GROUP_JOINTS[group],row))
                        collision=checker.check(keypoint)
                        if collision is not None: raise RuntimeError(f"{context['stage']} 合并后 FCL 关键点碰撞: {collision}")
        finally:
            if motion.scene_collision_checker.scene_model is not saved_world:
                motion.update_world(saved_world if saved_world is not None else SceneCfg())
            self.p._mesh_scenes[group]=saved_scene;checker.environment=saved_environment
            if checker.payload is not saved_payload: set_payload(self.p,saved_payload)
            self.p.update_gripper_state(self._collision_joints())

    def _record_motion(self,trajectory,stage,group):
        """Pack one stopped segment and advance the offline full joint state."""
        full=expand_joint_trajectory(trajectory,self.virtual_joints,self.joint_names)
        host=host_state_arrays(full.position,full.velocity,full.acceleration)
        if full.velocity is None and full.acceleration is None:
            positions,velocities=host,None
        else:
            positions=host[0]
            velocities=None if full.velocity is None else host[1]
        traj=numpy_traj_to_msg(self.joint_names,positions,velocities,dt=self.p.interpolation_dt,
                              velocity_scale=self.node.trajopt_velocity_scale)
        if full.acceleration is not None:
            acceleration=host[-1]*self.node.trajopt_velocity_scale**2
            for point,row in zip(traj.points,acceleration): point.accelerations=row.tolist()
        traj.header.frame_id=self.node.base_frame
        contact=stage in ('pick','lift','place','retreat')
        context={'stage':stage,'contact':contact,'joints':self._collision_joints(),
                 'world':self.p._planners[group].scene_collision_checker.scene_model,
                 'scene':self.p._mesh_scenes[group],'payload':self.p._mesh_checker.payload}
        if contact:
            from curobo.types import JointState
            first=JointState.from_position(trajectory.position.reshape(-1,trajectory.position.shape[-1])[:1],joint_names=trajectory.joint_names)
            context['line_start']=self.p.fk_tool_path(first,group+'_ee_link')[0]
            goal=self._target(stage).pose;v=goal.position;q=goal.orientation
            context['line_end']=np.array([v.x,v.y,v.z,q.w,q.x,q.y,q.z])
        self._segment_checks.append(context)
        self.trajectories.append(traj);self.trajectory_stages.append(stage)
        # Only this arm moved. Keep feedback-only joints at their original
        # precision; the float32 output must not look like new head/body feedback.
        # Gripper state is advanced explicitly by gripper(), not by arm output.
        arm=set(GROUP_JOINTS[group])
        finished=[(name,value) for name,value in zip(self.joint_names,traj.points[-1].positions)
                  if name in arm]
        self.virtual_joints.update(finished)
        self._remember_joints(finished)
        return traj

    def gripper(self,task,position,stage):
        name=self._validate_gripper_target(task,position)
        initial=float(self.virtual_joints[name]);target=float(position)
        duration=self.p._gripper_state.motion_duration(name,initial,target,self.p.interpolation_dt)
        wait=0. if stage=='open' else self.gripper_wait_time
        self.feedback(stage,f'joint_position={target:.6f} auto_duration_s={duration:.3f} settle_wait_s={wait:.3f} feedback_wait=false')
        if stage=='open':
            self._pending_open={'joint':name,'start':initial,'target':target,'duration':duration}
            self._open_collision_joint=name
        else:
            duration=max(duration,self.p.interpolation_dt)
            traj=gripper_segment(self.joint_names,self.virtual_joints,name,target,duration,self.p.interpolation_dt)
            if wait>0.:
                traj=insert_stationary_hold(traj,duration,wait)
            traj.header.frame_id=self.node.base_frame
            self._segment_checks.append(None)
            self.trajectories.append(traj);self.trajectory_stages.append(stage)
            self._open_collision_joint=None
        self.virtual_joints[name]=target
        self.last_gripper_position=target
        self._remember_joints(((name,target),))
        self.check()

    def attach(self,task):
        self.feedback('attach_pending')
        current=self._current()
        tool_matrix=self._tool_matrix(current,task.group_name+'_ee_link')
        offset=self.object_in_tool
        payload=Payload(task.group_name,self.object_size,offset,tool_matrix@offset)
        payload.close_position=float(task.close_position)
        payload.cloud_seq_at_attach=self.node._pc_seq
        set_payload(self.p,None)
        self.server.held=payload
        self.server.preview_new_payload=True
        self._set_placed(task.group_name)
        self.server.invalidate_world()

    def _set_placed(self,group=None):
        """Overlay placed boxes once, rebuilding GPU/FCL only on geometry changes."""
        from curobo._src.geom.types import SceneCfg
        self.p._placed_objects=dict(self.server.placed)
        placed=list(self.server.placed.values())
        def geometry(boxes):
            return sorted((box.name,tuple(box.pose),tuple(box.dims)) for box in boxes if box.name.startswith('placed_'))
        for name in ([group] if group is not None else self.p._planners):
            motion=self.p._planners[name]
            old=motion.scene_collision_checker.scene_model
            if old is not None and geometry(old.cuboid)==geometry(old.objects)==geometry(placed) and name in self.p._mesh_scenes:
                continue
            scene=copy.deepcopy(old) if old is not None else SceneCfg()
            # Rebuild objects as well as cuboid: FCL reads the objects list.
            scene=replace(scene,cuboid=[obj for obj in scene.cuboid if not obj.name.startswith('placed_')]+placed,
                          objects=None)
            prepared=self.p._mesh_checker.prepare_scene(scene)
            self._changed_groups.add(name)
            motion.update_world(scene)
            self.p._mesh_scenes[name]=prepared

    def _tool_matrix(self,current,tool):
        import torch
        pose=self.p.fk_tool_poses(current)[tool]
        pq=torch.cat((pose.position.detach().reshape(-1)[:3], pose.quaternion.detach().reshape(-1)[:4])).cpu().numpy()
        return pose_matrix(pq[:3], pq[3:])

    def detach(self,task):
        self.feedback('detach')
        from curobo._src.geom.types import Cuboid
        from scipy.spatial.transform import Rotation
        held=self.server.held
        current=self._current()
        transform=self._tool_matrix(current,held.tool)@held.in_tool
        q=Rotation.from_matrix(transform[:3,:3]).as_quat()
        key=f'placed_{self.server.placed_seq}'
        self.server.placed_seq+=1
        box=Cuboid(name=key,pose=[*transform[:3,3],q[3],q[0],q[1],q[2]],dims=held.size.tolist())
        self.released_box=(transform,held.size)
        self.released_key=key
        self.server.placed[key]=box
        if self.server.preview_new_payload:
            self.server.preview_exclusions.append((held.origin,held.size))
        self._set_placed(task.group_name)
        set_payload(self.p,None)
        self.server.held=None
        self.server.preview_new_payload=False
        self.server.invalidate_world()
        self.log('释放建模完成',f'placed_objects={len(self.server.placed)} retreat_object_collision=ignore end_object_collision=check')

    def cleanup(self):
        self.server.held=self.saved_held
        self.server.placed=self.saved_placed
        self.server.preview_new_payload,self.server.preview_exclusions=self.saved_preview
        # Restore the saved GPU/FCL scenes below; rebuilding the preview's
        # worlds first would immediately discard that work.
        self.p._placed_objects=dict(self.saved_placed)
        set_payload(self.p,self.saved_held)
        self.p.update_gripper_state(dict(self.node._joint_map))
        from curobo._src.geom.types import SceneCfg
        for group in self._changed_groups:
            scene=self.saved_worlds[group]
            self.p._planners[group].update_world(scene if scene is not None else SceneCfg())
        self.p._mesh_scenes=self.saved_mesh_scenes
        self.p._mesh_checker.environment=self.saved_fcl_environment
        self.server.invalidate_world()
