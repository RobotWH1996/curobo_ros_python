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
from curobo_ros_interfaces.action import Pick, Place, PickPlace
from curobo_ros_python.planning.manipulation import ManipulationWorkflow, TaskCancelled
from curobo_ros_python.planning.geom import quat_mul_wxyz, transform_pose_wxyz
from curobo_ros_python.planning.payload import Payload, pose_matrix, outside_box, resolve_object_geometry
from curobo_ros_python.planning.pick_collision import set_payload
from curobo_ros_python.planning.convert import stamped_to_pose, numpy_traj_to_msg, group_goal_state, joint_map_to_curobo
from curobo_ros_python.planning.offline_trajectory import gripper_segment, concatenate_segments
from curobo.trajectory_optimizer import expand_joint_trajectory
from curobo_ros_python.planning.groups import GROUP_JOINTS
from curobo_ros_python.planning.trajectory import select_trajectory_joints


DEFAULTS={
    'pick_place_default_object_size':[0.0,0.0,0.0],
    'pick_place_timeout':180.,
    'pick_place_joint_state_max_age':1.,
    'pick_place_cloud_max_age':2.,
    'pick_place_path_tolerance':.01,
    'pick_place_max_placed_objects':16,
}


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
    if not math.isfinite(task.gripper_duration) or task.gripper_duration<=0:
        raise ValueError('gripper_duration 必须为有限正秒数')


class PickPlaceServer:
    def __init__(self,node):
        self.node=node
        for name in ('pick_place_timeout','pick_place_joint_state_max_age',
                     'pick_place_cloud_max_age','pick_place_path_tolerance'):
            value=float(node._p(name))
            if not math.isfinite(value) or value<=0: raise ValueError(f'{name} 必须是有限正数')
        if not 1<=int(node._p('pick_place_max_placed_objects'))<=16:
            raise ValueError('pick_place_max_placed_objects 必须在1到16之间')
        self.lock=threading.Lock()
        self.busy=False
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
                cancel_callback=lambda handle:CancelResponse.ACCEPT,
                execute_callback=lambda handle,k=name,a=action:self.execute(k,a,handle),
                callback_group=node._action_group))

    def accept(self,kind,request):
        try:
            timeout=float(request.timeout)
            if not math.isfinite(timeout) or timeout<0: raise ValueError('timeout必须为有限非负数')
            tasks=(request.pick,request.place) if kind=='pick_place' else (request.task,)
            for task in tasks: validate_common(task)
            if kind=='pick_place' and request.pick.group_name!=request.place.group_name:
                raise ValueError('PickPlace 的抓取和放置必须使用同一手臂')
            if self.node.execute_trajectory and self.node.executor_helper is None:
                raise ValueError('当前启动配置启用了执行，但轨迹执行器尚未初始化')
            if self.node._mpc_following: raise RuntimeError('MPC跟踪中，请先取消')
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
            runner=ManipulationWorkflow(backend)
            if kind=='pick': runner.pick(handle.request.task)
            elif kind=='place': runner.place(handle.request.task)
            else: runner.pick_place(handle.request.pick,handle.request.place)
            backend.check()
            result.trajectory=concatenate_segments(backend.trajectories)
            result.trajectory.header.frame_id=self.node.base_frame
            result.trajectory.header.stamp=self.node.get_clock().now().to_msg()
            self.node._traj_pub.publish(result.trajectory)
            if self.node.execute_trajectory:
                backend.execute_plan(task.group_name)
                result.execution_completed=True
            result.success=True
            result.message='完整轨迹规划及控制器执行成功' if result.execution_completed else '完整轨迹已发布；当前启动配置不下发控制器'
        except TaskCancelled as exc:
            result.message=str(exc)
        except Exception as exc:
            result.message=f'{type(exc).__name__}: {exc}'
        finally:
            if backend is not None:
                result.stage=backend.stage
                result.trajectories=backend.trajectories
                result.trajectory_stages=backend.trajectory_stages
                try:
                    backend.cleanup()
                except Exception as exc:
                    result.success=False
                    result.message+=f'；恢复规划状态失败: {exc}'
            if held_gpu: self.node._gpu_lock.release()
            with self.lock: self.busy=False
        status='成功' if result.success else '取消' if handle.is_cancel_requested else '失败'
        message=(f'{context} event=结束 status={status} stage={result.stage or "validate"} '
            f'trajectories={len(result.trajectories)} '
            f'elapsed_ms={(time.monotonic()-started)*1000:.1f} reason={result.message!r}')
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
        timeout=float(handle.request.timeout) or float(self.node._p('pick_place_timeout'))
        self.deadline=time.monotonic()+timeout
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
        self._world_checked=False
        self._prepared_cloud=None
        self._return_segments={}
        self._changed_groups=set()
        self.virtual_joints=dict(self.node._joint_map)
        self.joint_names=[j.get('name') for j in self.p._mesh_checker.ordered
                          if j.get('type')!='fixed' and j.find('mimic') is None]
        self.released_box=None
        self.last_gripper_position=0.

    def check(self):
        if self.handle.is_cancel_requested: raise TaskCancelled('抓放任务已取消')
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

    def execute_plan(self,group):
        """Dispatch only after every task stage has been planned successfully."""
        executor=self.node.executor_helper
        for stage,full in zip(self.trajectory_stages,self.trajectories):
            self.feedback('execute_'+stage)
            self.check()
            gripper=stage in ('open','close','release')
            controller_group=group.split('_')[0]+'_gripper' if gripper else group
            names=[controller_group] if gripper else GROUP_JOINTS[group]
            trajectory=select_trajectory_joints(full,names)
            timeout=min(self.node.execute_timeout,self.deadline-time.monotonic())
            if timeout<=0: raise TimeoutError('任务执行超时')
            ok,message=executor.execute(controller_group,trajectory,timeout_sec=timeout,
                cancel_check=lambda:self.handle.is_cancel_requested or not rclpy.ok() or time.monotonic()>self.deadline)
            if not ok:
                self.check()
                raise RuntimeError(f'{stage} 控制器执行失败: {message}')
            self.log('执行完成',f'controller={controller_group} points={len(trajectory.points)}')

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
        if self.virtual_state is None:
            self.virtual_state,_=joint_map_to_curobo(self.p.joint_names,self.virtual_joints,self.p.device,
                                                    default_position=self.p.default_joint_position)
        self.p.update_gripper_state(self.virtual_joints)
        return self.virtual_state.clone()

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
        if not planned.success or planned.trajectory is None: raise RuntimeError(planned.message)
        traj=self._record_motion(planned.trajectory,stage)
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
        motion=self.p._sync_locked_arm(task.group_name,current)
        trajectory=self._return_segments.pop(stage,None)
        if trajectory is None:
            trajectory,details=interpolate_cartesian(self.p,motion,current,self._target(stage),
                task.group_name+'_ee_link',self.p.interpolation_dt,self.node.trajopt_velocity_scale,self.check)
        else:
            from curobo_ros_python.planning.cartesian_interpolation import validate_path
            actual=self.p.fk_tool_path(trajectory,task.group_name+'_ee_link')
            goal=self._target(stage).pose;v=goal.position;q=goal.orientation
            validate_path(actual,actual[0],np.array([v.x,v.y,v.z,q.w,q.x,q.y,q.z]),
                          self.p._position_tolerance,self.p._orientation_tolerance)
            details={'ik_samples':0,'ik_calls':0}
        if stage in ('pick','place'):
            first,last=('prepick','lift') if stage=='pick' else ('preplace','retreat')
            if np.allclose(matrix(self._target(first).pose),matrix(self._target(last).pose),atol=1e-7,rtol=0):
                reverse=trajectory.clone()
                for field,sign in [('position',1),('velocity',-1),('acceleration',1),('jerk',-1)]:
                    values=getattr(reverse,field,None)
                    if values is not None: setattr(reverse,field,values.flip(0)*sign)
                self._return_segments[last]=reverse
        self.p.last_timing={'interpolation':time.monotonic()-started}
        traj=self._record_motion(trajectory,stage)
        self.log('插补完成',f'ik_samples={details["ik_samples"]} ik_calls={details["ik_calls"]} points={len(traj.points)} '
            f'elapsed_ms={(time.monotonic()-started)*1000:.1f} collision=off')

    def _record_motion(self,trajectory,stage):
        """Pack one stopped segment and advance the offline full joint state."""
        full=expand_joint_trajectory(trajectory,self.virtual_joints,self.joint_names)
        positions=full.position.detach().cpu().numpy().reshape(-1,len(self.joint_names))
        velocities=None if full.velocity is None else full.velocity.detach().cpu().numpy().reshape(positions.shape)
        traj=numpy_traj_to_msg(self.joint_names,positions,velocities,dt=self.p.interpolation_dt,
                              velocity_scale=self.node.trajopt_velocity_scale)
        if full.acceleration is not None:
            acceleration=full.acceleration.detach().cpu().numpy().reshape(positions.shape)*self.node.trajopt_velocity_scale**2
            for point,row in zip(traj.points,acceleration): point.accelerations=row.tolist()
        traj.header.frame_id=self.node.base_frame
        self.trajectories.append(traj);self.trajectory_stages.append(stage)
        self.virtual_joints.update(zip(self.joint_names,traj.points[-1].positions))
        self.virtual_state=None
        return traj

    def gripper(self,task,position,stage):
        self.feedback(stage,f'joint_position={position:.6f} duration_s={task.gripper_duration:.3f}')
        name=self._validate_gripper_target(task,position)
        self._current()
        traj=gripper_segment(self.joint_names,self.virtual_joints,name,float(position),
                             float(task.gripper_duration),self.p.interpolation_dt)
        traj.header.frame_id=self.node.base_frame
        self.trajectories.append(traj);self.trajectory_stages.append(stage)
        self.virtual_joints.update(zip(self.joint_names,traj.points[-1].positions))
        self.last_gripper_position=float(position)
        self.virtual_state=None
        self.p.update_gripper_state(self.virtual_joints)
        self.log('夹爪轨迹完成',f'points={len(traj.points)} command_sent=false')
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
        pose=self.p.fk_tool_poses(current)[tool]
        return pose_matrix(pose.position.detach().cpu().numpy().reshape(-1,3)[0],
                           pose.quaternion.detach().cpu().numpy().reshape(-1,4)[0])

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
