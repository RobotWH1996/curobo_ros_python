"""规划结果转 JointTrajectory，以及 FollowJointTrajectory 执行。"""

from __future__ import annotations

import time
import threading
import copy
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration, Time
from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import SwitchController
from rclpy.action import ActionClient
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from curobo_ros_python.planning.groups import GROUP_ACTION, GROUP_CONTROLLER_SWITCH, GROUP_JOINTS


def _as_2d(arr) -> np.ndarray:
    data = np.asarray(arr, dtype=np.float64)
    while data.ndim > 2:
        data = data[0]
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def _duration(t: float) -> Duration:
    sec = int(t)
    nsec = int(round((t - sec) * 1e9))
    if nsec >= 1_000_000_000:
        sec += 1
        nsec -= 1_000_000_000
    return Duration(sec=sec, nanosec=max(nsec, 0))


def numpy_traj_to_msg(
    joint_names: Sequence[str],
    positions: np.ndarray,
    velocities: Optional[np.ndarray] = None,
    dt: float = 0.02,
    velocity_scale: float = 1.0,
) -> JointTrajectory:
    """positions: (T, dof)。velocity_scale < 1 会把时间拉长。"""
    pos = _as_2d(positions)
    vel = None if velocities is None else _as_2d(velocities)
    n_pts = pos.shape[0]
    scale = max(float(velocity_scale), 1e-6)
    time_scale = 1.0 / scale

    msg = JointTrajectory()
    msg.joint_names = list(joint_names)
    for i in range(n_pts):
        pt = JointTrajectoryPoint()
        pt.positions = pos[i].tolist()
        if vel is not None and vel.shape[0] == n_pts:
            pt.velocities = (vel[i] * scale).tolist()
        t = float(i) * float(dt) * time_scale
        pt.time_from_start = _duration(t)
        msg.points.append(pt)
    return msg


def joint_map_from_msg(names: Iterable[str], positions: Iterable[float]) -> Dict[str, float]:
    return {n: float(p) for n, p in zip(names, positions)}


def select_trajectory_joints(trajectory: JointTrajectory, names: Sequence[str]) -> JointTrajectory:
    """Select controller columns while preserving the planned stage timing."""
    missing = [name for name in names if name not in trajectory.joint_names]
    if missing:
        raise ValueError(f'轨迹缺少控制器关节: {missing}')
    indices = [trajectory.joint_names.index(name) for name in names]
    result = JointTrajectory(joint_names=list(names))
    result.header = copy.deepcopy(trajectory.header)
    for point in trajectory.points:
        out = JointTrajectoryPoint(time_from_start=copy.deepcopy(point.time_from_start))
        for field in ('positions', 'velocities', 'accelerations', 'effort'):
            values = getattr(point, field)
            if values:
                setattr(out, field, [values[index] for index in indices])
        result.points.append(out)
    return result


def combine_joint_trajectories(
    left: JointTrajectory,
    right: JointTrajectory,
    joint_names: Optional[Sequence[str]] = None,
) -> JointTrajectory:
    names = list(joint_names or GROUP_JOINTS.get("both_arms") or [])
    n = min(len(left.points), len(right.points))
    msg = JointTrajectory()
    msg.joint_names = names
    if n <= 0 or not names:
        return msg
    left_idx = {name: i for i, name in enumerate(left.joint_names)}
    right_idx = {name: i for i, name in enumerate(right.joint_names)}
    for k in range(n):
        lp, rp = left.points[k], right.points[k]
        pt = JointTrajectoryPoint()
        have_vel = bool(lp.velocities) and bool(rp.velocities)
        pos: List[float] = []
        vel: List[float] = []
        for name in names:
            if name in left_idx:
                src, idx = lp, left_idx[name]
            else:
                src, idx = rp, right_idx[name]
            pos.append(float(src.positions[idx]))
            if have_vel:
                vel.append(float(src.velocities[idx]))
        pt.positions = pos
        if have_vel:
            pt.velocities = vel
        pt.time_from_start = lp.time_from_start
        msg.points.append(pt)
    return msg


def trajectory_is_finite(traj: Optional[JointTrajectory]) -> bool:
    """位置、速度、加速度都是有限数才可以下发。"""
    if traj is None or not traj.points:
        return False
    for pt in traj.points:
        values = list(pt.positions) + list(pt.velocities) + list(pt.accelerations)
        if not values:
            return False
        if not np.isfinite(np.asarray(values, dtype=np.float64)).all():
            return False
    return True


def copy_joint_trajectory(src: JointTrajectory) -> JointTrajectory:
    out = JointTrajectory()
    out.joint_names = list(src.joint_names)
    out.header.frame_id = src.header.frame_id
    out.header.stamp = src.header.stamp
    for src_pt in src.points:
        pt = JointTrajectoryPoint()
        pt.positions = list(src_pt.positions)
        pt.velocities = list(src_pt.velocities)
        pt.accelerations = list(src_pt.accelerations)
        pt.effort = list(src_pt.effort)
        pt.time_from_start = src_pt.time_from_start
        out.points.append(pt)
    return out


def _point_time(pt: JointTrajectoryPoint) -> float:
    return float(pt.time_from_start.sec) + float(pt.time_from_start.nanosec) * 1e-9


def drop_passed_prefix(
    trajectory: JointTrajectory,
    skip_sec: float,
    min_keep_sec: float,
) -> Tuple[JointTrajectory, float]:
    """丢掉开头已经落后于手臂的那段，剩下的时间从 0 重新计。

    至少留下 min_keep_sec，避免把指令裁得比下一拍还短，手臂再次跑完停住。
    """
    if trajectory is None or len(trajectory.points) < 2:
        return trajectory, 0.0
    times = [_point_time(pt) for pt in trajectory.points]
    duration = times[-1]
    skip = min(max(float(skip_sec), 0.0), max(0.0, duration - float(min_keep_sec)))
    if skip <= 1e-4:
        return trajectory, 0.0
    kept = [pt for pt, t in zip(trajectory.points, times) if t + 1e-9 >= skip]
    if len(kept) < 2:
        return trajectory, 0.0
    origin = _point_time(kept[0])
    out = JointTrajectory()
    out.joint_names = list(trajectory.joint_names)
    out.header.frame_id = trajectory.header.frame_id
    for src in kept:
        pt = JointTrajectoryPoint()
        pt.positions = list(src.positions)
        pt.velocities = list(src.velocities)
        pt.time_from_start = _duration(_point_time(src) - origin)
        out.points.append(pt)
    return out, origin


def shift_trajectory_time(trajectory: JointTrajectory, offset_sec: float) -> JointTrajectory:
    """把命令序列整体往后挪 offset 秒。

    MPC 的 action_sequence 对应的是「当前状态之后一个 knot」开始的那几步，
    留出这段空档让控制器从实际位置平滑插到第一个命令点，避免起步就是台阶。
    """
    out = JointTrajectory()
    out.joint_names = list(trajectory.joint_names)
    out.header.frame_id = trajectory.header.frame_id
    offset = max(float(offset_sec), 0.0)
    for src in trajectory.points:
        pt = JointTrajectoryPoint()
        pt.positions = list(src.positions)
        pt.velocities = list(src.velocities)
        pt.time_from_start = _duration(_point_time(src) + offset)
        out.points.append(pt)
    return out


def _wait_future(future, timeout_sec: float, cancel_check=None) -> bool:
    deadline = time.monotonic() + timeout_sec
    while not future.done():
        if cancel_check is not None and cancel_check():
            return False
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.01)
    return True


class TrajectoryExecutor:
    def __init__(self, node: Node, switch_controllers: bool = True):
        self._node = node
        self._switch = switch_controllers
        self._actions = {**GROUP_ACTION,
                'left_gripper': '/left_gripper_controller/follow_joint_trajectory',
                'right_gripper': '/right_gripper_controller/follow_joint_trajectory'}
        self._clients = {group: ActionClient(node, FollowJointTrajectory, action)
                         for group, action in self._actions.items()}
        self._switch_client = node.create_client(
            SwitchController, "/controller_manager/switch_controller"
        )
        self._last_group: Optional[str] = None
        self._handles: Dict[str, object] = {}
        self._send_futs: List[object] = []
        self._ready_groups: set = set()
        self._unresolved_execution = None
        self._execution_fault = False
        self._awaiting_goal_ack = False
        self._parallel_run = None

    @property
    def execution_pending(self) -> bool:
        future = self._unresolved_execution
        if future is not None and future.done() and not self._awaiting_goal_ack:
            self._confirm_terminal(future)
        run = getattr(self, '_parallel_run', None)
        return (self._execution_fault or self._awaiting_goal_ack
                or (future is not None and not future.done())
                or (run is not None and not run.done()))

    def _confirm_terminal(self, future):
        try:
            wrapped=future.result()
            if wrapped is None or getattr(wrapped,'status',None) not in (
                    GoalStatus.STATUS_SUCCEEDED,GoalStatus.STATUS_CANCELED,GoalStatus.STATUS_ABORTED):
                self._execution_fault=True
        except Exception:
            self._execution_fault=True

    def _track_result(self, future):
        self._unresolved_execution=future
        future.add_done_callback(self._confirm_terminal)

    def switch_group(self, group_name: str, timeout_sec: float = 3.0) -> bool:
        if group_name not in GROUP_CONTROLLER_SWITCH:
            return True
        if not self._switch:
            self._last_group = group_name
            return True
        if group_name == self._last_group:
            return True
        if not self._switch_client.wait_for_service(timeout_sec=2.0):
            self._node.get_logger().warn("switch_controller 服务不可用，跳过切换")
            self._last_group = group_name
            return True
        activate, deactivate = GROUP_CONTROLLER_SWITCH[group_name]
        req = SwitchController.Request()
        req.activate_controllers = activate
        req.deactivate_controllers = deactivate
        req.strictness = SwitchController.Request.BEST_EFFORT
        future = self._switch_client.call_async(req)
        if not _wait_future(future, timeout_sec):
            self._node.get_logger().error("切换控制器超时")
            return False
        try:
            resp = future.result()
        except Exception as exc:  # noqa: BLE001
            self._node.get_logger().error(f"切换控制器异常: {exc}")
            return False
        if resp is None or not resp.ok:
            self._node.get_logger().error("切换控制器失败")
            return False
        self._last_group = group_name
        self._node.get_logger().info(f"控制器已切换到 {group_name}")
        return True

    def send_replace(self, group_name: str, trajectory: JointTrajectory) -> None:
        """异步下发并顶替上一拍，不等待执行结束。"""
        if self.execution_pending:
            self._node.get_logger().warn("上一条执行尚未确认停止，拒绝MPC顶替")
            return
        if trajectory is None or not trajectory.points:
            return
        client = self._clients.get(group_name)
        if client is None:
            self._node.get_logger().error(f"未知规划组: {group_name}")
            return
        if not self.switch_group(group_name):
            self._node.get_logger().warn(
                "MPC 控制器切换失败，本周期不下发", throttle_duration_sec=2.0
            )
            return
        if group_name not in self._ready_groups:
            if not client.server_is_ready() and not client.wait_for_server(timeout_sec=0.05):
                self._node.get_logger().warn(
                    f"Action 不可用: {self._actions[group_name]}",
                    throttle_duration_sec=2.0,
                )
                return
            self._ready_groups.add(group_name)
        elif not client.server_is_ready():
            self._ready_groups.discard(group_name)
            self._node.get_logger().warn(
                f"Action 已断开: {self._actions[group_name]}，本周期跳过",
                throttle_duration_sec=2.0,
            )
            return
        last_t = trajectory.points[-1].time_from_start
        dur = float(last_t.sec) + float(last_t.nanosec) * 1e-9
        self._node.get_logger().info(
            f"MPC 下发 {group_name}: {len(trajectory.points)} 点, {dur:.3f} s",
            throttle_duration_sec=2.0,
        )
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.joint_names = list(trajectory.joint_names)
        goal.trajectory.header.frame_id = trajectory.header.frame_id
        goal.trajectory.points = list(trajectory.points)
        fut = client.send_goal_async(goal)
        self._send_futs.append(fut)
        if len(self._send_futs) > 8:
            self._send_futs = self._send_futs[-4:]
        fut.add_done_callback(lambda f, g=group_name: self._on_goal_response(g, f))

    def send_mpc(
        self,
        left: Optional[JointTrajectory],
        right: Optional[JointTrajectory],
        start_delay: float = 0.0,
    ) -> None:
        if left is not None:
            left = shift_trajectory_time(left, start_delay)
        if right is not None:
            right = shift_trajectory_time(right, start_delay)
        if left is not None and right is not None:
            self.send_replace("both_arms", combine_joint_trajectories(left, right))
        elif left is not None:
            self.send_replace("left_arm", left)
        elif right is not None:
            self.send_replace("right_arm", right)

    def _on_goal_response(self, group_name: str, fut) -> None:
        try:
            handle = fut.result()
        except Exception as exc:  # noqa: BLE001
            self._node.get_logger().warn(
                f"MPC 下发异常: {exc}", throttle_duration_sec=2.0
            )
            return
        if handle is None or not handle.accepted:
            self._node.get_logger().warn(
                "控制器拒绝 MPC 轨迹", throttle_duration_sec=2.0
            )
            return
        self._handles[group_name] = handle

    def cancel_sent(self) -> None:
        for handle in list(self._handles.values()):
            try:
                handle.cancel_goal_async()
            except Exception:  # noqa: BLE001
                pass
        self._handles.clear()
        self._send_futs.clear()

    def execute(
        self,
        group_name: str,
        trajectory: JointTrajectory,
        timeout_sec: float = 60.0,
        cancel_check=None,
    ) -> tuple[bool, str]:
        if self.execution_pending:
            return False, "上一段轨迹尚未确认停止，拒绝发送新轨迹"
        if cancel_check is not None and cancel_check():
            return False, "执行前已取消或超时"
        if trajectory is None or not trajectory.points:
            return False, "轨迹为空"
        client = self._clients.get(group_name)
        if client is None:
            return False, f"未知规划组: {group_name}"
        if not self.switch_group(group_name):
            return False, "控制器切换失败"
        if not client.wait_for_server(timeout_sec=2.0):
            return False, f"Action 不可用: {self._actions[group_name]}"

        if cancel_check is not None and cancel_check():
            return False, "发送轨迹前已取消或超时"

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory
        send_future = client.send_goal_async(goal)
        if not _wait_future(send_future, min(5.0, timeout_sec), cancel_check):
            self._awaiting_goal_ack = True
            self._unresolved_execution = send_future
            def cancel_late(future):
                try:
                    late = future.result()
                    if late is not None and late.accepted:
                        late.cancel_goal_async()
                        self._track_result(late.get_result_async())
                    else:
                        self._unresolved_execution = None
                except Exception:
                    self._execution_fault = True
                finally:
                    self._awaiting_goal_ack = False
            send_future.add_done_callback(cancel_late)
            return False, "发送轨迹目标超时"
        handle = send_future.result()
        if handle is None or not handle.accepted:
            return False, "控制器拒绝轨迹"
        result_future = handle.get_result_async()
        self._track_result(result_future)
        if not _wait_future(result_future, timeout_sec, cancel_check):
            handle.cancel_goal_async()
            # Do not send the next segment while a canceled segment still runs.
            _wait_future(result_future, 3.0)
            return False, "等待轨迹执行超时"
        try:
            wrapped = result_future.result()
        except Exception as exc:
            self._execution_fault = True
            handle.cancel_goal_async()
            return False, f"控制器结果异常，停止状态未确认: {exc}"
        if wrapped is None:
            self._execution_fault=True
            return False, "执行结果为空，停止状态未确认"
        if getattr(wrapped, "status", GoalStatus.STATUS_SUCCEEDED) != GoalStatus.STATUS_SUCCEEDED:
            return False, f"执行未成功, status={wrapped.status}"
        result = wrapped.result
        error_code = getattr(result, "error_code", 0)
        successful = getattr(FollowJointTrajectory.Result, "SUCCESSFUL", 0)
        if error_code not in (0, successful):
            return False, f"执行失败, error_code={error_code}"
        return True, "执行成功"

    def dispatch(self,commands):
        """Submit the complete offline timeline; execution belongs to controllers."""
        from rclpy.duration import Duration as ClockDuration
        for group,trajectory in commands:
            if not trajectory_is_finite(trajectory): return False,'控制适配器收到无效轨迹'
            client=self._clients.get(group)
            if client is None or not client.server_is_ready():
                return False,f'控制器不可用: {group}；规划结果已保留'
        if not self.switch_group(commands[0][0]): return False,'控制器切换失败；规划结果已保留'
        stamp=(self._node.get_clock().now()+ClockDuration(seconds=.05)).to_msg()
        for group,trajectory in commands:
            goal=FollowJointTrajectory.Goal();goal.trajectory=copy_joint_trajectory(trajectory)
            goal.trajectory.header.stamp=copy.deepcopy(stamp)
            future=self._clients[group].send_goal_async(goal)
            future.add_done_callback(lambda result,g=group:self._on_goal_response(g,result))
        return True,'完整轨迹已异步提交控制层，未等待执行结果'

    def execute_concurrent(self, commands, timeout_sec: float = 60.0, cancel_check=None):
        """Preflight all controllers, then start against one future ROS timestamp.

        Late acknowledgements remain tracked and are cancelled. Only terminal
        action results confirm a stop; a cancellation acknowledgement does not.
        """
        if self.execution_pending:
            return False, "上一段轨迹尚未确认停止，拒绝发送新轨迹"
        deadline = time.monotonic() + timeout_sec
        cancelled = lambda: ((cancel_check is not None and cancel_check()) or time.monotonic() >= deadline)
        if cancelled(): return False, "执行前已取消或超时"
        if not commands: return False, "轨迹为空"
        if len({group for group, _ in commands}) != len(commands):
            return False, "并行轨迹不能重复使用同一控制器"
        for group, trajectory in commands:
            if not trajectory_is_finite(trajectory): return False, "轨迹为空或含非有限值"
            if group not in self._clients: return False, f"未知规划组: {group}"
            if cancelled(): return False, "准备控制器超时或已取消"
            if not self.switch_group(group, timeout_sec=min(3., max(0., deadline-time.monotonic()))):
                return False, "控制器切换失败"
            if not self._clients[group].wait_for_server(timeout_sec=min(2., max(0., deadline-time.monotonic()))):
                return False, f"Action 不可用: {self._actions[group]}"
        if cancelled(): return False, "发送轨迹前已取消或超时"
        # Both controllers receive the same timestamp, regardless of send/ACK order.
        # Reserve 100 ms for cancellation before the scheduled start if an ACK is late.
        lead = .5
        start_ns = self._node.get_clock().now().nanoseconds + round(lead*1e9)
        ack_deadline = min(deadline, time.monotonic()+lead-.1)
        run = _ControllerRun(self)
        self._parallel_run = run
        try:
            for group, trajectory in commands:
                goal = FollowJointTrajectory.Goal()
                goal.trajectory = copy_joint_trajectory(trajectory)
                goal.trajectory.header.stamp = Time(sec=start_ns//1_000_000_000, nanosec=start_ns%1_000_000_000)
                run.send(self._clients[group], goal)
        except Exception as exc:
            run.fail(f"发送控制器目标异常: {exc}", unknown=True)
        finally:
            run.sending = False
        acknowledged = type('Acknowledged', (), {'done': lambda _: run.acknowledged()})()
        ok = _wait_future(acknowledged, max(0.,ack_deadline-time.monotonic()),
                          lambda: cancelled() or bool(run.error))
        if not ok or run.error or cancelled():
            reason=run.error or "并行目标未在共同起步前确认或任务已取消"
            run.cancel()
            _wait_future(run, 3.)
            return False, reason + ("；已确认停止" if run.done() else "；停止尚未确认")
        if not _wait_future(run, max(0.,deadline-time.monotonic()), lambda: cancelled() or bool(run.error)) or run.error:
            reason=run.error or "并行执行超时或已取消"
            run.cancel()
            _wait_future(run, 3.)
            return False, reason + ("；已确认停止" if run.done() else "；停止尚未确认")
        return True, "执行成功"

    def _cancel_handles(self, handles) -> None:
        for handle in handles:
            try:
                handle.cancel_goal_async()
            except Exception:
                pass

    def _goal_error(self, wrapped) -> str:
        if wrapped is None:
            return "执行结果为空"
        if getattr(wrapped, "status", GoalStatus.STATUS_SUCCEEDED) != GoalStatus.STATUS_SUCCEEDED:
            return f"执行未成功, status={wrapped.status}"
        result = wrapped.result
        error_code = getattr(result, "error_code", 0)
        successful = getattr(FollowJointTrajectory.Result, "SUCCESSFUL", 0)
        if error_code not in (0, successful):
            return f"执行失败, error_code={error_code}"
        return ""


class _ControllerRun:
    """Own ACK/result futures until every controller has confirmed termination."""
    def __init__(self, executor):
        self.executor=executor
        self.lock=threading.RLock()
        self.pending_ack=set()
        self.pending_result=set()
        self.handles=[]
        self.sending=True
        self.cancelled=False
        self.error=''
        self.fault=False

    def done(self):
        with self.lock:
            return not (self.sending or self.pending_ack or self.pending_result or self.fault)

    def acknowledged(self):
        with self.lock: return not self.sending and not self.pending_ack

    def fail(self, message, unknown=False):
        with self.lock:
            self.error=self.error or message
            if unknown:
                self.fault=True
                self.executor._execution_fault=True

    def send(self, client, goal):
        future=client.send_goal_async(goal)
        with self.lock: self.pending_ack.add(future)
        future.add_done_callback(self._ack)

    def _ack(self, future):
        with self.lock:
            try:
                handle=future.result()
                if handle is None or not handle.accepted:
                    self.fail('控制器拒绝并行轨迹')
                else:
                    self.handles.append(handle)
                    result=handle.get_result_async()
                    self.pending_result.add(result)
                    result.add_done_callback(self._result)
                    if self.cancelled: self.executor._cancel_handles([handle])
            except Exception as exc:
                self.fail(f'控制器目标确认异常，停止状态未确认: {exc}',unknown=True)
            finally:
                self.pending_ack.discard(future)

    def _result(self, future):
        with self.lock:
            try:
                wrapped=future.result()
                terminal=(GoalStatus.STATUS_SUCCEEDED,GoalStatus.STATUS_CANCELED,GoalStatus.STATUS_ABORTED)
                if wrapped is None or getattr(wrapped,'status',None) not in terminal:
                    self.fail('控制器未返回明确的终止状态',unknown=True)
                else:
                    error=self.executor._goal_error(wrapped)
                    if error: self.fail(error)
            except Exception as exc:
                self.fail(f'控制器结果异常，停止状态未确认: {exc}',unknown=True)
            finally:
                self.pending_result.discard(future)

    def cancel(self):
        with self.lock:
            self.cancelled=True
            self.executor._cancel_handles(list(self.handles))
