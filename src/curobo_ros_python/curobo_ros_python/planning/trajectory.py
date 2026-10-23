"""规划结果转 JointTrajectory，以及 FollowJointTrajectory 执行。"""

from __future__ import annotations

import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
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


def _wait_future(future, timeout_sec: float) -> bool:
    deadline = time.monotonic() + timeout_sec
    while not future.done():
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.01)
    return True


class TrajectoryExecutor:
    def __init__(self, node: Node, switch_controllers: bool = True):
        self._node = node
        self._switch = switch_controllers
        self._clients = {
            group: ActionClient(node, FollowJointTrajectory, action)
            for group, action in GROUP_ACTION.items()
        }
        self._switch_client = node.create_client(
            SwitchController, "/controller_manager/switch_controller"
        )
        self._last_group: Optional[str] = None
        self._handles: Dict[str, object] = {}
        self._send_futs: List[object] = []
        self._ready_groups: set = set()

    def switch_group(self, group_name: str, timeout_sec: float = 3.0) -> bool:
        if not self._switch or group_name not in GROUP_CONTROLLER_SWITCH:
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
                    f"Action 不可用: {GROUP_ACTION[group_name]}",
                    throttle_duration_sec=2.0,
                )
                return
            self._ready_groups.add(group_name)
        elif not client.server_is_ready():
            self._ready_groups.discard(group_name)
            self._node.get_logger().warn(
                f"Action 已断开: {GROUP_ACTION[group_name]}，本周期跳过",
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
    ) -> tuple[bool, str]:
        if trajectory is None or not trajectory.points:
            return False, "轨迹为空"
        client = self._clients.get(group_name)
        if client is None:
            return False, f"未知规划组: {group_name}"
        if not self.switch_group(group_name):
            return False, "控制器切换失败"
        if not client.wait_for_server(timeout_sec=2.0):
            return False, f"Action 不可用: {GROUP_ACTION[group_name]}"

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory
        send_future = client.send_goal_async(goal)
        if not _wait_future(send_future, 5.0):
            return False, "发送轨迹目标超时"
        handle = send_future.result()
        if handle is None or not handle.accepted:
            return False, "控制器拒绝轨迹"
        result_future = handle.get_result_async()
        if not _wait_future(result_future, timeout_sec):
            return False, "等待轨迹执行超时"
        wrapped = result_future.result()
        if wrapped is None:
            return False, "执行结果为空"
        if getattr(wrapped, "status", GoalStatus.STATUS_SUCCEEDED) != GoalStatus.STATUS_SUCCEEDED:
            return False, f"执行未成功, status={wrapped.status}"
        result = wrapped.result
        error_code = getattr(result, "error_code", 0)
        successful = getattr(FollowJointTrajectory.Result, "SUCCESSFUL", 0)
        if error_code not in (0, successful):
            return False, f"执行失败, error_code={error_code}"
        return True, "执行成功"
