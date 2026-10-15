"""ROS 消息 <-> cuRobo 类型。algorithms/ 禁止 import 本模块。"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory

from curobo.types import JointState, Pose
from curobo_ros_interfaces.action import FollowMpc
from curobo_ros_interfaces.msg import MpcArmResult, MpcPlan
from curobo_ros_python.planning.algorithms.common import pose_error, pose_xyz_wxyz
from curobo_ros_python.planning.groups import GROUP_JOINTS
from curobo_ros_python.planning.trajectory import numpy_traj_to_msg


def pose_from_xyz_xyzw(xyz: Sequence[float], xyzw: Sequence[float], device_cfg) -> Pose:
    return Pose.from_list(
        [
            float(xyz[0]),
            float(xyz[1]),
            float(xyz[2]),
            float(xyzw[0]),
            float(xyzw[1]),
            float(xyzw[2]),
            float(xyzw[3]),
        ],
        device_cfg=device_cfg,
        q_xyzw=True,
    )


def stamped_to_pose(stamped: PoseStamped, device_cfg) -> Pose:
    p = stamped.pose.position
    q = stamped.pose.orientation
    return pose_from_xyz_xyzw([p.x, p.y, p.z], [q.x, q.y, q.z, q.w], device_cfg)


def pose_to_stamped(pose: Pose, frame_id: str, stamp) -> PoseStamped:
    xyz, wxyz = pose_xyz_wxyz(pose)
    stamped = PoseStamped()
    stamped.header.stamp = stamp
    stamped.header.frame_id = frame_id
    stamped.pose.position.x = xyz[0]
    stamped.pose.position.y = xyz[1]
    stamped.pose.position.z = xyz[2]
    stamped.pose.orientation.w = wxyz[0]
    stamped.pose.orientation.x = wxyz[1]
    stamped.pose.orientation.y = wxyz[2]
    stamped.pose.orientation.z = wxyz[3]
    return stamped


def joint_map_to_curobo(
    names: Sequence[str],
    joint_map: Dict[str, float],
    device,
    default_position=None,
    joint_vel: Optional[Dict[str, float]] = None,
) -> Tuple[JointState, List[str]]:
    missing: List[str] = []
    q: List[float] = []
    for i, name in enumerate(names):
        if name in joint_map:
            q.append(float(joint_map[name]))
        elif default_position is not None:
            q.append(float(default_position[i].item()))
            missing.append(name)
        else:
            q.append(0.0)
            missing.append(name)
    tensor = torch.tensor([q], device=device, dtype=torch.float32)
    state = JointState.from_position(tensor, joint_names=list(names))
    # TrajOpt 不传 joint_vel，起点速度保持 None；MPC 传（可能是空 dict），按缺失补 0。
    if joint_vel is not None:
        vels = [float(joint_vel.get(name, 0.0)) for name in names]
        state.velocity = torch.tensor([vels], device=device, dtype=torch.float32)
        state.acceleration = torch.zeros_like(state.velocity)
    return state, missing


def group_goal_state(
    all_names: Sequence[str],
    group_name: str,
    joint_angles: Sequence[float],
    current: JointState,
) -> JointState:
    group_joints = GROUP_JOINTS[group_name]
    if len(joint_angles) != len(group_joints):
        raise ValueError(
            f"{group_name} 需要 {len(group_joints)} 个关节角，收到 {len(joint_angles)}"
        )
    name_to_idx = {n: i for i, n in enumerate(all_names)}
    pos = current.position.reshape(-1).clone()
    for jn, val in zip(group_joints, joint_angles):
        if jn not in name_to_idx:
            raise ValueError(f"cuRobo 配置中没有关节 {jn}")
        pos[name_to_idx[jn]] = float(val)
    return JointState.from_position(pos.reshape(1, -1), joint_names=list(all_names))


def _as_traj_2d(tensor: torch.Tensor) -> np.ndarray:
    data = tensor.detach()
    while data.ndim > 2:
        data = data[0]
    if data.ndim == 1:
        data = data.unsqueeze(0)
    return data.cpu().numpy()


def curobo_traj_to_arrays(
    traj: JointState,
    group_name: str,
) -> Tuple[List[str], np.ndarray, Optional[np.ndarray]]:
    names = list(traj.joint_names or [])
    keep = list(GROUP_JOINTS.get(group_name) or names)
    missing = [n for n in keep if n not in names]
    if missing:
        raise RuntimeError(f"轨迹里缺少 {group_name} 的关节: {missing}")
    idx = [names.index(n) for n in keep]
    pos = _as_traj_2d(traj.position)[:, idx]
    vel = None
    if traj.velocity is not None:
        vel = _as_traj_2d(traj.velocity)[:, idx]
    return keep, pos, vel


def curobo_traj_to_msg(
    traj: JointState,
    group_name: str,
    dt: float,
    velocity_scale: float = 1.0,
) -> JointTrajectory:
    names, pos, vel = curobo_traj_to_arrays(traj, group_name)
    return numpy_traj_to_msg(names, pos, vel, dt=dt, velocity_scale=velocity_scale)


def mpc_action_to_arrays(
    result,
    group_name: str,
    joint_names: Optional[Sequence[str]] = None,
) -> Optional[Tuple[np.ndarray, Optional[np.ndarray]]]:
    seq = getattr(result, "action_sequence", None) if result is not None else None
    if seq is None or seq.position is None:
        return None
    names = list(seq.joint_names or joint_names or [])
    want = GROUP_JOINTS.get(group_name) or []
    # 该组关节必须全在优化变量里，缺一个就退回「保持原位」，避免列数和 joint_names 对不上。
    if not want or any(n not in names for n in want):
        return None
    idx = [names.index(n) for n in want]
    pos = _as_traj_2d(seq.position)[:, idx]
    vel = None
    if seq.velocity is not None:
        vel = _as_traj_2d(seq.velocity)[:, idx]
    return pos, vel


def ik_group_angles(
    solution: JointState,
    group_name: str,
    current_map: Dict[str, float],
) -> List[float]:
    names = list(solution.joint_names or [])
    pos = solution.position
    while pos.ndim > 1:
        pos = pos[0]
    q = pos.detach().float().cpu().tolist()
    joint_map = {n: float(v) for n, v in zip(names, q)}
    for n, v in current_map.items():
        joint_map.setdefault(n, float(v))
    return [joint_map[n] for n in GROUP_JOINTS[group_name]]


def mpc_arm_result(
    *,
    group_name: str,
    target_valid: bool,
    used_pose: Optional[Pose],
    current_ee: Optional[Pose],
    message: str,
    result,
    joint_map: Dict[str, float],
    command_dt: float,
    velocity_scale: float,
    frame_id: str,
    stamp,
    joint_names: Optional[Sequence[str]] = None,
) -> MpcArmResult:
    arm = MpcArmResult()
    arm.target_valid = bool(target_valid)
    arm.message = message
    if used_pose is not None:
        arm.used_target = pose_to_stamped(used_pose, frame_id, stamp)
    if current_ee is not None and used_pose is not None:
        pos_e, ori_e = pose_error(current_ee, used_pose)
        arm.position_error = pos_e
        arm.orientation_error = ori_e
    packed = mpc_action_to_arrays(result, group_name, joint_names=joint_names)
    names = GROUP_JOINTS.get(group_name) or []
    if packed is None:
        q = [[joint_map.get(n, 0.0) for n in names]]
        arm.trajectory = numpy_traj_to_msg(
            names, q, None, dt=command_dt, velocity_scale=velocity_scale
        )
    else:
        pos, vel = packed
        arm.trajectory = numpy_traj_to_msg(
            names, pos, vel, dt=command_dt, velocity_scale=velocity_scale
        )
    arm.trajectory.header.stamp = stamp
    arm.trajectory.header.frame_id = frame_id
    return arm


def mpc_step_to_msgs(
    step,
    *,
    left_ee: str,
    right_ee: str,
    left_valid: bool,
    right_valid: bool,
    left_note: str,
    right_note: str,
    joint_map: Dict[str, float],
    command_dt: float,
    velocity_scale: float,
    frame_id: str,
    stamp,
    joint_names: Optional[Sequence[str]] = None,
) -> Tuple[MpcPlan, FollowMpc.Feedback]:
    bits = [x for x in (left_note, right_note, step.message) if x]
    header_msg = "；".join(bits) if bits else "ok"
    out = MpcPlan()
    out.header.stamp = stamp
    out.header.frame_id = frame_id
    out.success = bool(step.success)
    out.message = header_msg
    out.solve_time = float(step.solve_time)
    out.left = mpc_arm_result(
        group_name="left_arm",
        target_valid=left_valid,
        used_pose=step.used.get(left_ee),
        current_ee=step.current_ee.get(left_ee),
        message=left_note,
        result=step.result,
        joint_map=joint_map,
        command_dt=command_dt,
        velocity_scale=velocity_scale,
        frame_id=frame_id,
        stamp=stamp,
        joint_names=joint_names,
    )
    out.right = mpc_arm_result(
        group_name="right_arm",
        target_valid=right_valid,
        used_pose=step.used.get(right_ee),
        current_ee=step.current_ee.get(right_ee),
        message=right_note,
        result=step.result,
        joint_map=joint_map,
        command_dt=command_dt,
        velocity_scale=velocity_scale,
        frame_id=frame_id,
        stamp=stamp,
        joint_names=joint_names,
    )
    fb = FollowMpc.Feedback()
    fb.left_target_valid = bool(left_valid)
    fb.right_target_valid = bool(right_valid)
    fb.left_position_error = float(out.left.position_error)
    fb.left_orientation_error = float(out.left.orientation_error)
    fb.right_position_error = float(out.right.position_error)
    fb.right_orientation_error = float(out.right.orientation_error)
    fb.solve_time = float(step.solve_time)
    fb.plan_success = bool(out.success)
    fb.message = header_msg
    return out, fb
