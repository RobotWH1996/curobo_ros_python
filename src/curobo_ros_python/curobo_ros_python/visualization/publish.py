"""把规划结果发到 RViz。消息拼装在 rviz.py；这里只看节点开关并 publish。

末端绿线 / MPC 预览的 FK 在这里做，不进 algorithms/。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from curobo.types import JointState
from trajectory_msgs.msg import JointTrajectory

from curobo_ros_python.planning.algorithms.common import pose_xyz_wxyz
from curobo_ros_python.planning.convert import mpc_action_to_arrays, pose_to_stamped
from curobo_ros_python.planning.groups import GROUP_EE_LINK, GROUP_JOINTS
from curobo_ros_python.planning.trajectory import numpy_traj_to_msg
from curobo_ros_python.visualization.rviz import (
    clear_mpc_markers,
    clear_trajopt_markers,
    collision_voxels_to_markers,
    ee_paths_to_markers,
    empty_display,
    goal_waypoints_to_markers,
    joint_traj_to_display,
    mpc_preview_to_markers,
)


def _as_pos_2d(tensor) -> Optional[object]:
    if tensor is None:
        return None
    data = tensor.detach()
    while data.ndim > 2:
        data = data[0]
    if data.ndim == 1:
        data = data.unsqueeze(0)
    return data


def ee_paths_from_joint_state(
    compute_kinematics,
    traj: JointState,
    frames: Sequence[str],
    max_points: Optional[int] = None,
) -> Dict[str, List[List[float]]]:
    frames = [name for name in frames if name]
    if not frames or traj is None or traj.position is None:
        return {}
    pos = _as_pos_2d(traj.position)
    if pos is None:
        return {}
    n = int(pos.shape[0])
    if n <= 0:
        return {}
    if max_points is not None and n > max_points:
        step = max(1, int(np.ceil(n / float(max_points))))
        idxs = list(range(0, n, step))
        if idxs[-1] != n - 1:
            idxs.append(n - 1)
        pos = pos[idxs].contiguous()
    else:
        pos = pos.contiguous()
    names = list(traj.joint_names or [])
    batch = JointState.from_position(pos, joint_names=names)
    kin = compute_kinematics(batch)
    if kin.tool_poses is None:
        return {}
    out: Dict[str, List[List[float]]] = {}
    for frame in frames:
        xyz = kin.tool_poses.get_link_pose(frame).position.detach().reshape(-1, 3)
        out[frame] = [[float(v) for v in row] for row in xyz.cpu().numpy()]
    return {name: pts for name, pts in out.items() if pts}


def ee_paths_from_mpc_result(
    compute_kinematics,
    result,
    frames: Sequence[str],
    fallback_names: Sequence[str],
    max_points: int = 16,
) -> Dict[str, List[List[float]]]:
    if result is None:
        return {}
    rss = getattr(result, "robot_state_sequence", None)
    seq = getattr(rss, "joint_state", None) if rss is not None else None
    if seq is None or seq.position is None:
        seq = getattr(result, "action_sequence", None)
    if seq is None or seq.position is None:
        return {}
    if not seq.joint_names:
        seq = seq.clone()
        seq.joint_names = list(fallback_names)
    return ee_paths_from_joint_state(compute_kinematics, seq, frames, max_points=max_points)


def trajopt_ee_frames(group_name: str, tool_frames: Sequence[str]) -> List[str]:
    if group_name == "both_arms":
        frames = list(GROUP_EE_LINK.values())
    elif group_name in GROUP_EE_LINK:
        frames = [GROUP_EE_LINK[group_name]]
    else:
        frames = list(tool_frames)
    return [name for name in frames if name in set(tool_frames)]


def joint_goal_marker_items(node, group_name: str, goals) -> list:
    """关节角路点做 FK。前面的是 via，最后一个是目标。"""
    frames = trajopt_ee_frames(group_name, node.planner.tool_frames)
    if not frames or not goals:
        return []
    stamp = node.get_clock().now().to_msg()
    items = []
    last = len(goals) - 1
    for i, goal in enumerate(goals):
        poses = node.planner.fk_tool_poses(goal)
        prefix = "" if i == last else "via:"
        for frame in frames:
            pose = poses.get(frame)
            if pose is None:
                continue
            items.append((prefix + frame, pose_to_stamped(pose, node.base_frame, stamp)))
    return items


def publish_goal_waypoints(node, items) -> bool:
    """发布笛卡尔或关节角目标路点。与规划成败无关，关 visualize_goal_waypoints 才不发。"""
    pub = getattr(node, "_goal_marker_pub", None)
    if not getattr(node, "visualize_goal_waypoints", False) or pub is None:
        return False
    pub.publish(goal_waypoints_to_markers(items, frame_id=node.base_frame))
    return True


def publish_collision_world(node, points) -> bool:
    if not getattr(node, "visualize_mesh_obstacles", False) or node._world_marker_pub is None:
        return False
    node._world_marker_pub.publish(
        collision_voxels_to_markers(
            points,
            pitch=node.pointcloud_pitch,
            frame_id=node.pointcloud_parent_frame or node.base_frame,
        )
    )
    return True


def publish_trajopt(node, traj: JointTrajectory, ee_paths) -> List[str]:
    published: List[str] = []
    if not getattr(node, "visualize_trajectory", False) or node._display_pub is None:
        return published
    node._display_pub.publish(
        joint_traj_to_display(
            traj, node._joint_map, model_id=node.robot_model_id, frame_id=node.base_frame
        )
    )
    published.append("/display_planned_path")
    if node._marker_pub is not None:
        node._marker_pub.publish(ee_paths_to_markers(ee_paths, frame_id=node.base_frame))
        published.append(node._marker_topic)
    return published


def clear_trajopt_preview(node) -> None:
    """清掉 TrajOpt 绿线和 /display_planned_path，避免切到 MPC 后还留着。"""
    if not getattr(node, "visualize_trajectory", False):
        return
    if node._marker_pub is not None:
        node._marker_pub.publish(clear_trajopt_markers(frame_id=node.base_frame))
    if node._display_pub is not None:
        node._display_pub.publish(
            empty_display(model_id=node.robot_model_id, frame_id=node.base_frame)
        )


def publish_mpc_preview(node, mpc, step) -> None:
    if not getattr(node, "visualize_trajectory", False):
        return
    ee_paths = {}
    if step is not None and step.result is not None:
        try:
            ee_paths = ee_paths_from_mpc_result(
                mpc.mpc.compute_kinematics,
                step.result,
                (mpc.left_ee, mpc.right_ee),
                mpc.joint_names,
            )
        except Exception as exc:  # noqa: BLE001
            node.get_logger().warn(
                f"MPC 末端 FK 失败，RViz 预览可能没有: {exc}",
                throttle_duration_sec=2.0,
            )
    used_xyz = {}
    for name, pose in (step.used if step is not None else {}).items():
        if pose is None:
            continue
        xyz, _wxyz = pose_xyz_wxyz(pose)
        used_xyz[name] = xyz
    if node._marker_pub is not None:
        node._marker_pub.publish(
            mpc_preview_to_markers(
                ee_paths,
                used_xyz,
                frame_id=node.base_frame,
                left_ee=mpc.left_ee,
                right_ee=mpc.right_ee,
            )
        )
    if node._display_pub is None or step is None or step.result is None:
        return
    # 单臂跟踪时另一侧被锁，不在优化变量里，退回只画在跟的那只臂。
    packed = None
    for group in ("both_arms", "left_arm", "right_arm"):
        packed = mpc_action_to_arrays(step.result, group, joint_names=mpc.joint_names)
        if packed is not None:
            break
    if packed is None:
        return
    pos, vel = packed
    names = GROUP_JOINTS.get(group) or []
    traj = numpy_traj_to_msg(
        names, pos, vel, dt=mpc.command_dt, velocity_scale=float(node.mpc_velocity_scale)
    )
    traj.header.stamp = node.get_clock().now().to_msg()
    traj.header.frame_id = node.base_frame
    node._display_pub.publish(
        joint_traj_to_display(
            traj, node._joint_map, model_id=node.robot_model_id, frame_id=node.base_frame
        )
    )


def clear_mpc_preview(node) -> None:
    if not getattr(node, "visualize_trajectory", False) or node._marker_pub is None:
        return
    node._marker_pub.publish(clear_mpc_markers(frame_id=node.base_frame))
