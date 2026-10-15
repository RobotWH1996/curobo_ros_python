"""RViz Marker / DisplayTrajectory。只给调试 launch 用。"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

import numpy as np
from geometry_msgs.msg import Point, PoseStamped
from moveit_msgs.msg import DisplayTrajectory, RobotState, RobotTrajectory
from sensor_msgs.msg import JointState
from std_msgs.msg import ColorRGBA, Header
from trajectory_msgs.msg import JointTrajectory
from visualization_msgs.msg import Marker, MarkerArray

from curobo_ros_python.planning.geom import quat_mul_wxyz

_GREEN = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
_ORANGE = ColorRGBA(r=1.0, g=0.45, b=0.05, a=0.45)
_CYAN = ColorRGBA(r=0.1, g=0.75, b=1.0, a=1.0)
_MAGENTA = ColorRGBA(r=1.0, g=0.25, b=0.85, a=1.0)
_YELLOW = ColorRGBA(r=1.0, g=0.85, b=0.05, a=1.0)
_WHITE = ColorRGBA(r=0.92, g=0.92, b=0.92, a=1.0)
_RED = ColorRGBA(r=1.0, g=0.15, b=0.1, a=1.0)
_BLUE = ColorRGBA(r=0.15, g=0.35, b=1.0, a=1.0)
_NS = "curobo_traj"
_MPC_NS = "curobo_mpc"
_WORLD_NS = "curobo_world"
_GOAL_NS = "curobo_goal"
# 箭头沿 marker 局部 +X。右乘后：+90° 绕 Z 指到姿态 +Y，-90° 绕 Y 指到姿态 +Z。
_AXIS_LOCAL_WXYZ = (
    (1.0, 0.0, 0.0, 0.0),
    (0.70710678, 0.0, 0.0, 0.70710678),
    (0.70710678, 0.0, -0.70710678, 0.0),
)
_AXIS_COLORS = (_RED, _GREEN, _BLUE)
_MAX_VOXEL_MARKERS = 40000


def joint_traj_to_display(
    traj: JointTrajectory,
    joint_map: Mapping[str, float],
    model_id: str = "",
    frame_id: str = "base_link",
) -> DisplayTrajectory:
    msg = DisplayTrajectory()
    msg.model_id = model_id
    start = RobotState()
    start.is_diff = False
    js = JointState()
    js.header = Header(frame_id=frame_id)
    js.name = list(joint_map.keys())
    js.position = [float(joint_map[n]) for n in js.name]
    start.joint_state = js
    msg.trajectory_start = start
    robot_traj = RobotTrajectory()
    robot_traj.joint_trajectory = traj
    if not robot_traj.joint_trajectory.header.frame_id:
        robot_traj.joint_trajectory.header.frame_id = frame_id
    msg.trajectory.append(robot_traj)
    return msg


def _delete_all(frame_id: str, ns: str) -> Marker:
    clear = Marker()
    clear.header.frame_id = frame_id
    clear.ns = ns
    clear.id = 0
    clear.action = Marker.DELETEALL
    return clear


def _marker(
    frame_id: str,
    marker_id: int,
    mtype: int,
    points: Sequence[Point],
    *,
    line_width: float = 0.0,
    sphere_radius: float = 0.0,
    color: Optional[ColorRGBA] = None,
    ns: str = _NS,
) -> Marker:
    m = Marker()
    m.header.frame_id = frame_id
    m.ns = ns
    m.id = marker_id
    m.type = mtype
    m.action = Marker.ADD
    m.color = color if color is not None else _GREEN
    m.pose.orientation.w = 1.0
    m.lifetime.sec = 0
    m.points = list(points)
    if mtype == Marker.LINE_STRIP:
        m.scale.x = line_width
    else:
        m.scale.x = m.scale.y = m.scale.z = sphere_radius
    return m


def ee_paths_to_markers(
    ee_paths: Optional[Dict[str, Sequence[Sequence[float]]]],
    frame_id: str = "base_link",
    line_width: float = 0.008,
    sphere_radius: float = 0.012,
    ns: str = _NS,
    colors: Optional[Mapping[str, ColorRGBA]] = None,
) -> MarkerArray:
    array = MarkerArray()
    array.markers.append(_delete_all(frame_id, ns))
    if not ee_paths:
        return array
    marker_id = 1
    for link, xyzs in ee_paths.items():
        points = []
        for xyz in xyzs:
            if len(xyz) < 3:
                continue
            pt = Point()
            pt.x, pt.y, pt.z = float(xyz[0]), float(xyz[1]), float(xyz[2])
            points.append(pt)
        if not points:
            continue
        color = None if colors is None else colors.get(link)
        line_pts = points if len(points) >= 2 else points + points
        array.markers.append(
            _marker(
                frame_id,
                marker_id,
                Marker.LINE_STRIP,
                line_pts,
                line_width=line_width,
                color=color,
                ns=ns,
            )
        )
        marker_id += 1
        array.markers.append(
            _marker(
                frame_id,
                marker_id,
                Marker.SPHERE_LIST,
                points,
                sphere_radius=sphere_radius,
                color=color,
                ns=ns,
            )
        )
        marker_id += 1
    return array


def mpc_preview_to_markers(
    ee_paths: Optional[Dict[str, Sequence[Sequence[float]]]],
    used_xyz: Optional[Mapping[str, Sequence[float]]] = None,
    frame_id: str = "base_link",
    left_ee: str = "left_arm_ee_link",
    right_ee: str = "right_arm_ee_link",
) -> MarkerArray:
    """MPC 预测末端（左青 / 右品红）+ 本周期实际目标球。"""
    colors = {left_ee: _CYAN, right_ee: _MAGENTA}
    array = ee_paths_to_markers(
        ee_paths,
        frame_id=frame_id,
        line_width=0.01,
        sphere_radius=0.01,
        ns=_MPC_NS,
        colors=colors,
    )
    array.markers.insert(0, _delete_all(frame_id, _NS))
    marker_id = 100
    for link, xyz in (used_xyz or {}).items():
        if xyz is None or len(xyz) < 3:
            continue
        pt = Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]))
        color = colors.get(link, _GREEN)
        array.markers.append(
            _marker(
                frame_id,
                marker_id,
                Marker.SPHERE,
                [],
                sphere_radius=0.028,
                color=color,
                ns=_MPC_NS,
            )
        )
        array.markers[-1].pose.position = pt
        marker_id += 1
    return array


def _set_quat(marker: Marker, wxyz: Sequence[float]) -> None:
    marker.pose.orientation.w = float(wxyz[0])
    marker.pose.orientation.x = float(wxyz[1])
    marker.pose.orientation.y = float(wxyz[2])
    marker.pose.orientation.z = float(wxyz[3])


def _goal_sphere_color(name: str) -> ColorRGBA:
    if str(name).startswith("via:"):
        return _WHITE
    if "right" in (name or ""):
        return _MAGENTA
    return _YELLOW


def goal_waypoints_to_markers(
    items: Sequence[tuple],
    frame_id: str = "base_link",
) -> MarkerArray:
    """目标路点球 + RGB 坐标轴。items 为 (名字, PoseStamped)，规划失败也不清。"""
    array = MarkerArray()
    array.markers.append(_delete_all(frame_id, _GOAL_NS))
    marker_id = 1
    for name, pose in items:
        if not isinstance(pose, PoseStamped):
            continue
        p = pose.pose.position
        q = pose.pose.orientation
        src = (pose.header.frame_id or frame_id).lstrip("/") or frame_id
        pose_q = (q.w, q.x, q.y, q.z)
        via = str(name).startswith("via:")
        sphere = _marker(
            src,
            marker_id,
            Marker.SPHERE,
            [],
            sphere_radius=0.015 if via else 0.025,
            color=_goal_sphere_color(str(name)),
            ns=_GOAL_NS,
        )
        sphere.pose.position.x = float(p.x)
        sphere.pose.position.y = float(p.y)
        sphere.pose.position.z = float(p.z)
        array.markers.append(sphere)
        marker_id += 1
        for local_q, color in zip(_AXIS_LOCAL_WXYZ, _AXIS_COLORS):
            arrow = _marker(
                src,
                marker_id,
                Marker.ARROW,
                [],
                color=color,
                ns=_GOAL_NS,
            )
            arrow.pose.position.x = float(p.x)
            arrow.pose.position.y = float(p.y)
            arrow.pose.position.z = float(p.z)
            _set_quat(arrow, quat_mul_wxyz(pose_q, local_q))
            arrow.scale.x = 0.06 if via else 0.12
            arrow.scale.y = 0.006 if via else 0.008
            arrow.scale.z = 0.012 if via else 0.016
            array.markers.append(arrow)
            marker_id += 1
    return array


def clear_mpc_markers(frame_id: str = "base_link") -> MarkerArray:
    array = MarkerArray()
    array.markers.append(_delete_all(frame_id, _MPC_NS))
    return array


def clear_trajopt_markers(frame_id: str = "base_link") -> MarkerArray:
    array = MarkerArray()
    array.markers.append(_delete_all(frame_id, _NS))
    return array


def empty_display(model_id: str = "", frame_id: str = "base_link") -> DisplayTrajectory:
    msg = DisplayTrajectory()
    msg.model_id = model_id
    start = RobotState()
    start.is_diff = False
    js = JointState()
    js.header = Header(frame_id=frame_id)
    start.joint_state = js
    msg.trajectory_start = start
    return msg


def collision_voxels_to_markers(
    points: np.ndarray,
    pitch: float,
    frame_id: str = "base_link",
    max_voxels: int = _MAX_VOXEL_MARKERS,
) -> MarkerArray:
    """把点云体素化结果画成 CUBE_LIST，对应 Mesh.from_pointcloud 的占用格子。"""
    array = MarkerArray()
    array.markers.append(_delete_all(frame_id, _WORLD_NS))

    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pitch = max(float(pitch), 1e-4)
    if pts.shape[0] == 0:
        return array
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    if pts.shape[0] == 0:
        return array
    origin = pts.min(axis=0) - pitch
    ijk = np.floor((pts - origin) / pitch).astype(np.int64)
    ijk = np.unique(ijk, axis=0)
    if ijk.shape[0] > max_voxels:
        sel = np.linspace(0, ijk.shape[0] - 1, max_voxels).astype(np.int64)
        ijk = ijk[sel]
    centers = (ijk.astype(np.float64) + 0.5) * pitch + origin

    cubes = Marker()
    cubes.header.frame_id = frame_id
    cubes.ns = _WORLD_NS
    cubes.id = 1
    cubes.type = Marker.CUBE_LIST
    cubes.action = Marker.ADD
    cubes.pose.orientation.w = 1.0
    cubes.scale.x = cubes.scale.y = cubes.scale.z = pitch
    cubes.color = _ORANGE
    cubes.lifetime.sec = 0
    cubes.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in centers]
    array.markers.append(cubes)
    return array
