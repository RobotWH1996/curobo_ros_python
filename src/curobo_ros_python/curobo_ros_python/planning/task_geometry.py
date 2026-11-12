"""Shared request geometry and validation; execution belongs to each backend."""
import copy
import math
from types import SimpleNamespace as NS
import numpy as np
from curobo_ros_python.planning.geom import quat_mul_wxyz, transform_pose_wxyz
from curobo_ros_python.planning.payload import pose_matrix, resolve_object_geometry
from curobo_ros_python.planning.groups import GROUP_JOINTS
from curobo_ros_python.planning.robot_profile import active_profile


def xyz(vector):
    return np.array([vector.x, vector.y, vector.z], dtype=np.float64)


def matrix(pose):
    return pose_matrix(
        xyz(pose.position),
        [pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z],
    )


def relative_wxyz(pose):
    """Normalize a relative pose. An all-zero quaternion means identity."""
    translation = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float64)
    quaternion = np.array(
        [pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z],
        dtype=np.float64,
    )
    if not np.isfinite(translation).all() or not np.isfinite(quaternion).all():
        raise ValueError("相对目标位姿含无效数值")
    if float(np.linalg.norm(translation)) > 0.5:
        raise ValueError("相对目标位姿的平移不能超过 0.5 m")
    if float(np.linalg.norm(quaternion)) < 1e-8:
        quaternion = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        quaternion /= np.linalg.norm(quaternion)
    return translation, quaternion


def compose_pose(target, relative, in_base_frame=False):
    """Apply relative at the reference pose. Base adds it in base_link; otherwise T_target * T_relative."""
    translation, quaternion = relative_wxyz(relative)
    position = target.pose.position
    orientation = target.pose.orientation
    reference_t = (position.x, position.y, position.z)
    reference_q = (orientation.w, orientation.x, orientation.y, orientation.z)
    if in_base_frame:
        result_t = tuple(float(reference_t[i]) + float(translation[i]) for i in range(3))
        result_q = quat_mul_wxyz(quaternion, reference_q)
    else:
        result_t, result_q = transform_pose_wxyz(reference_t, reference_q, translation, quaternion)
    pose = copy.deepcopy(target)
    pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = result_t
    (
        pose.pose.orientation.w,
        pose.pose.orientation.x,
        pose.pose.orientation.y,
        pose.pose.orientation.z,
    ) = result_q
    return pose


def validate_common(task):
    if task.group_name not in ("left_arm", "right_arm"):
        raise ValueError("首版抓放Action只支持 left_arm/right_arm；原双臂多目标规划服务保留")
    active_profile().gripper_joint_name(task.group_name)
    if not math.isfinite(float(task.open_position)):
        raise ValueError("open_position 必须是有限关节位置")


class TaskGeometry:
    def validate_pick(self, task):
        self._preflight(task)
        self._validate_gripper_target(task, task.close_position)
        if abs(task.close_position - task.open_position) < 1e-6:
            raise ValueError("抓取的开、闭关节位置必须不同")
        self.base_target = self.node._pose_in_base(task.tool_pose)
        matrix(self.base_target.pose)
        in_base = bool(task.in_base_frame)
        self.base_pick = compose_pose(self.base_target, task.pick_pose, in_base)
        self.base_prepick = compose_pose(self.base_target, task.prepick_in_target, in_base)
        self.base_lift = compose_pose(self.base_target, task.lift_in_target, in_base)
        self._require_separation(self.base_pick, self.base_prepick, "预抓取")
        self._require_separation(self.base_pick, self.base_lift, "抬升")
        self.object_size, self.object_in_tool = self._geometry(
            task.object_size, self.base_target, self.base_pick
        )
        self.pick_via = self._resolve_route(task)
        self.pick_end = self._resolve_motion(task.end_pose, task.group_name, allow_empty=True)

    def _validate_place_fields(self, task):
        validate_common(task)
        reference = self.node._pose_in_base(task.tool_pose)
        matrix(reference.pose)
        self.place_reference = reference
        in_base = bool(task.in_base_frame)
        self.base_place = compose_pose(reference, task.place_pose, in_base)
        self.base_preplace = compose_pose(reference, task.preplace_in_target, in_base)
        self.base_retreat = compose_pose(reference, task.retreat_in_target, in_base)
        self._require_separation(self.base_place, self.base_preplace, "预放置")
        self._require_separation(self.base_place, self.base_retreat, "撤退")
        self.place_via = self._resolve_route(task)
        self.place_end = self._resolve_motion(task.end_pose, task.group_name, allow_empty=True)

    def _resolve_motion(self, waypoint, group, allow_empty=False):
        self.check()
        kind = waypoint.type.strip().lower()
        if not kind and allow_empty:
            if (
                waypoint.joint_positions
                or waypoint.pose.header.frame_id
                or not np.allclose(matrix(waypoint.pose.pose), np.eye(4))
            ):
                raise ValueError("end_pose 已填写目标时必须指定 type=pose/joint")
            return None
        if kind == "joint":
            values = np.asarray(waypoint.joint_positions, dtype=np.float64)
            if len(values) != len(GROUP_JOINTS[group]) or not np.isfinite(values).all():
                raise ValueError(f"{group} 关节路点需要{len(GROUP_JOINTS[group])}个有限弧度值")
            if waypoint.pose.header.frame_id or not np.allclose(
                matrix(waypoint.pose.pose), np.eye(4)
            ):
                raise ValueError("joint 路点不能同时填写 pose")
            return NS(kind="joint", positions=values.tolist(), pose=None)
        if kind == "pose":
            if waypoint.joint_positions:
                raise ValueError("pose 路点不能同时填写 joint_positions")
            pose = self.node._pose_in_base(waypoint.pose)
            matrix(pose.pose)
            return NS(kind="pose", positions=None, pose=pose)
        raise ValueError("路点 type 必须是 pose 或 joint；只有空 end_pose 可以省略 type")

    def _resolve_route(self, task):
        return [self._resolve_motion(w, task.group_name) for w in task.waypoints]

    def _validate_gripper_target(self, task, position):
        name = active_profile().gripper_joint_name(task.group_name)
        limits = self.p._gripper_state._limits
        if name not in limits:
            raise ValueError(f"模型没有夹爪关节 {name}")
        lo, hi = limits[name]
        if not math.isfinite(position) or not lo <= position <= hi:
            raise ValueError(f"{name} 目标须在模型关节限位 [{lo}, {hi}] 内")
        return name

    def _geometry(self, size, reference, grasp):
        in_tool = np.linalg.inv(matrix(grasp.pose)) @ matrix(reference.pose)
        return resolve_object_geometry(
            xyz(size),
            in_tool,
            False,
            self.node._p("pick_place_default_object_size"),
            [0, 0, 0, 0, 0, 0, 1],
        )

    def _require_separation(self, start, goal, name):
        distance = float(np.linalg.norm(xyz(goal.pose.position) - xyz(start.pose.position)))
        if not 0.001 <= distance <= 0.5:
            raise ValueError(f"{name}与目标位姿的距离须在1mm至0.5m之间")
