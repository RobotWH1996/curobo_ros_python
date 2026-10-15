"""规划算法共用的位姿小工具和锁关节配置。TrajOpt / MPC 都从这里引，互不依赖。"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from curobo.config_io import join_path, load_yaml
from curobo.content import get_robot_configs_path
from curobo.types import Pose

from curobo_ros_python.planning.geom import quat_angle_wxyz
from curobo_ros_python.planning.groups import GROUP_JOINTS


def squeeze_pose(pose: Pose) -> Pose:
    return Pose(
        position=pose.position.reshape(-1, 3)[:1].contiguous(),
        quaternion=pose.quaternion.reshape(-1, 4)[:1].contiguous(),
        normalize_rotation=False,
    )


def pose_xyz_wxyz(pose: Pose) -> Tuple[List[float], List[float]]:
    p = pose.position.detach().reshape(-1).cpu().numpy()
    q = pose.quaternion.detach().reshape(-1).cpu().numpy()
    return [float(p[0]), float(p[1]), float(p[2])], [
        float(q[0]),
        float(q[1]),
        float(q[2]),
        float(q[3]),
    ]


def pose_error(current: Pose, target: Pose) -> Tuple[float, float]:
    c_xyz, c_wxyz = pose_xyz_wxyz(current)
    t_xyz, t_wxyz = pose_xyz_wxyz(target)
    pos = float(np.linalg.norm(np.array(t_xyz) - np.array(c_xyz)))
    ang = quat_angle_wxyz(c_wxyz, t_wxyz)
    return pos, ang


def robot_cfg_with_locked_joints(
    robot: str,
    current: Optional[Dict[str, float]],
    extra_lock_names: Optional[Sequence[str]] = None,
) -> Tuple[Dict, Dict[str, float]]:
    """锁住不下发的关节：躯干/头，以及本次没用的那一侧手臂。

    TrajOpt / MPC 都会优化 cspace 里所有未锁关节。只规划左臂却让右臂参与时，
    求解器会把「靠近目标」分给右臂，而且自由度多一倍。
    """
    data = load_yaml(join_path(str(get_robot_configs_path()), robot))
    kinematics = data["robot_cfg"]["kinematics"]
    cspace = kinematics["cspace"]
    all_names = list(cspace["joint_names"])
    arm_names = set(GROUP_JOINTS.get("both_arms") or [])
    extra = set(extra_lock_names or [])
    if not arm_names and not extra:
        return data, {}
    defaults = list(cspace.get("default_joint_position") or [])
    locks = dict(kinematics.get("lock_joints") or {})
    for i, name in enumerate(all_names):
        if name in locks:
            continue
        if name in arm_names and name not in extra:
            continue
        if current and name in current:
            locks[name] = float(current[name])
        elif i < len(defaults):
            locks[name] = float(defaults[i])
        else:
            locks[name] = 0.0
    kinematics["lock_joints"] = locks
    return data, locks
