"""规划组、关节顺序和控制器名称。

关节顺序与 FollowJointTrajectory 控制器一致，也是服务请求里每个路点
`positions` 的约定顺序。

由 ``apply_robot(robot_yaml)`` 按 ``robot`` 参数切换（规划节点启动时调用）。
当前支持 ``k100.yml`` / ``lycus.yml``。
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

K100_LEFT = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]
K100_RIGHT = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

LYCUS_LEFT = [
    "AR5-5_08L-W4C1C5-ZY2_joint_1",
    "AR5-5_08L-W4C1C5-ZY2_joint_2",
    "AR5-5_08L-W4C1C5-ZY2_joint_3",
    "AR5-5_08L-W4C1C5-ZY2_joint_4",
    "AR5-5_08L-W4C1C5-ZY2_joint_5",
    "AR5-5_08L-W4C1C5-ZY2_joint_6",
    "AR5-5_08L-W4C1C5-ZY2_joint_7",
]
LYCUS_RIGHT = [
    "AR5-5_08R-W4C1C5-ZY2_joint_1",
    "AR5-5_08R-W4C1C5-ZY2_joint_2",
    "AR5-5_08R-W4C1C5-ZY2_joint_3",
    "AR5-5_08R-W4C1C5-ZY2_joint_4",
    "AR5-5_08R-W4C1C5-ZY2_joint_5",
    "AR5-5_08R-W4C1C5-ZY2_joint_6",
    "AR5-5_08R-W4C1C5-ZY2_joint_7",
]

_CONTROLLER_SWITCH = {
    "both_arms": (
        ["both_arms_controller"],
        ["left_arm_controller", "right_arm_controller"],
    ),
    "left_arm": (
        ["left_arm_controller", "right_arm_controller"],
        ["both_arms_controller"],
    ),
    "right_arm": (
        ["left_arm_controller", "right_arm_controller"],
        ["both_arms_controller"],
    ),
}

_ROBOTS: Dict[str, Dict[str, object]] = {
    "k100": {
        "left": K100_LEFT,
        "right": K100_RIGHT,
        "ee_prefix": {
            "left_arm_ee_link": "left_",
            "right_arm_ee_link": "right_",
        },
    },
    "lycus": {
        "left": LYCUS_LEFT,
        "right": LYCUS_RIGHT,
        "ee_prefix": {
            "left_arm_ee_link": "AR5-5_08L-",
            "right_arm_ee_link": "AR5-5_08R-",
        },
    },
}

GROUP_JOINTS: Dict[str, List[str]] = {}
GROUP_EE_LINK: Dict[str, str] = {}
EE_TO_ARM_PREFIX: Dict[str, str] = {}
GROUP_ACTION: Dict[str, str] = {}
GROUP_CONTROLLER_SWITCH: Dict[str, Tuple[List[str], List[str]]] = {}
SUPPORTED_GROUPS: Tuple[str, ...] = ()
_EE_JOINTS: Dict[str, List[str]] = {}
ACTIVE_ROBOT = ""


def robot_id_from_yaml(robot: str) -> str:
    """``k100.yml`` / ``/path/lycus.yml`` → ``k100`` / ``lycus``。"""
    stem = os.path.basename(robot or "").rsplit(".", 1)[0].strip().lower()
    return stem or "robot"


def resolve_robot_id(robot: str) -> str:
    stem = robot_id_from_yaml(robot)
    if stem in _ROBOTS:
        return stem
    matches = [key for key in _ROBOTS if key in stem]
    if matches:
        return max(matches, key=len)
    raise ValueError(
        f"未知机器人配置: {robot}（识别为 '{stem}'）。"
        f"已支持: {sorted(_ROBOTS)}。请用 robot:=k100.yml 或 robot:=lycus.yml。"
    )


def apply_robot(robot: str) -> str:
    """Switch joint/controller tables to match a cuRobo robot yaml.

    Mutates the module-level dicts in place so existing ``from groups import
    GROUP_JOINTS`` bindings stay valid.
    """
    global ACTIVE_ROBOT, SUPPORTED_GROUPS
    robot_id = resolve_robot_id(robot)
    profile = _ROBOTS[robot_id]
    left = list(profile["left"])
    right = list(profile["right"])
    ee_prefix = dict(profile["ee_prefix"])
    ACTIVE_ROBOT = robot_id

    GROUP_JOINTS.clear()
    GROUP_JOINTS.update(
        {
            "left_arm": left,
            "right_arm": right,
            "both_arms": left + right,
        }
    )
    GROUP_EE_LINK.clear()
    GROUP_EE_LINK.update(
        {"left_arm": "left_arm_ee_link", "right_arm": "right_arm_ee_link"}
    )
    EE_TO_ARM_PREFIX.clear()
    EE_TO_ARM_PREFIX.update(ee_prefix)
    GROUP_ACTION.clear()
    GROUP_ACTION.update(
        {
            "left_arm": "/left_arm_controller/follow_joint_trajectory",
            "right_arm": "/right_arm_controller/follow_joint_trajectory",
            "both_arms": "/both_arms_controller/follow_joint_trajectory",
        }
    )
    GROUP_CONTROLLER_SWITCH.clear()
    GROUP_CONTROLLER_SWITCH.update(
        {k: (list(a), list(d)) for k, (a, d) in _CONTROLLER_SWITCH.items()}
    )
    _EE_JOINTS.clear()
    _EE_JOINTS.update(
        {
            "left_arm_ee_link": left,
            "right_arm_ee_link": right,
        }
    )
    SUPPORTED_GROUPS = tuple(GROUP_JOINTS.keys())
    return ACTIVE_ROBOT


def unknown_group(group_name: str) -> Optional[str]:
    if group_name not in GROUP_JOINTS:
        return f"未知的规划组: {group_name}，可选 {list(SUPPORTED_GROUPS)}"
    return None


def default_ee_link(group_name: str, ee_link: str) -> str:
    """单臂默认用该臂末端；双臂只给一个末端时默认右臂。"""
    ee_link = (ee_link or "").strip()
    if ee_link:
        return ee_link
    return GROUP_EE_LINK.get(group_name, "right_arm_ee_link")


def both_arm_ee_links() -> Tuple[str, str]:
    return GROUP_EE_LINK["left_arm"], GROUP_EE_LINK["right_arm"]


def resolve_ee_link(group_name: str, ee_link: str) -> Tuple[str, Optional[str]]:
    ee = default_ee_link(group_name, ee_link)
    expected = GROUP_EE_LINK.get(group_name)
    if expected is not None and ee != expected:
        return ee, f"{group_name} 的末端应为 {expected}，收到 {ee}"
    return ee, None


def joints_for_ee(ee_link: str, all_joint_names: List[str]) -> List[str]:
    listed = _EE_JOINTS.get(ee_link)
    if listed:
        names = set(all_joint_names)
        return [n for n in listed if n in names]
    prefix = EE_TO_ARM_PREFIX.get(ee_link)
    if prefix is None:
        lower = ee_link.lower()
        if "left" in lower or "-08l-" in lower:
            prefix = EE_TO_ARM_PREFIX.get("left_arm_ee_link", "left_")
        elif "right" in lower or "-08r-" in lower:
            prefix = EE_TO_ARM_PREFIX.get("right_arm_ee_link", "right_")
        else:
            return []
    return [n for n in all_joint_names if n.startswith(prefix)]
