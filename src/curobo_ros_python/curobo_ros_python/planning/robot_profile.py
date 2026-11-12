"""Robot-specific names, loaded from config/robots/<id>.yaml.

One file per robot. Code asks for the active robot instead of keeping its own
copy of joint names, gripper links or camera frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class RobotProfile:
    robot_id: str
    left_joints: tuple
    right_joints: tuple
    ee_links: dict
    ee_joint_prefix: dict
    gripper_joints_by_group: dict[str, str]
    gripper_controllers: dict
    posture_joints: tuple
    gripper_base_link: str
    jaw_links: tuple
    payload_link: str
    fcl_self_collision: str
    camera_parent_frame: str
    camera_optical_frame: str
    posture_l: dict

    @property
    def posture_tables(self) -> dict:
        """Named nullspace targets. ``zero`` is 0 for every joint listed in ``l``."""
        return {"l": dict(self.posture_l), "zero": dict.fromkeys(self.posture_l, 0.0)}

    @property
    def gripper_joints(self) -> tuple[str, ...]:
        return tuple(self.gripper_joints_by_group.values())

    def gripper_joint_name(self, group: str) -> str:
        name = self.gripper_joints_by_group.get(group)
        if name is None:
            raise ValueError(f"{self.robot_id} 的 {group} 未配置夹爪，不支持抓放 Action")
        return name

    def ee_link_name(self, group: str) -> str:
        return self.ee_links[group]

    def payload_link_name(self, group: str) -> str:
        return self.payload_link.format(group=group)

    def jaw_link_names(self, side: str) -> tuple:
        return tuple(template.format(side=side) for template in self.jaw_links)

    def gripper_base_name(self, side: str) -> str:
        return self.gripper_base_link.format(side=side) if self.gripper_base_link else ""


def config_dir() -> Path:
    """Package config directory, from the source tree or the installed share."""
    source = Path(__file__).resolve().parents[2] / "config"
    if (source / "robots").is_dir():
        return source
    from ament_index_python.packages import get_package_share_directory

    return Path(get_package_share_directory("curobo_ros_python")) / "config"


_PROFILES: dict[str, RobotProfile] = {}


def _as_tuple(data: dict, key: str) -> tuple:
    value = data.get(key) or []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{data.get('robot_id')} 的 {key} 必须是字符串列表")
    return tuple(value)


def _load_one(path: Path) -> RobotProfile:
    data = yaml.safe_load(path.read_text())
    robot_id = str(data.get("robot_id") or "").strip()
    if not robot_id:
        raise ValueError(f"{path} 缺少 robot_id")
    goals = data.get("posture_goals") or {}
    if not isinstance(goals, dict):
        raise ValueError(f"{robot_id} 的 posture_goals 必须是映射")
    posture_l = {str(name): float(value) for name, value in (goals.get("l") or {}).items()}
    controllers = data.get("gripper_controllers") or {}
    grippers = data.get("gripper_joints") or {}
    if not isinstance(grippers, dict) or any(
        group not in ("left_arm", "right_arm") or not isinstance(joint, str)
        or joint not in controllers
        for group, joint in grippers.items()
    ):
        raise ValueError(f"{robot_id} 的 gripper_joints 必须将手臂映射到已配置控制器的夹爪关节")
    return RobotProfile(
        robot_id=robot_id,
        left_joints=_as_tuple(data, "left_joints"),
        right_joints=_as_tuple(data, "right_joints"),
        ee_links={str(k): str(v) for k, v in (data.get("ee_links") or {}).items()},
        ee_joint_prefix={str(k): str(v) for k, v in (data.get("ee_joint_prefix") or {}).items()},
        gripper_joints_by_group=dict(grippers),
        gripper_controllers={str(k): str(v) for k, v in controllers.items()},
        posture_joints=_as_tuple(data, "posture_joints"),
        gripper_base_link=str(data.get("gripper_base_link") or ""),
        jaw_links=_as_tuple(data, "jaw_links"),
        payload_link=str(data.get("payload_link") or ""),
        fcl_self_collision=str(data.get("fcl_self_collision") or ""),
        camera_parent_frame=str(data.get("camera_parent_frame") or ""),
        camera_optical_frame=str(data.get("camera_optical_frame") or ""),
        posture_l=posture_l,
    )


def profiles() -> dict[str, RobotProfile]:
    if not _PROFILES:
        directory = config_dir() / "robots"
        for path in sorted(directory.glob("*.yaml")):
            profile = _load_one(path)
            _PROFILES[profile.robot_id] = profile
        if not _PROFILES:
            raise FileNotFoundError(f"找不到机器人配置目录: {directory}")
    return _PROFILES


def robot_profile(robot: str) -> RobotProfile:
    """``k100.yml`` or ``/path/lycus.yml`` resolves to that robot's profile."""
    stem = Path(robot or "").name.split(".")[0].strip().lower()
    found = profiles()
    if stem in found:
        return found[stem]
    matches = [key for key in found if key in stem]
    if matches:
        return found[max(matches, key=len)]
    raise ValueError(f"未知机器人配置: {robot}（识别为 '{stem}'）。已支持: {sorted(found)}。")


def active_profile() -> RobotProfile:
    """Profile of the robot selected at startup. Lycus names before that."""
    from curobo_ros_python.planning.groups import ACTIVE_ROBOT

    return robot_profile(ACTIVE_ROBOT or "lycus")
