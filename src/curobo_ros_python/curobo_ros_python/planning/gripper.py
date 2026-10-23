"""从反馈更新夹爪几何；夹爪与 mimic 手指都不占用优化自由度。"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from curobo.config_io import join_path, load_yaml
from curobo.content import get_assets_path, get_robot_configs_path

GRIPPER_JOINTS = ("left_gripper", "right_gripper")


@dataclass
class _Finger:
    source: str
    child: str
    origin: torch.Tensor
    direction: torch.Tensor
    multiplier: float
    offset: float


def _vector(element: Any, attribute: str, default: str) -> list[float]:
    return [float(v) for v in (default if element is None else element.get(attribute, default)).split()]


def _rotation(rpy: Sequence[float]) -> torch.Tensor:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return torch.tensor([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=torch.float32)


class GripperStateUpdater:
    """原地更新固定变换，保留求解器维数和 CUDA Graph 使用的张量地址。"""

    def __init__(self, urdf_path: str):
        joints = {j.get("name"): j for j in ET.parse(urdf_path).getroot().findall("joint")}
        self.joint_names = tuple(name for name in GRIPPER_JOINTS if name in joints)
        self._limits = {}
        self._fingers = []
        self._bindings = {}
        self._previous = None
        for name in self.joint_names:
            limit = joints[name].find("limit")
            if limit is None:
                raise ValueError(f"夹爪 {name} 缺少 URDF 限位")
            self._limits[name] = (float(limit.get("lower")), float(limit.get("upper")))
        for name, joint in joints.items():
            mimic = joint.find("mimic")
            source = name if name in self.joint_names else (None if mimic is None else mimic.get("joint"))
            if source not in self.joint_names:
                continue
            if joint.get("type") != "prismatic":
                raise ValueError(f"夹爪状态更新仅支持直线关节，收到 {name}")
            origin = joint.find("origin")
            rot = _rotation(_vector(origin, "rpy", "0 0 0"))
            mat = torch.zeros((3, 4), dtype=torch.float32)
            mat[:, :3] = rot
            mat[:, 3] = torch.tensor(_vector(origin, "xyz", "0 0 0"), dtype=torch.float32)
            axis = torch.tensor(_vector(joint.find("axis"), "xyz", "1 0 0"), dtype=torch.float32)
            norm = torch.linalg.vector_norm(axis)
            if float(norm) < 1e-8:
                raise ValueError(f"夹爪 {name} 的关节轴无效")
            self._fingers.append(_Finger(
                source, joint.find("child").get("link"), mat, rot @ (axis / norm),
                1.0 if mimic is None else float(mimic.get("multiplier", "1")),
                0.0 if mimic is None else float(mimic.get("offset", "0")),
            ))

    @classmethod
    def from_robot(cls, robot: str) -> GripperStateUpdater:
        data = load_yaml(join_path(str(get_robot_configs_path()), robot))
        path = data["robot_cfg"]["kinematics"]["urdf_path"]
        return cls(join_path(str(get_assets_path()), path))

    def update(self, values: Mapping[str, float], models: Sequence[Any]) -> bool:
        """使用米制关节位置更新所有 FK/碰撞副本；缺失反馈时拒绝使用默认开合。"""
        if not self.joint_names:
            return False
        missing = [name for name in self.joint_names if name not in values]
        if missing:
            raise ValueError(f"缺少夹爪状态 {missing}，请在 /joint_states 发布实测开合位置（米）")
        positions = {}
        for name, (lower, upper) in self._limits.items():
            q = float(values[name])
            if not math.isfinite(q) or q < lower - 1e-6 or q > upper + 1e-6:
                raise ValueError(f"夹爪 {name}={q} 超出 URDF 限位 [{lower}, {upper}] m")
            positions[name] = min(upper, max(lower, q))
        changed = positions != self._previous
        matrices = torch.stack([
            finger.origin.clone() for finger in self._fingers
        ])
        for i, finger in enumerate(self._fingers):
            q = positions[finger.source] * finger.multiplier + finger.offset
            matrices[i, :, 3] += finger.direction * q
        if not models:
            raise RuntimeError("没有可更新的夹爪运动学模型")
        seen = set()
        for model in models:
            kin = model.config.kinematics_config
            transforms = kin.fixed_transforms
            key = id(transforms)
            if any(name in model.joint_names for name in self.joint_names):
                raise RuntimeError("夹爪仍在优化关节中，请在机器人配置 lock_joints 中锁定夹爪")
            if key not in seen:
                seen.add(key)
                binding = self._bindings.get(key)
                if binding is None:
                    rows = [kin.link_name_to_idx_map[f.child] for f in self._fingers]
                    if any(int(kin.joint_map_type[row]) != -1 for row in rows):
                        raise RuntimeError("夹爪或 mimic 手指未锁定，拒绝修改活动关节变换")
                    indices = torch.tensor(rows, dtype=torch.long, device=transforms.device)
                    # 持有原张量引用，避免对象释放后 id 被重用。
                    binding = (transforms, indices)
                    self._bindings[key] = binding
                    refresh = True
                else:
                    refresh = changed
                if refresh:
                    transforms.index_copy_(0, binding[1], matrices.to(device=transforms.device, dtype=transforms.dtype))
            lock = model.lock_jointstate
            if lock is not None:
                for name, q in positions.items():
                    if name not in lock.joint_names:
                        continue
                    column = lock.joint_names.index(name)
                    lock.position[..., column] = q
                    for field in ("velocity", "acceleration", "jerk"):
                        tensor = getattr(lock, field, None)
                        if tensor is not None:
                            tensor[..., column] = 0.0
        self._previous = positions
        return changed
