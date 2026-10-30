"""Feed measured fixed joints into every GPU kinematics copy in place."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import math
import xml.etree.ElementTree as ET
import torch

from curobo_ros_python.planning.gripper import _rotation, _vector


@dataclass
class _FixedJoint:
    name: str
    child: str
    origin: torch.Tensor
    axis: torch.Tensor
    kind: str


class FixedJointStateUpdater:
    """Update locked revolute/prismatic joints without making them optimized DOFs."""

    def __init__(self, urdf_path: str, defaults: Mapping[str, float]):
        joints = {j.get('name'): j for j in ET.parse(urdf_path).getroot().findall('joint')}
        self.positions = {name: float(q) for name, q in defaults.items()}
        self._joints = []
        self._bindings = {}
        for name in defaults:
            joint = joints[name]
            kind = joint.get('type')
            if kind not in ('revolute', 'continuous', 'prismatic') or joint.find('mimic') is not None:
                raise ValueError(f'Unsupported fixed feedback joint: {name} ({kind})')
            origin = torch.zeros(3, 4, dtype=torch.float32)
            origin[:, :3] = _rotation(_vector(joint.find('origin'), 'rpy', '0 0 0'))
            origin[:, 3] = torch.tensor(_vector(joint.find('origin'), 'xyz', '0 0 0'), dtype=torch.float32)
            axis = torch.tensor(_vector(joint.find('axis'), 'xyz', '1 0 0'), dtype=torch.float32)
            norm = float(axis.norm())
            if not math.isfinite(norm) or norm < 1e-8:
                raise ValueError(f'Invalid fixed joint axis: {name}')
            self._joints.append(_FixedJoint(name, joint.find('child').get('link'), origin, axis / norm, kind))

    @classmethod
    def from_robot(cls, robot: str, excluded_names: Sequence[str]) -> FixedJointStateUpdater:
        from curobo.config_io import join_path, load_yaml
        from curobo.content import get_assets_path, get_robot_configs_path
        config = load_yaml(join_path(str(get_robot_configs_path()), robot))['robot_cfg']['kinematics']
        cspace = config['cspace']
        defaults = dict(zip(cspace['joint_names'], cspace['default_joint_position']))
        defaults.update(config.get('lock_joints', {}))
        excluded = set(excluded_names)
        defaults = {name: q for name, q in defaults.items() if name not in excluded}
        return cls(join_path(str(get_assets_path()), config['urdf_path']), defaults)

    def _matrix(self, joint: _FixedJoint, q: float) -> torch.Tensor:
        matrix = joint.origin.clone()
        if joint.kind == 'prismatic':
            matrix[:, 3] += matrix[:, :3] @ (joint.axis * q)
        else:
            x, y, z = joint.axis.tolist()
            skew = torch.tensor([[0., -z, y], [z, 0., -x], [-y, x, 0.]], dtype=joint.origin.dtype)
            rotation = torch.eye(3, dtype=joint.origin.dtype) + math.sin(q) * skew + (1. - math.cos(q)) * (skew @ skew)
            matrix[:, :3] = matrix[:, :3] @ rotation
        return matrix

    def update(self, values: Mapping[str, float], models: Sequence[Any]) -> bool:
        positions = dict(self.positions)
        for name in positions:
            if name in values:
                positions[name] = float(values[name])
            if not math.isfinite(positions[name]):
                raise ValueError(f'Non-finite fixed joint feedback: {name}')
        changed = positions != self.positions
        if not self._joints:
            return False
        if not models:
            raise RuntimeError('No kinematics models to update fixed joints')
        # Validate every binding before changing any geometry.
        bindings = []
        for model in models:
            if any(j.name in model.joint_names for j in self._joints):
                raise RuntimeError('Feedback-only joint unexpectedly participates in optimization')
            kin = model.config.kinematics_config
            transforms = kin.fixed_transforms
            key = id(transforms)
            binding = self._bindings.get(key)
            if binding is None:
                rows = [kin.link_name_to_idx_map[j.child] for j in self._joints]
                if any(int(kin.joint_map_type[row]) != -1 for row in rows):
                    raise RuntimeError('Cannot overwrite an active joint transform')
                binding = (transforms, rows, None)
            bindings.append((key, binding, model))
        matrices = None
        for key, (transforms, rows, applied), model in bindings:
            # The same tensor may be shared by several rollout models.
            existing = self._bindings.get(key)
            if existing is not None:
                applied = existing[2]
            if applied != positions:
                if matrices is None:
                    matrices = torch.stack([self._matrix(j, positions[j.name]) for j in self._joints])
                for row, matrix in zip(rows, matrices):
                    transforms[row].copy_(matrix.to(device=transforms.device, dtype=transforms.dtype))
                self._bindings[key] = (transforms, rows, dict(positions))
            lock = model.lock_jointstate
            if lock is not None:
                for name, q in positions.items():
                    if name not in lock.joint_names:
                        continue
                    column = lock.joint_names.index(name)
                    lock.position[..., column] = q
                    for field in ('velocity', 'acceleration', 'jerk'):
                        tensor = getattr(lock, field, None)
                        if tensor is not None:
                            tensor[..., column] = 0.
        self.positions = positions
        return changed
