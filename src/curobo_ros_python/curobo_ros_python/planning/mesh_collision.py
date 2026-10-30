"""Cached CPU FCL surface-intersection checks for candidate joint states."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import itertools
import xml.etree.ElementTree as ET

import numpy as np
import trimesh


@dataclass(frozen=True)
class MeshCollision:
    kind: str
    first: str
    second: str


def _origin(element: Any) -> np.ndarray:
    if element is None:
        return np.eye(4)
    matrix = trimesh.transformations.euler_matrix(
        *np.fromstring(element.get('rpy', '0 0 0'), sep=' '))
    matrix[:3, 3] = np.fromstring(element.get('xyz', '0 0 0'), sep=' ')
    return matrix


class MeshCollisionChecker:
    """Check configured robot collision links and a replaceable mesh scene.

    BVHs and FCL objects are reused. Pair exclusions are explicit robot config;
    this checks triangle intersections, not continuous motion or containment.
    Instances are used under the planner's existing serialized request handling.
    """

    def __init__(self, urdf_path: str, config: Mapping[str, Any]):
        import fcl
        self.fcl = fcl
        path = Path(urdf_path)
        root = ET.parse(path).getroot()
        self.base_link = config['base_link']
        self.defaults = dict(zip(config['cspace']['joint_names'],
                                 config['cspace']['default_joint_position']))
        self.defaults.update(config.get('lock_joints', {}))
        self.joints = {joint.get('name'): joint for joint in root.findall('joint')}
        self.ordered = []
        known = {self.base_link}
        pending = list(self.joints.values())
        while pending:
            ready = [j for j in pending if j.find('parent').get('link') in known]
            if not ready:
                break
            for joint in ready:
                self.ordered.append(joint)
                known.add(joint.find('child').get('link'))
                pending.remove(joint)
        self.origins = {j.get('name'): _origin(j.find('origin')) for j in self.ordered}
        self.names, self.objects, corners = [], [], []
        self.skipped_links = []
        for name in config['collision_link_names']:
            link = root.find(f"link[@name='{name}']")
            if link is None:
                raise ValueError(f'URDF missing configured collision link: {name}')
            parts = []
            for collision in link.findall('collision'):
                geometry = collision.find('geometry')
                mesh_element = geometry.find('mesh')
                if mesh_element is not None:
                    filename = mesh_element.get('filename')
                    if '://' in filename:
                        raise ValueError(f'Unresolved mesh URI: {filename}')
                    mesh = trimesh.load(path.parent / filename, force='mesh')
                    mesh.apply_scale(np.fromstring(mesh_element.get('scale', '1 1 1'), sep=' '))
                elif geometry.find('box') is not None:
                    mesh = trimesh.creation.box(np.fromstring(geometry.find('box').get('size'), sep=' '))
                elif geometry.find('sphere') is not None:
                    mesh = trimesh.creation.icosphere(radius=float(geometry.find('sphere').get('radius')))
                elif geometry.find('cylinder') is not None:
                    e = geometry.find('cylinder')
                    mesh = trimesh.creation.cylinder(radius=float(e.get('radius')), height=float(e.get('length')))
                else:
                    raise ValueError(f'Unsupported collision geometry: {name}')
                mesh.apply_transform(_origin(collision.find('origin')))
                parts.append(mesh)
            if not parts:
                self.skipped_links.append(name)
                continue
            if name not in known:
                raise ValueError(f'Collision link disconnected from {self.base_link}: {name}')
            mesh = trimesh.util.concatenate(parts)
            self.names.append(name)
            self.objects.append(self._object(mesh))
            corners.append(np.array(list(itertools.product(*zip(*mesh.bounds)))))
        if not self.names:
            raise ValueError('No robot collision geometry')
        # Do not require feedback for unrelated branches such as wheel joints
        # when those links are outside the configured collision model.
        by_child = {j.find('child').get('link'): j for j in self.ordered}
        required = set()
        for link in self.names:
            while link != self.base_link:
                required.add(link)
                link = by_child[link].find('parent').get('link')
        self.ordered = [j for j in self.ordered if j.find('child').get('link') in required]
        self.corners = np.asarray(corners)
        ignored = {frozenset((a, b)) for a, bs in config.get('self_collision_ignore', {}).items() for b in bs}
        self.pairs = np.array([(i, j) for i, j in itertools.combinations(range(len(self.names)), 2)
                               if frozenset((self.names[i], self.names[j])) not in ignored], dtype=int).reshape(-1, 2)
        self.environment: dict[str, tuple[Any, np.ndarray]] = {}

    @classmethod
    def from_robot(cls, robot: str) -> MeshCollisionChecker:
        from curobo.config_io import join_path, load_yaml
        from curobo.content import get_assets_path, get_robot_configs_path
        config = load_yaml(join_path(str(get_robot_configs_path()), robot))['robot_cfg']['kinematics']
        # Packaged OPT policy is independent of the GPU sphere policy. Other
        # robots retain their existing policy until a dedicated table is supplied.
        if Path(robot).stem == 'lycus':
            import yaml
            policy_path = Path(__file__).with_name('lycus_fcl_collision.yaml')
            policy = yaml.safe_load(policy_path.read_text())
            ignored = policy['self_collision_ignore']
            if not isinstance(ignored, dict) or any(
                not isinstance(names, list) for names in ignored.values()
            ):
                raise ValueError(f'Invalid FCL exclusion policy: {policy_path}')
            config = dict(config, self_collision_ignore=ignored)
        return cls(join_path(str(get_assets_path()), config['urdf_path']), config)

    def _object(self, mesh: trimesh.Trimesh) -> Any:
        if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
            raise ValueError('Invalid collision mesh')
        model = self.fcl.BVHModel()
        model.beginModel(len(mesh.vertices), len(mesh.faces))
        model.addSubModel(np.asarray(mesh.vertices), np.asarray(mesh.faces))
        model.endModel()
        return self.fcl.CollisionObject(model)

    def prepare_scene(self, scene: Any) -> dict[str, tuple[Any, np.ndarray]]:
        """Build a replacement scene before mutating either collision backend."""
        if scene is None:
            return {}
        if isinstance(scene, list):
            raise ValueError('FCL endpoint checks support one scene only')
        if getattr(scene, 'voxel', None):
            raise ValueError('FCL needs source point-cloud mesh, not an ESDF-only scene')
        objects = {}
        for obstacle in scene.objects:
            mesh = obstacle.get_trimesh_mesh(process=False, transform_with_pose=True)
            objects[obstacle.name] = (self._object(mesh), np.asarray(mesh.bounds))
        return objects

    def _transforms(self, positions: Mapping[str, float]) -> np.ndarray:
        values = dict(self.defaults)
        values.update(positions)
        transforms = {self.base_link: np.eye(4)}

        def value(name: str, seen: tuple[str, ...] = ()) -> float:
            if name in seen:
                raise ValueError(f'Cyclic mimic joint: {name}')
            joint = self.joints[name]
            mimic = joint.find('mimic')
            if mimic is not None:
                q = value(mimic.get('joint'), seen + (name,)) * float(mimic.get('multiplier', 1)) + float(mimic.get('offset', 0))
            elif joint.get('type') == 'fixed':
                q = 0.
            elif name in values:
                q = float(values[name])
            else:
                raise ValueError(f'Missing joint state: {name}')
            if not np.isfinite(q):
                raise ValueError(f'Non-finite joint state: {name}')
            return q

        for joint in self.ordered:
            name = joint.get('name')
            typ = joint.get('type')
            q = value(name)
            transform = self.origins[name]
            axis_element = joint.find('axis')
            axis = np.fromstring(axis_element.get('xyz', '1 0 0') if axis_element is not None else '1 0 0', sep=' ')
            if typ in ('revolute', 'continuous'):
                transform = transform @ trimesh.transformations.rotation_matrix(q, axis)
            elif typ == 'prismatic':
                delta = np.eye(4)
                delta[:3, 3] = q * axis / np.linalg.norm(axis)
                transform = transform @ delta
            elif typ != 'fixed':
                raise ValueError(f'Unsupported joint type: {typ}')
            transforms[joint.find('child').get('link')] = transforms[joint.find('parent').get('link')] @ transform
        return np.array([transforms[name] for name in self.names])

    def check(self, positions: Mapping[str, float]) -> MeshCollision | None:
        transforms = self._transforms(positions)
        for obj, transform in zip(self.objects, transforms):
            obj.setTransform(self.fcl.Transform(transform[:3, :3], transform[:3, 3]))
        points = np.einsum('nij,nkj->nki', transforms[:, :3, :3], self.corners) + transforms[:, None, :3, 3]
        low, high = points.min(1), points.max(1)

        def intersects(first: Any, second: Any) -> bool:
            result = self.fcl.CollisionResult()
            self.fcl.collide(first, second, self.fcl.CollisionRequest(), result)
            return bool(result.is_collision)

        pairs = self.pairs
        overlap = ((high[pairs[:, 0]] >= low[pairs[:, 1]]) &
                   (high[pairs[:, 1]] >= low[pairs[:, 0]])).all(1)
        for i, j in pairs[overlap]:
            if intersects(self.objects[i], self.objects[j]):
                return MeshCollision('self', self.names[i], self.names[j])
        for name, (obj, bounds) in self.environment.items():
            overlap = ((high >= bounds[0]) & (bounds[1] >= low)).all(1)
            for i in np.flatnonzero(overlap):
                if intersects(self.objects[i], obj):
                    return MeshCollision('environment', self.names[i], name)
        return None
