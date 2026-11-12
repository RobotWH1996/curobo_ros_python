"""Cached CPU FCL surface-intersection checks for candidate joint states."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import itertools
import time
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

from curobo_ros_python.planning.motion_collision import moving_links


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

    def __init__(self, urdf_path: str, config: Mapping[str, Any], retain_meshes: bool = False, logger=None):
        started = time.perf_counter()
        timing = dict(mesh_load=0., transform=0., merge=0., bvh=0., objects=0., bounds=0.)
        if logger is not None:
            logger.info("[FCL初始化] event=开始")
        import fcl
        self.fcl = fcl
        path = Path(urdf_path)
        root = ET.parse(path).getroot()
        timing["urdf"] = time.perf_counter() - started
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
        self.names, self.objects, self.meshes, corners = [], [], [], []
        self.mesh_models = []
        self.skipped_links = []
        for name in config['collision_link_names']:
            link_started = time.perf_counter()
            link_timing = dict.fromkeys(('mesh_load', 'transform', 'merge', 'bvh', 'objects', 'bounds'), 0.)
            link = root.find(f"link[@name='{name}']")
            if link is None:
                raise ValueError(f'URDF missing configured collision link: {name}')
            parts = []
            for collision in link.findall('collision'):
                load_started = time.perf_counter()
                geometry = collision.find('geometry')
                mesh_element = geometry.find('mesh')
                if mesh_element is not None:
                    filename = mesh_element.get('filename')
                    if '://' in filename:
                        raise ValueError(f'Unresolved mesh URI: {filename}')
                    mesh = trimesh.load(path.parent / filename, force='mesh')
                elif geometry.find('box') is not None:
                    mesh = trimesh.creation.box(np.fromstring(geometry.find('box').get('size'), sep=' '))
                elif geometry.find('sphere') is not None:
                    mesh = trimesh.creation.icosphere(radius=float(geometry.find('sphere').get('radius')))
                elif geometry.find('cylinder') is not None:
                    e = geometry.find('cylinder')
                    mesh = trimesh.creation.cylinder(radius=float(e.get('radius')), height=float(e.get('length')))
                else:
                    raise ValueError(f'Unsupported collision geometry: {name}')
                transform_started = time.perf_counter()
                link_timing['mesh_load'] += transform_started - load_started
                if mesh_element is not None:
                    mesh.apply_scale(np.fromstring(mesh_element.get('scale', '1 1 1'), sep=' '))
                mesh.apply_transform(_origin(collision.find('origin')))
                link_timing['transform'] += time.perf_counter() - transform_started
                parts.append(mesh)
            if not parts:
                self.skipped_links.append(name)
                continue
            if name not in known:
                raise ValueError(f'Collision link disconnected from {self.base_link}: {name}')
            merge_started = time.perf_counter()
            mesh = trimesh.util.concatenate(parts)
            link_timing['merge'] = time.perf_counter() - merge_started
            self.names.append(name)
            if retain_meshes:
                self.meshes.append(mesh)
            bvh_started = time.perf_counter()
            model = self._model(mesh)
            link_timing['bvh'] = time.perf_counter() - bvh_started
            if retain_meshes:
                self.mesh_models.append(model)
            object_started = time.perf_counter()
            self.objects.append(self.fcl.CollisionObject(model))
            link_timing['objects'] = time.perf_counter() - object_started
            bounds_started = time.perf_counter()
            corners.append(np.array(list(itertools.product(*zip(*mesh.bounds)))))
            link_timing['bounds'] = time.perf_counter() - bounds_started
            for key, value in link_timing.items():
                timing[key] += value
            if logger is not None:
                details = ' '.join(f'{key}_ms={value*1000:.1f}' for key, value in link_timing.items())
                logger.info(f'[FCL初始化] event=连杆 link={name} vertices={len(mesh.vertices)} '
                            f'faces={len(mesh.faces)} {details} total_ms={(time.perf_counter()-link_started)*1000:.1f}')
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
        self.environment_ignore_links = set()
        self.payload = None
        self.skip_fcl_checks = False
        self._moving_links = None
        self._motion_masks = {}
        self._moving_mask = np.ones(len(self.names), dtype=bool)
        self._motion_pairs = self.pairs
        total = time.perf_counter() - started
        timing["setup"] = total - sum(timing.values())
        if logger is not None:
            details = " ".join(f"{key}_ms={value*1000:.1f}" for key, value in timing.items())
            logger.info(f"[FCL初始化] event=完成 meshes={len(self.names)} {details} total_ms={total*1000:.1f}")

    def set_motion_joints(self, joint_names) -> None:
        """Select pairs involving a moving link; None restores full-robot checks."""
        key = None if joint_names is None else tuple(sorted(joint_names))
        if key not in self._motion_masks:
            links = None if key is None else moving_links(self.joints.values(), key)
            mask = np.array([links is None or name in links for name in self.names])
            pairs = self.pairs[(mask[self.pairs[:, 0]] | mask[self.pairs[:, 1]])]
            self._motion_masks[key] = (links, mask, pairs)
        self._moving_links, self._moving_mask, self._motion_pairs = self._motion_masks[key]

    @staticmethod
    def _fcl_policy_path(name: str) -> Path:
        """FCL 忽略表放在包的 config/，与 GPU 球体表分开。"""
        from curobo_ros_python.planning.robot_profile import config_dir

        path = config_dir() / name
        if path.is_file():
            return path
        raise FileNotFoundError(f'找不到 FCL 自碰撞表 {name}: {path}')

    @classmethod
    def from_robot(cls, robot: str, retain_meshes: bool = False, logger=None) -> MeshCollisionChecker:
        started = time.perf_counter()
        from curobo.config_io import join_path, load_yaml
        from curobo.content import get_assets_path, get_robot_configs_path
        config = load_yaml(join_path(str(get_robot_configs_path()), robot))['robot_cfg']['kinematics']
        # Packaged OPT policy is independent of the GPU sphere policy. Other
        # robots retain their existing policy until a dedicated table is supplied.
        from curobo_ros_python.planning.robot_profile import robot_profile

        policy_name = robot_profile(robot).fcl_self_collision
        if policy_name:
            import yaml
            policy_path = cls._fcl_policy_path(policy_name)
            policy = yaml.safe_load(policy_path.read_text())
            ignored = policy['self_collision_ignore']
            if not isinstance(ignored, dict) or any(
                not isinstance(names, list) for names in ignored.values()
            ):
                raise ValueError(f'Invalid FCL exclusion policy: {policy_path}')
            config = dict(config, self_collision_ignore=ignored)
        if logger is not None:
            logger.info(f'[FCL初始化] event=配置完成 elapsed_ms={(time.perf_counter()-started)*1000:.1f}')
        return cls(join_path(str(get_assets_path()), config['urdf_path']), config,
                   retain_meshes=retain_meshes, logger=logger)

    def _object(self, mesh: trimesh.Trimesh) -> Any:
        return self.fcl.CollisionObject(self._model(mesh))

    def _model(self, mesh: trimesh.Trimesh) -> Any:
        if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
            raise ValueError('Invalid collision mesh')
        model = self.fcl.BVHModel()
        faces = np.asarray(mesh.faces, dtype=np.int32).reshape(-1, 3)
        model.beginModel(len(mesh.vertices), len(faces))
        model.addSubModel(np.asarray(mesh.vertices), faces)
        model.endModel()
        return model

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

    def _transforms(self, positions: Mapping[str, float], link_names=None) -> np.ndarray:
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

        ordered = self.ordered
        if link_names is not None:
            by_child = {j.find('child').get('link'): j for j in self.joints.values()}
            ordered = []
            seen = set()
            for link in link_names:
                path = []
                while link != self.base_link:
                    joint = by_child[link]
                    path.append(joint)
                    link = joint.find('parent').get('link')
                for joint in reversed(path):
                    if joint.get('name') not in seen:
                        seen.add(joint.get('name'))
                        ordered.append(joint)
        for joint in ordered:
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
        return np.array([transforms[name] for name in (self.names if link_names is None else link_names)])

    def check(self, positions: Mapping[str, float]) -> MeshCollision | None:
        if self.skip_fcl_checks:
            return None
        transforms = self._transforms(positions)
        for obj, transform in zip(self.objects, transforms):
            obj.setTransform(self.fcl.Transform(transform[:3, :3], transform[:3, 3]))
        points = np.einsum('nij,nkj->nki', transforms[:, :3, :3], self.corners) + transforms[:, None, :3, 3]
        low, high = points.min(1), points.max(1)

        def intersects(first: Any, second: Any) -> bool:
            result = self.fcl.CollisionResult()
            self.fcl.collide(first, second, self.fcl.CollisionRequest(), result)
            return bool(result.is_collision)

        pairs = self._motion_pairs
        overlap = ((high[pairs[:, 0]] >= low[pairs[:, 1]]) &
                   (high[pairs[:, 1]] >= low[pairs[:, 0]])).all(1)
        for i, j in pairs[overlap]:
            if intersects(self.objects[i], self.objects[j]):
                return MeshCollision('self', self.names[i], self.names[j])
        for name, (obj, bounds) in self.environment.items():
            overlap = self._moving_mask & ((high >= bounds[0]) & (bounds[1] >= low)).all(1)
            for i in np.flatnonzero(overlap):
                if self.names[i] in self.environment_ignore_links:
                    continue
                if intersects(self.objects[i], obj):
                    return MeshCollision('environment', self.names[i], name)
        return self._check_payload(positions, transforms)

    def _check_payload(self, positions, robot_transforms):
        payload = self.payload
        if payload is None:
            return None
        from curobo_ros_python.planning.payload import outside_box
        transform = self._transforms(positions, [payload.tool])[0] @ payload.in_tool
        obj = self.fcl.CollisionObject(self.fcl.Box(*payload.size),
            self.fcl.Transform(transform[:3, :3], transform[:3, 3]))
        from curobo_ros_python.planning.robot_profile import active_profile

        side = payload.group.split('_')[0]
        profile = active_profile()
        allowed = {payload.tool, profile.gripper_base_name(side), *profile.jaw_link_names(side)}
        payload_moves = self._moving_links is None or payload.tool in self._moving_links
        for name, robot_obj in zip(self.names, self.objects):
            if name in allowed or (not payload_moves and name not in self._moving_links):
                continue
            result = self.fcl.CollisionResult()
            if self.fcl.collide(obj, robot_obj, self.fcl.CollisionRequest(), result):
                return MeshCollision('payload_self', 'payload', name)
        for name, (environment_obj, bounds) in (self.environment.items() if payload_moves else ()):
            result = self.fcl.CollisionResult()
            if self.fcl.collide(obj, environment_obj, self.fcl.CollisionRequest(), result):
                return MeshCollision('payload_environment', 'payload', name)
        return None
