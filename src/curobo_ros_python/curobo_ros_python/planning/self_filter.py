"""Current-state mesh self masking. Uncertain observations are retained."""
from __future__ import annotations

import math
import copy
import time
import numpy as np
import trimesh
import fcl
from scipy.spatial import cKDTree


class MeshSelfFilter:
    """Classify points using exact meshes, not planning spheres or convex hulls.

    Watertight meshes support inside tests. Open meshes only support distance
    tests; ambiguous interior points are retained. Model geometry is shared with
    OPT FCL, but this class never changes its objects or self-collision pairs.
    """

    def __init__(self, checker, group_name=None, *, vertex_tree_cache=None):
        if group_name is not None:
            from curobo_ros_python.planning.groups import GROUP_JOINTS
            if group_name not in GROUP_JOINTS:
                raise ValueError(f'未知本体过滤规划组: {group_name}')
            # Follow URDF descendants of the active arm joints, including tools
            # and fingers. Do not select shared torso/head ancestors as meshes.
            active_joints = set(GROUP_JOINTS[group_name])
            arm_links = set()
            for joint in checker.ordered:
                if (joint.get('name') in active_joints
                        or joint.find('parent').get('link') in arm_links):
                    arm_links.add(joint.find('child').get('link'))
            indices = [i for i, name in enumerate(checker.names) if name in arm_links]
            if not indices:
                raise ValueError(f'{group_name} 没有可用于过滤的手臂mesh')
            selected = copy.copy(checker)
            selected.names = [checker.names[i] for i in indices]
            selected.meshes = [checker.meshes[i] for i in indices]
            models = checker.mesh_models
            selected.mesh_models = [models[i] for i in indices] if models else []
            # FK still needs ancestors (e.g. torso), but never the other arm.
            by_child = {j.find('child').get('link'): j for j in checker.ordered}
            required_links = set()
            for link in selected.names:
                while link != checker.base_link:
                    required_links.add(link)
                    link = by_child[link].find('parent').get('link')
            selected.ordered = [j for j in checker.ordered
                                if j.find('child').get('link') in required_links]
            checker = selected
        self.checker = checker
        self.meshes = checker.meshes
        if not self.meshes or len(self.meshes) != len(checker.names):
            raise ValueError('缺少过滤mesh，请在启动时启用self_filter_mode并重新初始化模型')
        self.required = tuple(j.get('name') for j in checker.ordered
                              if j.get('type') != 'fixed' and j.find('mimic') is None)
        self.open_links = [n for n, m in zip(checker.names, self.meshes) if not m.is_watertight]
        # Startup builds left/right/both filters from the same retained meshes.
        # Share immutable vertex indices; FCL objects and query state stay local.
        if vertex_tree_cache is None:
            vertex_tree_cache = {}
        self.vertex_trees = []
        for mesh in self.meshes:
            key = id(mesh)
            if key not in vertex_tree_cache:
                vertex_tree_cache[key] = cKDTree(np.asarray(mesh.vertices))
            self.vertex_trees.append(vertex_tree_cache[key])
        models = checker.mesh_models
        if not models:
            models = []
            for mesh in self.meshes:
                model = fcl.BVHModel()
                model.beginModel(len(mesh.vertices), len(mesh.faces))
                model.addSubModel(np.asarray(mesh.vertices), np.asarray(mesh.faces))
                model.endModel()
                models.append(model)
        # Reuse BVHs, but own local objects: never modify OPT's world transforms.
        self.local_objects = [fcl.CollisionObject(model) for model in models]
        self._sphere = None
        self._sphere_padding = None

    def mask(self, points, positions, padding: float, budget_ms=None) -> np.ndarray:
        started = time.perf_counter()
        def check_budget():
            if budget_ms is not None and (time.perf_counter() - started) * 1000 > budget_ms:
                raise TimeoutError(f'本体过滤超出 {budget_ms:.0f}ms 预算，整帧不删除')
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if self._sphere_padding != padding:
            self._sphere = fcl.CollisionObject(fcl.Sphere(padding)) if padding > 0 else None
            self._sphere_padding = padding
        removed = np.zeros(len(pts), dtype=bool)
        transforms = self.checker._transforms(positions)
        request = fcl.CollisionRequest()
        for mesh, tree, obj, transform in zip(self.meshes, self.vertex_trees, self.local_objects, transforms):
            check_budget()
            # Transform points to the cached link-local mesh, preserving its R-tree.
            local = (pts - transform[:3, 3]) @ transform[:3, :3]
            bounds = mesh.bounds
            candidate = (~removed & np.isfinite(local).all(1)
                         & (local >= bounds[0] - padding).all(1)
                         & (local <= bounds[1] + padding).all(1))
            indices = np.flatnonzero(candidate)
            for start in range(0, len(indices), 64):
                check_budget()
                batch = indices[start:start + 64]
                query = local[batch]
                # Vertices are on the surface: distance <= padding is a valid
                # positive certificate, not an inflated approximation.
                vertex_distance, _ = tree.query(query)
                near = vertex_distance <= padding
                uncertain = np.flatnonzero(~near)
                if len(uncertain):
                    if padding == 0:
                        _, distance, _ = trimesh.proximity.closest_point(mesh, query[uncertain])
                        near[uncertain] = np.isfinite(distance) & (distance <= 0)
                    else:
                        # A sphere centred at the point intersects a triangle iff
                        # the point is within padding of that mesh surface.
                        for index in uncertain:
                            check_budget()
                            self._sphere.setTranslation(query[index])
                            result = fcl.CollisionResult()
                            fcl.collide(obj, self._sphere, request, result)
                            near[index] = result.is_collision
                if mesh.is_watertight:
                    unresolved = np.flatnonzero(~near)
                    if len(unresolved):
                        near[unresolved] = mesh.contains(query[unresolved])
                removed[batch] = near
        check_budget()
        return removed


class SelfFilterMixin:
    """Shared ROS point-cloud integration for OPT and MPC."""

    def _setup_self_filter(self) -> None:
        self.self_filter_mode = str(self._p('self_filter_mode'))
        self.self_filter_padding = float(self._p('self_filter_padding'))
        self.self_filter_max_processing_ms = float(self._p('self_filter_max_processing_ms'))
        if self.self_filter_mode not in ('off', 'observe', 'apply'):
            raise ValueError('self_filter_mode 必须为 off / observe / apply')
        if not math.isfinite(self.self_filter_padding) or not 0 <= self.self_filter_padding <= .05:
            raise ValueError('self_filter_padding 必须为 [0, 0.05] m')
        if not math.isfinite(self.self_filter_max_processing_ms) or self.self_filter_max_processing_ms <= 0:
            raise ValueError('self_filter_max_processing_ms 必须为有限正数')
        self._self_filter = None
        self._self_filters = {}
        self._self_filter_cache = None
        self._self_filter_publishers = {}
        self._self_filter_removed_points = np.zeros((0, 3), np.float32)
        self._last_prepared_body = np.zeros((0, 3), np.float32)
        if self.self_filter_mode != 'off':
            if self.pointcloud_parent_frame != self.base_frame.lstrip('/'):
                raise ValueError('启用本体过滤时 pointcloud_parent_frame 必须等于 base_frame')
            if self.base_frame.lstrip('/') != self.planner._mesh_checker.base_link:
                raise ValueError('本体过滤参考系必须与机器人模型base_link一致')
            from sensor_msgs.msg import PointCloud2
            for name in ('input', 'removed', 'environment'):
                self._self_filter_publishers[name] = self.create_publisher(
                    PointCloud2, '/curobo/self_filter/' + name, 1)
            self.get_logger().info(f'本体点云过滤={self.self_filter_mode}, 余量={self.self_filter_padding:.3f}m; '
                                   '使用当前关节反馈，失败保留原始点；apply可能误删近距离障碍')
            # Build indices during startup, not in the first MPC tick.
            self.get_logger().info('正在预构建本体过滤mesh索引，首次准备可能较慢')
            vertex_tree_cache = {}
            for group in ('left_arm', 'right_arm', 'both_arms'):
                filter_ = MeshSelfFilter(self.planner._mesh_checker, group,
                                         vertex_tree_cache=vertex_tree_cache)
                self._self_filters[group] = filter_
                for mesh in filter_.meshes:
                    if mesh.is_watertight or self.self_filter_padding == 0:
                        _ = mesh.triangles_tree
                self.get_logger().info(f'[本体过滤] group={group} meshes={len(filter_.meshes)} open_meshes={len(filter_.open_links)} ready=true')
                self.get_logger().debug(f'[本体过滤] group={group} links={filter_.checker.names}')
                if filter_.open_links:
                    self.get_logger().warn('本体过滤：非封闭mesh仅检查表面距离: ' + str(filter_.open_links))
            self._self_filter = self._self_filters['both_arms']

    def _publish_self_filter(self, msg, points, removed) -> None:
        from sensor_msgs.msg import PointCloud2, PointField
        from std_msgs.msg import Header
        arrays = {'input': points, 'removed': points[removed], 'environment': points[~removed]}
        for name, publisher in self._self_filter_publishers.items():
            if not publisher.get_subscription_count():
                continue
            data = np.asarray(arrays[name], dtype='<f4').reshape(-1, 3)
            cloud = PointCloud2(header=Header(stamp=msg.header.stamp, frame_id=self.pointcloud_parent_frame),
                height=1, width=len(data), is_bigendian=False, point_step=12, row_step=12 * len(data),
                fields=[PointField(name=n, offset=i * 4, datatype=PointField.FLOAT32, count=1)
                        for i, n in enumerate(('x', 'y', 'z'))],
                is_dense=bool(np.isfinite(data).all()), data=data.tobytes())
            publisher.publish(cloud)

    def _filter_robot_points(self, msg, points, group_name='both_arms', *, protected=None):
        self._self_filter_all_removed = False
        self._self_filter_succeeded = self.self_filter_mode == 'off'
        if self.self_filter_mode == 'off':
            self._self_filter_removed_points = np.zeros((0, 3), np.float32)
            return points
        started = time.perf_counter()
        try:
            if group_name not in ('left_arm', 'right_arm', 'both_arms'):
                raise ValueError(f'未知本体过滤规划组: {group_name}')
            self._self_filter = self._self_filters[group_name]
            current = dict(self._joint_map)
            positions = {}
            for name in self._self_filter.required:
                if name not in current or not math.isfinite(float(current[name])):
                    raise ValueError(f'{name} 缺少有效的当前关节反馈')
                positions[name] = float(current[name])
            # Repeated geometry can reuse its mask even when ROS creates a new message.
            cached = self._self_filter_cache
            key = (group_name, tuple(sorted(positions.items())), self.self_filter_padding)
            cache_hit=(cached is not None and cached[0] == key
                       and np.array_equal(cached[1], points))
            if cache_hit:
                removed = cached[2]
            else:
                removed = self._self_filter.mask(points, positions, self.self_filter_padding,
                    budget_ms=self.self_filter_max_processing_ms)
                self._self_filter_cache = (key, points.copy(), removed)
            candidate_count=int(removed.sum())
            protected_count=0
            if protected is not None:
                protected=np.asarray(protected,dtype=bool)
                if protected.shape!=removed.shape: raise ValueError('工件标签与点云长度不一致')
                protected_count=int((removed&protected).sum())
                removed=removed&~protected
            self._publish_self_filter(msg, points, removed)
            self._self_filter_removed_points = np.ascontiguousarray(points[removed], dtype=np.float32).reshape(-1, 3)
            self._self_filter_succeeded = True
            count = int(removed.sum())
            self._self_filter_all_removed = bool(len(points) and count == len(points))
            deleted=count if self.self_filter_mode=='apply' else 0
            self.get_logger().info(f'[本体过滤] mode={self.self_filter_mode} group={group_name} input={len(points)} '
                f'self_candidates={candidate_count} protected_objects={protected_count} removed={deleted} output={len(points)-deleted} cache={"hit" if cache_hit else "miss"} '
                f'elapsed_ms={(time.perf_counter()-started)*1000:.1f}', throttle_duration_sec=2.0)
            if len(points) and count / len(points) > .8:
                self.get_logger().warn(f'[本体过滤] mode={self.self_filter_mode} group={group_name} '
                                       f'本体候选比例={count/len(points):.1%}，请检查标定和模型；observe不删规划点',
                                       throttle_duration_sec=2.0)
            return points[~removed] if self.self_filter_mode == 'apply' else points
        except Exception as exc:
            self._self_filter_succeeded = False
            self._self_filter_all_removed = False
            self._self_filter_cache = None
            self._self_filter_removed_points = np.zeros((0, 3), np.float32)
            self.get_logger().warn(f'[本体过滤] mode={self.self_filter_mode} group={group_name} status=跳过 '
                f'kept={len(points)} reason={str(exc)!r}', throttle_duration_sec=2.0)
            self._publish_self_filter(msg, points, np.zeros(len(points), dtype=bool))
            return points
