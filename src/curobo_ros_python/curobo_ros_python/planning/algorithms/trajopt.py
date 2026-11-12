"""cuRobo 规划封装：位姿规划、关节规划、IK。机器人由 ``robot`` YAML 决定。

只吃 / 吐 cuRobo 类型。ROS 消息和 RViz 绿线不进这里。
"""

from __future__ import annotations

import math
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from curobo.config_io import join_path, load_yaml
from curobo.content import get_assets_path, get_task_configs_path
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg, format_ik_failure, format_plan_failure
from curobo.types import GoalToolPose, JointState, Pose

from curobo_ros_python.planning.algorithms.common import (
    pop_cached_mesh,
    restore_cached_mesh,
    robot_cfg_with_locked_joints,
    squeeze_pose,
)
from curobo_ros_python.planning.groups import (
    GROUP_JOINTS,
    apply_robot,
    both_arm_ee_links,
    joints_for_ee,
    resolve_ee_link,
    unknown_group,
)
from curobo_ros_python.planning.gripper import GripperStateUpdater
from curobo_ros_python.planning.tool_z_vertical import install_vertical_cost, vertical_tool_scope
from curobo_ros_python.planning.fixed_state import FixedJointStateUpdater
from curobo_ros_python.planning.mesh_collision import MeshCollision, MeshCollisionChecker
from curobo_ros_python.planning.motion_collision import configure_motion_collisions, install_motion_collision
from curobo_ros_python.planning.ik_collision_filter import install_ik_filter


@dataclass
class TrajResult:
    success: bool
    message: str
    trajectory: Optional[JointState] = None
    solution: Optional[JointState] = None
    planning_time: float = 0.0
    n_waypoints: int = 1
    status: Optional[str] = None


_COLLISION_KIND = {
    "self": "自碰撞",
    "environment": "环境碰撞",
    "payload_self": "携物自碰撞",
    "payload_environment": "携物环境碰撞",
}


def waypoint_collision_label(index: int, count: int) -> str:
    """0 是初始点，最后一个是目标点，中间依次是路点1、路点2。"""
    if index <= 0:
        return "初始点"
    if count <= 1 or index >= count - 1:
        return "目标点"
    return f"路点{index}"


def segment_collision_labels(index: int, count: int) -> Tuple[str, str]:
    """第 index 段的两端。count 是目标路点个数，最后一段的终点是目标点。"""
    start = "初始点" if index <= 0 else f"路点{index}"
    end = "目标点" if count <= 1 or index >= count - 1 else f"路点{index + 1}"
    return start, end


def trajectory_sample_label(index: int, count: int, start: str, end: str) -> str:
    if index <= 0:
        return start
    if index >= count - 1:
        return end
    return f"{start}到{end}的轨迹第{index + 1}/{count}点"


def format_collision_pair(kind: str, first: str, second: str) -> str:
    return f"{_COLLISION_KIND.get(kind, kind)} {first} / {second}"


def trajectory_sample_indices(count: int, limit: int = 8) -> List[int]:
    if count <= 0:
        return []
    if count <= limit:
        return list(range(count))
    return sorted({int(round(i * (count - 1) / (limit - 1))) for i in range(limit)})


def _unique(names: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def _hold_named_joints(traj: JointState, hold: JointState, names: Sequence[str]) -> JointState:
    if not names:
        return traj
    hold_idx = {n: i for i, n in enumerate(hold.joint_names or [])}
    traj_idx = {n: i for i, n in enumerate(traj.joint_names or [])}
    hold_q = hold.position.reshape(-1)
    traj = traj.clone()
    for name in names:
        if name not in hold_idx or name not in traj_idx:
            continue
        i = traj_idx[name]
        q = hold_q[hold_idx[name]]
        traj.position[..., i] = q
        if traj.velocity is not None:
            traj.velocity[..., i] = 0
        if traj.acceleration is not None:
            traj.acceleration[..., i] = 0
        if traj.jerk is not None:
            traj.jerk[..., i] = 0
    return traj


# 建求解器时先锁在这个角度上，用来在 fixed_transforms 里认出被锁的那一行。
# 左右臂局部关节原点经常一样，锁在 0 会和仍在动的那只臂分不清。
_LOCK_SENTINEL = 0.4
# 关节状态的数值抖动小于这个阈值时，保留已经建好的 PRM。
_LOCK_POSE_EPS = 1e-3
_PLAN_GROUPS = ("both_arms", "left_arm", "right_arm")


@dataclass
class _LockedJoint:
    name: str
    row: int
    axis: Tuple[float, float, float]
    origin: torch.Tensor


def _idle_arm_joints(group_name: str) -> List[str]:
    if group_name == "left_arm":
        return list(GROUP_JOINTS.get("right_arm") or [])
    if group_name == "right_arm":
        return list(GROUP_JOINTS.get("left_arm") or [])
    return []


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> torch.Tensor:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return torch.tensor(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=torch.float32,
    )


def _axis_matrix(axis: Sequence[float], angle: float) -> torch.Tensor:
    x, y, z = (float(v) for v in axis)
    norm = math.sqrt(x * x + y * y + z * z)
    if norm < 1e-8:
        return torch.eye(3, dtype=torch.float32)
    x, y, z = x / norm, y / norm, z / norm
    c, s = math.cos(angle), math.sin(angle)
    v = 1.0 - c
    return torch.tensor(
        [
            [c + x * x * v, x * y * v - z * s, x * z * v + y * s],
            [y * x * v + z * s, c + y * y * v, y * z * v - x * s],
            [z * x * v - y * s, z * y * v + x * s, c + z * z * v],
        ],
        dtype=torch.float32,
    )


def _joint_matrix(
    xyz: Sequence[float],
    rpy: Sequence[float],
    axis: Sequence[float],
    angle: float,
) -> torch.Tensor:
    """URDF 关节变换：origin 再绕关节轴转 ``angle``。形状 (3, 4)。"""
    rot = _rpy_matrix(float(rpy[0]), float(rpy[1]), float(rpy[2])) @ _axis_matrix(axis, angle)
    out = torch.zeros(3, 4, dtype=torch.float32)
    out[:, :3] = rot
    out[:, 3] = torch.tensor([float(v) for v in xyz], dtype=torch.float32)
    return out


def _urdf_joint_table(urdf_path: str) -> Dict[str, Tuple[List[float], List[float], List[float]]]:
    root = ET.parse(urdf_path).getroot()
    table = {}
    for joint in root.findall("joint"):
        name = joint.get("name")
        if not name:
            continue
        origin = joint.find("origin")
        xyz = [0.0, 0.0, 0.0]
        rpy = [0.0, 0.0, 0.0]
        if origin is not None:
            if origin.get("xyz"):
                xyz = [float(v) for v in origin.get("xyz").split()]
            if origin.get("rpy"):
                rpy = [float(v) for v in origin.get("rpy").split()]
        axis_el = joint.find("axis")
        axis = [1.0, 0.0, 0.0]
        if axis_el is not None and axis_el.get("xyz"):
            axis = [float(v) for v in axis_el.get("xyz").split()]
        table[name] = (xyz, rpy, axis)
    return table


def _match_locked_rows(
    fixed_transforms: torch.Tensor,
    urdf_joints: Dict[str, Tuple[List[float], List[float], List[float]]],
    names: Sequence[str],
    angle: float,
) -> List[_LockedJoint]:
    """找出每个被锁关节在 ``fixed_transforms`` 里的那一行。"""
    rows = fixed_transforms.detach().float().cpu()
    used = set()
    specs: List[_LockedJoint] = []
    for name in names:
        if name not in urdf_joints:
            raise RuntimeError(f"URDF 里没有要锁的关节 {name}")
        xyz, rpy, axis = urdf_joints[name]
        target = _joint_matrix(xyz, rpy, axis, angle)
        err = (rows - target).abs().reshape(rows.shape[0], -1).amax(dim=1)
        best = int(torch.argmin(err).item())
        second = err.clone()
        second[best] = 1e9
        if float(err[best]) > 1e-3 or best in used or float(second.min()) < 1e-3:
            raise RuntimeError(
                f"无法唯一对应被锁关节 {name} 的运动学行："
                f"best={best} err={float(err[best]):.3g} "
                f"second={float(second.min()):.3g}"
            )
        used.add(best)
        specs.append(
            _LockedJoint(
                name,
                best,
                (float(axis[0]), float(axis[1]), float(axis[2])),
                _joint_matrix(xyz, rpy, axis, 0.0),
            )
        )
    return specs


def _robot_models(planner: MotionPlanner) -> List[Any]:
    """求解器里每一份运动学。CUDA graph 读的是这些张量本身。"""
    found: List[Any] = []
    seen = set()

    def add(robot) -> None:
        config = getattr(robot, "config", None)
        kin = getattr(config, "kinematics_config", None)
        transforms = getattr(kin, "fixed_transforms", None)
        if transforms is None:
            return
        key = id(robot)
        if key in seen:
            return
        seen.add(key)
        found.append(robot)

    def add_rollout(rollout) -> None:
        for name in ("transition_model", "metrics_transition_model"):
            transition = getattr(rollout, name, None)
            if transition is not None:
                add(getattr(transition, "robot_model", None))

    add(planner.kinematics)
    for solver in (planner.ik_solver, planner.trajopt_solver):
        core = getattr(solver, "core", None)
        if core is None:
            continue
        for rollout in core.get_all_rollout_instances(
            include_optimizer_rollouts=True, include_auxiliary_rollout=True
        ):
            add_rollout(rollout)
        extra = getattr(core, "additional_metrics_rollouts", None) or {}
        for rollout in extra.values():
            add_rollout(rollout)
        seed = getattr(solver, "seed_ik_solver", None)
        if seed is not None:
            add(seed.kinematics)
            add(seed._robot_model)
    graph = planner.graph_planner
    if graph is not None:
        add_rollout(getattr(graph, "feasibility_rollout", None))
        add_rollout(getattr(graph, "auxiliary_rollout", None))
        add(getattr(graph, "kinematics", None))
    return found


def _apply_locked_joints(models: Sequence[Any], specs: Sequence[_LockedJoint], values: Dict[str, float]) -> None:
    if not specs:
        return
    matrices: Dict[str, torch.Tensor] = {}
    for spec in specs:
        angle = float(values.get(spec.name, 0.0))
        rot = spec.origin[:, :3] @ _axis_matrix(spec.axis, angle)
        mat = spec.origin.clone()
        mat[:, :3] = rot
        matrices[spec.name] = mat
    for robot in models:
        transforms = robot.config.kinematics_config.fixed_transforms
        for spec in specs:
            transforms[spec.row].copy_(
                matrices[spec.name].to(device=transforms.device, dtype=transforms.dtype)
            )
        lock = robot.lock_jointstate
        if lock is None or lock.position is None:
            continue
        index = {name: i for i, name in enumerate(lock.joint_names or [])}
        for spec in specs:
            if spec.name not in index:
                continue
            lock.position[..., index[spec.name]] = float(values.get(spec.name, 0.0))
            if lock.velocity is not None:
                lock.velocity[..., index[spec.name]] = 0.0


class CuroboPlanner:
    def __init__(
        self,
        robot: str = "lycus.yml",
        scene_model: Optional[str] = None,
        warmup_iterations: int = 3,
        use_cuda_graph: bool = True,
        interpolation_dt: float = 0.02,
        position_tolerance: float = 0.005,
        orientation_tolerance: float = 0.05,
        waypoint_joint_tolerance: float = 0.1,
        collision_cache: Optional[Dict[str, Any]] = None,
        num_trajopt_seeds: int = 2,
        trajopt_num_iters: int = 50,
        finetune_attempts: int = 1,
        logger=None,
        self_filter_geometry: bool = False,
        keep_tool_z_vertical: bool = False,
        trajopt_interpolation_steps: int = 2,
        warmup_groups: Optional[Sequence[str]] = None,
    ):
        self._log = logger
        warmup_groups = tuple(_PLAN_GROUPS if warmup_groups is None else dict.fromkeys(warmup_groups))
        if not warmup_groups or any(group not in _PLAN_GROUPS for group in warmup_groups):
            raise ValueError("trajopt_prewarm_groups 必须包含 left_arm / right_arm / both_arms 中的规划组")
        self.keep_tool_z_vertical = bool(keep_tool_z_vertical)
        seeds = int(num_trajopt_seeds)
        iters = int(trajopt_num_iters)
        finetune = int(finetune_attempts)
        interpolation_steps = int(trajopt_interpolation_steps)
        if not 1 <= interpolation_steps <= 32:
            raise ValueError("trajopt_interpolation_steps 必须在 [1, 32] 内")
        if seeds < 1:
            raise ValueError(f"trajopt_num_seeds 必须 >= 1，收到 {seeds}")
        if finetune < 0:
            raise ValueError(f"trajopt_finetune_attempts 必须 >= 0，收到 {finetune}")
        if iters < 25 or iters % 25 != 0:
            raise ValueError(
                f"trajopt_num_iters={iters} 必须是 25 的倍数且不小于 25"
            )
        self._finetune_attempts = finetune
        joint_tol = float(waypoint_joint_tolerance)
        if joint_tol <= 0.0:
            raise ValueError(f"waypoint_joint_tolerance 必须 > 0，收到 {joint_tol}")
        apply_robot(robot)
        if collision_cache is not None:
            cache = dict(collision_cache)
        elif scene_model is not None and getattr(scene_model, "voxel", None):
            cache = {}
        else:
            cache = {"mesh": 1}
        if cache and "mesh" not in cache and not getattr(scene_model, "voxel", None):
            cache.setdefault("mesh", 1)
        cache = dict(cache or {})
        cache.setdefault("cuboid", 32)
        self._cache = cache
        self._scene_model = scene_model or None
        self._use_cuda_graph = use_cuda_graph
        self._interpolation_dt = float(interpolation_dt)
        self._position_tolerance = position_tolerance
        self._orientation_tolerance = orientation_tolerance
        self._seeds = seeds
        self._iters = iters
        self._interpolation_steps = interpolation_steps
        self._joint_tol = joint_tol
        self._robot = robot
        self._info(f"加载 cuRobo 机器人配置: {robot}")
        self._info(
            f"TrajOpt: seeds={seeds}, finetune_attempts={finetune}, num_iters={iters}, "
            f"interpolation_steps={interpolation_steps}"
        )
        urdf_joints = _urdf_joint_table(self._urdf_path(robot))
        from curobo_ros_python.planning.robot_profile import robot_profile

        self._gripper_state = GripperStateUpdater(
            self._urdf_path(robot), robot_profile(robot).gripper_joints
        )
        self._fixed_state = FixedJointStateUpdater.from_robot(
            robot, list(GROUP_JOINTS['both_arms']) + list(self._gripper_state.joint_names))
        self._planners: Dict[str, MotionPlanner] = {}
        self._locked: Dict[str, List[_LockedJoint]] = {}
        self._models: Dict[str, List[Any]] = {}
        self._motion: Optional[MotionPlanner] = None
        self._cloud_sig = None
        self._cloud_vertices = None
        self._cloud_faces = None
        self._cloud_fcl_sig = None
        self._cloud_fcl_scene = None
        self._cloud_loaded: Dict[str, Any] = {}
        self._locked_pose: Dict[str, Dict[str, float]] = {}
        self.last_timing: Dict[str, float] = {}
        self._placed_objects: Dict[str, Any] = {}
        self._fcl_candidate_ms = 0.0
        self._info(f"OPT 启动预热: groups={list(warmup_groups)} iterations={warmup_iterations}")
        for group_name in _PLAN_GROUPS:
            idle = _idle_arm_joints(group_name)
            self._info(self._group_banner(group_name, idle))
            motion = self._build_motion(group_name, idle)
            if idle:
                specs = _match_locked_rows(
                    motion.kinematics.config.kinematics_config.fixed_transforms,
                    urdf_joints,
                    idle,
                    _LOCK_SENTINEL,
                )
                self._locked[group_name] = specs
                models = _robot_models(motion)
                self._require_same_kinematics(group_name, motion, models)
                _apply_locked_joints(models, specs, {name: 0.0 for name in idle})
                self._models[group_name] = models
            self._planners[group_name] = motion
            if warmup_iterations > 0 and group_name in warmup_groups:
                self._info(f"Warmup {group_name} ({warmup_iterations} 次)...")
                motion.warmup(enable_graph=True, num_warmup_iterations=warmup_iterations)
                if idle:
                    models = _robot_models(motion)
                    _apply_locked_joints(models, self._locked[group_name], {name: 0.0 for name in idle})
                    self._models[group_name] = models
                self._info(f"Warmup {group_name} 完成")
            elif warmup_iterations > 0:
                self._info(f"OPT group={group_name} startup_prewarm=false first_request=cold_start")
            if idle:
                self._locked_pose[group_name] = {name: 0.0 for name in idle}
        self.planner = self._planners["both_arms"]
        self._mesh_checker = MeshCollisionChecker.from_robot(
            robot, retain_meshes=self_filter_geometry, logger=self._log)
        self._mesh_positions = {}
        self._mesh_feedback = {}
        self._mesh_scenes = {}
        self.last_fcl_candidates = {}
        for group, motion in self._planners.items():
            scene_cfg = motion.config.scene_collision_cfg
            scene = None if scene_cfg is None else scene_cfg.scene_model
            self._mesh_scenes[group] = self._mesh_checker.prepare_scene(scene)
            install_ik_filter(motion.ik_solver, self._check_mesh_candidate, self._report_mesh_candidates)
        if self._mesh_checker.skipped_links:
            self._warn(f"FCL: 以下配置连杆无 collision 几何: {self._mesh_checker.skipped_links}")
        self._info("OPT: IK 候选已启用运动相关 FCL 检查，保留运动臂与静止本体碰撞检查")

    def _check_mesh_candidate(self, positions: Mapping[str, float]) -> Optional[MeshCollision]:
        started = time.perf_counter()
        values = dict(self._mesh_feedback)
        values.update(self._mesh_positions)
        values.update(positions)
        try:
            return self._mesh_checker.check(values)
        finally:
            self._fcl_candidate_ms = self._fcl_candidate_ms + (time.perf_counter() - started) * 1000

    def _fcl_environment_label(self) -> str:
        count = len(self._mesh_checker.environment)
        return f"mesh障碍={count}" if count else "空场景（未检查环境障碍）"

    def _report_mesh_candidates(self, checked: int, rejected: Sequence[MeshCollision]) -> None:
        if self._mesh_checker.skip_fcl_checks:
            self._fcl_candidate_ms = 0.0
            self.last_fcl_candidates = {}
            return
        elapsed_ms = self._fcl_candidate_ms
        self._fcl_candidate_ms = 0.0
        self.last_fcl_candidates = {"checked": checked, "rejected": len(rejected), "elapsed_ms": elapsed_ms}
        self._info(f"FCL IK筛选: 检查={checked}, 通过={checked - len(rejected)}, "
                   f"剔除={len(rejected)}, FCL耗时={elapsed_ms:.2f}ms; "
                   f"环境={self._fcl_environment_label()}" +
                   ("; 无成功IK候选可检查" if checked == 0 else "") +
                   (f"; 首个碰撞={rejected[0].kind}: {rejected[0].first} / {rejected[0].second}" if rejected else ""))

    def _check_mesh_endpoint(self, state: JointState, label: str = "目标点") -> Optional[str]:
        if self._mesh_checker.skip_fcl_checks:
            return None
        started = time.perf_counter()
        row = state.position.reshape(-1, state.position.shape[-1])[-1].detach().cpu().tolist()
        previous_ms = self._fcl_candidate_ms
        try:
            hit = self._check_mesh_candidate(dict(zip(state.joint_names, row)))
        except Exception as exc:
            self._warn(f"FCL {label}: 检查异常，拒绝放行: {exc}")
            raise
        finally:
            self._fcl_candidate_ms = previous_ms
        elapsed_ms = (time.perf_counter() - started) * 1000
        detail = "通过" if hit is None else f"碰撞: {format_collision_pair(hit.kind, hit.first, hit.second)}"
        self._info(f"FCL {label}: {'通过' if hit is None else '拒绝'}, {detail}; "
                   f"环境={self._fcl_environment_label()}, 耗时={elapsed_ms:.2f}ms")
        if hit is None:
            return None
        return f"{label}碰撞: {format_collision_pair(hit.kind, hit.first, hit.second)}"

    def _urdf_path(self, robot: str) -> str:
        from curobo.config_io import load_yaml
        from curobo.content import get_robot_configs_path

        data = load_yaml(join_path(str(get_robot_configs_path()), robot))
        relative = data["robot_cfg"]["kinematics"]["urdf_path"]
        return join_path(str(get_assets_path()), relative)

    def _group_banner(self, group_name: str, idle: Sequence[str]) -> str:
        if not idle:
            return "双臂求解器: 左右臂都参与优化"
        active = "左臂" if group_name == "left_arm" else "右臂"
        other = "右臂" if group_name == "left_arm" else "左臂"
        return f"{active}求解器: 锁住{other} {len(idle)} 个关节"

    def _build_motion(self, group_name: str, idle: Sequence[str]) -> MotionPlanner:
        robot, _locks = robot_cfg_with_locked_joints(
            self._robot,
            {name: _LOCK_SENTINEL for name in idle},
            idle,
        )
        from curobo_ros_python.planning.payload import configure_payload_links
        robot = configure_payload_links(robot)
        moving = configure_motion_collisions(robot, GROUP_JOINTS[group_name])
        transition = load_yaml(join_path(
            get_task_configs_path(), "trajopt/transition_bspline_trajopt.yml"))
        transition["transition_model_cfg"]["interpolation_steps"] = self._interpolation_steps
        cfg = MotionPlannerCfg.create(
            robot=robot,
            scene_model=self._scene_model,
            collision_cache=self._cache,
            use_cuda_graph=self._use_cuda_graph,
            interpolation_dt=self._interpolation_dt,
            position_tolerance=self._position_tolerance,
            orientation_tolerance=self._orientation_tolerance,
            num_trajopt_seeds=self._seeds,
            trajopt_num_iters=self._iters,
            trajopt_transition_model=transition,
        )
        motion = MotionPlanner(cfg)
        install_motion_collision(motion, moving)
        install_vertical_cost(motion)
        motion.waypoint_joint_tolerance = self._joint_tol
        return motion

    @property
    def gripper_joint_names(self) -> Tuple[str, ...]:
        return self._gripper_state.joint_names

    def update_gripper_state(self, joint_map: Dict[str, float]) -> None:
        """同一反馈快照更新夹爪、躯干/头和 FCL；保留优化自由度和图地址。"""
        models = [model for motion in self._planners.values() for model in _robot_models(motion)]
        gripper_changed = self._gripper_state.update(joint_map, models)
        fixed_changed = self._fixed_state.update(joint_map, models)
        if gripper_changed or fixed_changed:
            for motion in self._planners.values():
                if motion.graph_planner is not None:
                    motion.graph_planner.reset_buffer()
        self._mesh_feedback = {name: float(value) for name, value in joint_map.items()}
        self._mesh_feedback.update(self._fixed_state.positions)

    def _require_same_kinematics(self, group_name: str, motion: MotionPlanner, models: Sequence[Any]) -> None:
        ref = motion.kinematics.config.kinematics_config.fixed_transforms.detach()
        if not models:
            raise RuntimeError(f"{group_name} 没有可更新的运动学")
        for robot in models:
            transforms = robot.config.kinematics_config.fixed_transforms
            if transforms.shape != ref.shape or not torch.allclose(transforms, ref, atol=1e-4):
                raise RuntimeError(f"{group_name} 的运动学副本不一致，无法一起更新被锁手臂")

    def _sync_locked_arm(self, group_name: str, current: JointState) -> MotionPlanner:
        motion = self._planners[group_name]
        self._motion = motion
        names = list(current.joint_names or [])
        cpu = current.position.reshape(-1).detach().cpu()
        self._mesh_positions = dict(zip(names, cpu.tolist()))
        self._mesh_checker.environment = self._mesh_scenes[group_name]
        self._mesh_checker.set_motion_joints(GROUP_JOINTS[group_name])
        self.last_fcl_candidates = {}
        specs = self._locked.get(group_name) or []
        if not specs:
            return motion
        index = {name: i for i, name in enumerate(names)}
        missing = [spec.name for spec in specs if spec.name not in index]
        if missing:
            raise ValueError(f"当前关节状态缺少被锁关节 {missing[0]}")
        values = {spec.name: float(cpu[index[spec.name]]) for spec in specs}
        previous = self._locked_pose.get(group_name) or {}
        moved = any(
            abs(values[name] - float(previous.get(name, 0.0))) > _LOCK_POSE_EPS
            for name in values
        )
        if not moved:
            return motion
        _apply_locked_joints(self._models[group_name], specs, values)
        if motion.graph_planner is not None:
            motion.graph_planner.reset_buffer()
            self._info(f"{group_name} 锁定手臂姿态已变化，PRM 缓存已清空")
        self._locked_pose[group_name] = values
        return motion

    def destroy(self):
        for motion in self._planners.values():
            motion.destroy()

    def clear_pointcloud_world(self, group_name: Optional[str] = None) -> None:
        """Clear a valid cloud filtered to zero; don't reuse old body obstacles."""
        from curobo._src.geom.types import SceneCfg
        if self._scene_model is not None:
            raise ValueError('本体过滤后为空且配置了静态场景，拒绝清空混合环境')
        targets = [group_name] if group_name in self._planners else list(self._planners)
        for group in targets:
            scene = SceneCfg(cuboid=list(self._placed_objects.values()))
            self._planners[group].update_world(scene)
            self._mesh_scenes[group] = self._mesh_checker.prepare_scene(scene)
            self._cloud_loaded.pop(group, None)
        self._info('本体过滤后环境点为零，GPU/FCL旧点云场景已清空')

    def update_world_from_points(
        self,
        points,
        pitch: float = 0.02,
        name: str = "environment",
        group_name: Optional[str] = None,
    ) -> int:
        from curobo._src.geom.types import Mesh, SceneCfg
        import hashlib
        import numpy as np

        pts = np.ascontiguousarray(np.asarray(points, dtype=np.float32).reshape(-1, 3))
        n = int(pts.shape[0])
        pitch = float(pitch)
        digest = hashlib.blake2b(pts.tobytes(), digest_size=8).digest()
        sig = (pitch, n, digest)
        if group_name in self._planners:
            targets = [group_name]
        else:
            targets = list(self._planners)
        pending = [group for group in targets if self._cloud_loaded.get(group) != sig]
        if not pending:
            return n
        if self._cloud_sig != sig or self._cloud_vertices is None:
            built = Mesh.from_pointcloud(pts, pitch=pitch, name=name or "environment")
            self._cloud_vertices = np.ascontiguousarray(built.vertices, dtype=np.float32)
            self._cloud_faces = np.ascontiguousarray(built.faces, dtype=np.int32)
            self._cloud_sig = sig
            self._cloud_loaded = {}
        mesh_name = name or "environment"
        if self._cloud_fcl_sig != (sig, mesh_name):
            fcl_mesh = Mesh(name=mesh_name, pose=[0., 0., 0., 1., 0., 0., 0.],
                            vertices=self._cloud_vertices, faces=self._cloud_faces)
            fcl_scene = self._mesh_checker.prepare_scene(SceneCfg(mesh=[fcl_mesh], cuboid=list(self._placed_objects.values())))
            self._cloud_fcl_scene = fcl_scene
            self._cloud_fcl_sig = (sig, mesh_name)
        fcl_scene = self._cloud_fcl_scene
        for group in pending:
            motion = self._planners[group]
            mesh = Mesh(
                name=mesh_name,
                pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                vertices=np.array(self._cloud_vertices, copy=True),
                faces=np.array(self._cloud_faces, copy=True),
            )
            checker = motion.scene_collision_checker
            old = pop_cached_mesh(checker, mesh.name)
            try:
                motion.update_world(SceneCfg(mesh=[mesh], cuboid=list(self._placed_objects.values())))
            except Exception:
                restore_cached_mesh(checker, mesh.name, old)
                raise
            finally:
                del old
            self._cloud_loaded[group] = sig
            self._mesh_scenes[group] = fcl_scene
        self._info(
            f"已用点云更新碰撞场景: {n} 点, pitch={pitch:.3f} m, "
            f"求解器={','.join(pending)}"
        )
        return n

    def update_world_from_esdf(
        self,
        points,
        pitch: float,
        dims: Sequence[float],
        center: Sequence[float],
        name: str = "environment",
    ) -> Tuple[int, int]:
        """把点云写成固定格子的距离场。返回 (输入点数, 落在盒子里的点数)。"""
        import numpy as np
        import torch
        from curobo.scene import Scene, VoxelGrid

        from curobo_ros_python.planning.pointcloud import points_to_esdf, snap_voxel_dims

        pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        dims_s = snap_voxel_dims(dims, pitch)
        esdf, n_inside = points_to_esdf(pts, center, dims_s, pitch)
        from curobo.scene import Mesh
        low = np.asarray(center) - np.asarray(dims_s) / 2
        high = low + np.asarray(dims_s)
        inside = pts[((pts >= low) & (pts < high)).all(1)]
        fcl_scene = self._mesh_checker.prepare_scene(Scene(mesh=[
            Mesh.from_pointcloud(inside, pitch=pitch, name=name or "environment")])) if len(inside) else {}
        cx, cy, cz = (float(v) for v in center)
        feature = torch.from_numpy(np.ascontiguousarray(esdf))
        for motion in self._planners.values():
            grid = VoxelGrid(
                name=name or "environment",
                pose=[cx, cy, cz, 1.0, 0.0, 0.0, 0.0],
                dims=[float(v) for v in dims_s],
                voxel_size=float(pitch),
                feature_tensor=feature.clone(),
            )
            motion.update_world(Scene(voxel=[grid]))
        self._mesh_scenes = {group: fcl_scene for group in self._planners}
        self._info(
            f"已用 ESDF 更新碰撞场景: 框内 {n_inside}/{int(pts.shape[0])} 点, "
            f"pitch={float(pitch):.3f} m, "
            f"dims=({dims_s[0]:.2f}, {dims_s[1]:.2f}, {dims_s[2]:.2f}) m, "
            f"center=({cx:.2f}, {cy:.2f}, {cz:.2f})"
        )
        return int(pts.shape[0]), n_inside

    def update_world_from_voxel(self, voxel_grid) -> None:
        raise ValueError("FCL 目标筛选需要原始点云/mesh，不能仅用 ESDF VoxelGrid 更新环境")

    @property
    def joint_names(self) -> List[str]:
        return list(self.planner.joint_names)

    @property
    def tool_frames(self) -> List[str]:
        return list(self.planner.tool_frames)

    @property
    def device(self):
        return self.planner.device_cfg.device

    @property
    def interpolation_dt(self) -> float:
        return float(self.planner.trajopt_solver.config.interpolation_dt)

    @property
    def default_joint_position(self):
        return self.planner.default_joint_state.position.reshape(-1)

    @property
    def device_cfg(self):
        return self.planner.device_cfg

    def _info(self, msg: str):
        if self._log is not None:
            self._log.info(msg)
        else:
            print(msg, flush=True)

    def _warn(self, msg: str):
        if self._log is not None:
            self._log.warn(msg)
        else:
            print("WARN:", msg, flush=True)

    def _collision_hit(self, state: Optional[JointState]) -> Optional[MeshCollision]:
        if state is None or self._mesh_checker.skip_fcl_checks:
            return None
        position = state.position.detach().reshape(-1, state.position.shape[-1])[-1].cpu().tolist()
        names = list(state.joint_names or [])
        if len(names) != len(position):
            return None
        return self._check_mesh_candidate(dict(zip(names, position)))

    def _ik_collision_state(self, motion, ik_result) -> Optional[JointState]:
        if ik_result is None or getattr(ik_result, "solution", None) is None:
            return None
        solution = ik_result.solution.detach().reshape(-1, ik_result.solution.shape[-1])
        if solution.numel() == 0:
            return None
        index = 0
        error = getattr(ik_result, "position_error", None)
        if error is not None and error.numel() > 0:
            flat = error.detach().reshape(-1)
            count = min(int(solution.shape[0]), int(flat.shape[0]))
            if count > 0:
                index = int(torch.argmin(flat[:count]).item())
        names = list(getattr(getattr(motion, "ik_solver", None), "joint_names", []) or [])
        if len(names) != int(solution.shape[-1]):
            return None
        return JointState.from_position(solution[index].reshape(1, -1), joint_names=names)

    def _trajectory_rows(self, result):
        state = getattr(result, "js_solution", None)
        if state is None or getattr(state, "position", None) is None:
            state = getattr(result, "interpolated_trajectory", None)
        if state is None or getattr(state, "position", None) is None:
            return None, None
        position = state.position.detach()
        names = list(state.joint_names or [])
        seed = 0
        rank = getattr(result, "seed_rank", None)
        if rank is not None:
            try:
                seed = max(int(rank.detach().reshape(-1)[0].item()), 0)
            except Exception:
                seed = 0
        if position.ndim == 4:
            seed = seed if seed < position.shape[1] else 0
            position = position[0, seed]
        elif position.ndim == 3:
            if position.shape[0] == 1:
                position = position[0]
            else:
                seed = seed if seed < position.shape[0] else 0
                position = position[seed]
        if position.ndim != 2 or position.shape[0] < 1:
            return None, None
        if len(names) != int(position.shape[-1]):
            return None, None
        return position, names

    def _with_collision_detail(
        self,
        message: str,
        result,
        current: Optional[JointState],
        goals: Optional[Sequence[JointState]] = None,
        failed_index: Optional[int] = None,
        n_goals: int = 1,
    ) -> str:
        """失败时补充碰撞发生在初始点、路点还是目标点，以及相撞的两侧。"""
        if self._mesh_checker.skip_fcl_checks:
            return message
        previous_ms = self._fcl_candidate_ms
        try:
            notes: List[str] = []
            checked = set()
            found_interior = False
            start, end = segment_collision_labels(
                0 if failed_index is None else int(failed_index), max(int(n_goals), 1)
            )

            def record(label: str, state: Optional[JointState], interior: bool = False) -> None:
                nonlocal found_interior
                if state is None or (interior and found_interior) or (not interior and label in checked):
                    return
                if not interior:
                    checked.add(label)
                hit = self._collision_hit(state)
                if hit is None:
                    return
                notes.append(f"{label}碰撞: {format_collision_pair(hit.kind, hit.first, hit.second)}")
                if interior:
                    found_interior = True

            record(start, current)
            if goals:
                count = len(goals) + 1
                for index, goal in enumerate(goals):
                    record(waypoint_collision_label(index + 1, count), goal)
            rows, names = self._trajectory_rows(result)
            if rows is None and result is None:
                motion = self._motion or self.planner
                record(f"{end}的IK候选", self._ik_collision_state(motion, getattr(motion, "last_ik_result", None)))
            if rows is not None:
                total = int(rows.shape[0])
                for index in trajectory_sample_indices(total):
                    label = trajectory_sample_label(index, total, start, end)
                    if index == total - 1:
                        label = f"{end}的失败轨迹末端候选"
                    interior = 0 < index < total - 1
                    row = rows[index].detach().reshape(1, -1)
                    record(label, JointState.from_position(row, joint_names=names), interior)
                    if len(notes) >= 4:
                        break
            if notes:
                message += "；" + "；".join(notes[:4])
        except Exception as exc:
            message += f"；碰撞明细检查失败: {exc}"
        finally:
            # Failure diagnostics must not enter the next IK-filter timing.
            self._fcl_candidate_ms = previous_ms
        return message

    def _fail(self, message: str) -> TrajResult:
        candidates = self.last_fcl_candidates
        if candidates.get("checked", 0) and candidates.get("checked") == candidates.get("rejected"):
            message += "；FCL 已剔除全部 IK 候选（目标状态自碰撞或环境碰撞）"
        return TrajResult(False, message)

    def _prepare(self, group_name: str, waypoints=None) -> Optional[TrajResult]:
        self.last_fcl_candidates = {}
        self._fcl_candidate_ms = 0.0
        err = unknown_group(group_name)
        if err:
            return self._fail(err)
        if waypoints is not None and not waypoints:
            return self._fail("waypoints 为空")
        return None

    def _active(self, current: JointState) -> JointState:
        motion = self._motion or self.planner
        return motion.kinematics.get_active_js(current.clone())

    def fk_tool_poses(self, joint_state: JointState) -> Dict[str, Pose]:
        motion = self._motion or self.planner
        active = motion.kinematics.get_active_js(joint_state.clone())
        kin = motion.compute_kinematics(active)
        if kin.tool_poses is None:
            return {}
        return {
            name: squeeze_pose(kin.tool_poses.get_link_pose(name))
            for name in self.tool_frames
        }

    def fk_tool_path(self, joint_state: JointState, tool: str):
        """Batch FK as CPU xyz/wxyz rows, using one device-to-host transfer."""
        motion = self._motion or self.planner
        active = motion.kinematics.get_active_js(joint_state)
        path = JointState.from_position(
            active.position.reshape(1, -1, len(active.joint_names)),
            joint_names=active.joint_names,
        )
        pose = motion.compute_kinematics(path).tool_poses.get_link_pose(tool)
        return torch.cat((pose.position.reshape(-1, 3), pose.quaternion.reshape(-1, 4)), dim=-1).detach().cpu().numpy()

    def _goal_for_ee_poses(
        self,
        current_state: JointState,
        pose_by_ee: Dict[str, Pose],
        current_fk: Optional[Dict[str, Pose]] = None,
    ) -> Tuple[GoalToolPose, List[str]]:
        if not pose_by_ee:
            raise ValueError("没有末端目标")
        frames = list((self._motion or self.planner).tool_frames)
        missing = [name for name in pose_by_ee if name not in frames]
        if missing:
            raise ValueError(
                f"末端 {missing} 不在 cuRobo tool_frames 中: {frames}"
            )
        needs_fk = any(name not in pose_by_ee for name in frames)
        fk = (current_fk if current_fk is not None else self.fk_tool_poses(current_state)) if needs_fk else {}
        pose_dict = {}
        idle_frames = []
        for name in frames:
            if name in pose_by_ee:
                pose_dict[name] = squeeze_pose(pose_by_ee[name])
            elif name in fk:
                pose_dict[name] = fk[name].clone()
                idle_frames.append(name)
            else:
                first = next(iter(pose_by_ee.values()))
                pose_dict[name] = squeeze_pose(first)
        goal = GoalToolPose.from_poses(
            pose_dict, ordered_tool_frames=frames, num_goalset=1
        )
        idle_joints: List[str] = []
        for frame in idle_frames:
            idle_joints.extend(joints_for_ee(frame, self.joint_names))
        return goal, _unique(idle_joints)

    def _make_plan_output(
        self,
        traj: JointState,
        hold_state: JointState,
        hold_joints: Sequence[str],
        n_wp: int,
        planning_time: float,
        status: Optional[str] = None,
    ) -> TrajResult:
        if traj is None or traj.position is None:
            return self._fail("规划成功但插值轨迹为空")
        if not traj.joint_names:
            traj.joint_names = list(self.joint_names)
        if hold_state is not None:
            traj = self._expand_to_reference(traj, hold_state)
        if hold_state is not None and hold_joints:
            traj = _hold_named_joints(traj, hold_state, hold_joints)
        collision_error = self._check_mesh_endpoint(traj)
        if collision_error:
            return self._fail(collision_error)
        extra = f", 路点数={n_wp}" if n_wp > 1 else ""
        n_pts = int(traj.position.reshape(-1, traj.position.shape[-1]).shape[0])
        msg = f"规划成功, 轨迹点数={n_pts}{extra}"
        if status:
            msg += f"（{status}）"
        return TrajResult(
            True,
            msg,
            trajectory=traj,
            planning_time=planning_time,
            n_waypoints=n_wp,
            status=status,
        )

    def _expand_to_reference(self, traj: JointState, hold_state: JointState) -> JointState:
        """单臂求解器只输出 7 个关节。补成双臂关节顺序，缺的关节保持当前角度。"""
        ref = list(self.joint_names)
        names = list(traj.joint_names or [])
        if names == ref:
            return traj
        pos = traj.position
        while pos.ndim > 2:
            pos = pos[0]
        if pos.ndim == 1:
            pos = pos.unsqueeze(0)
        hold_q = hold_state.position.reshape(-1)
        hold_names = list(hold_state.joint_names or [])
        hold_map = {name: hold_q[i] for i, name in enumerate(hold_names)}
        out = pos.new_zeros(pos.shape[0], len(ref))
        src = {name: i for i, name in enumerate(names)}
        for i, name in enumerate(ref):
            if name in src:
                out[:, i] = pos[:, src[name]]
            elif name in hold_map:
                out[:, i] = hold_map[name]
        state = JointState.from_position(out, joint_names=ref)
        for field in ("velocity", "acceleration", "jerk"):
            values = getattr(traj, field, None)
            if values is None:
                setattr(state, field, None)
                continue
            while values.ndim > 2:
                values = values[0]
            if values.ndim == 1:
                values = values.unsqueeze(0)
            expanded = values.new_zeros(out.shape)
            for i, name in enumerate(ref):
                if name in src and src[name] < values.shape[-1]:
                    expanded[:, i] = values[:, src[name]]
            setattr(state, field, expanded)
        return state

    def _from_single_result(
        self,
        result,
        hold_state: JointState,
        hold_joints: Sequence[str],
    ) -> TrajResult:
        if result is None or not result.success.any():
            motion = self._motion or self.planner
            message = f"规划失败: {format_plan_failure(result, motion.last_ik_result)}"
            return self._fail(self._with_collision_detail(message, result, hold_state))
        traj = result.get_interpolated_plan()
        if traj is None:
            return self._fail("规划成功但插值轨迹为空")
        return self._make_plan_output(
            traj,
            hold_state,
            hold_joints,
            1,
            float(
                getattr(result, "total_time", 0.0)
                or getattr(result, "planning_time", 0.0)
                or 0.0
            ),
        )

    def _from_waypoint_result(
        self,
        wp_result,
        hold_state: JointState,
        hold_joints: Sequence[str],
        n_wp: int,
    ) -> TrajResult:
        if wp_result.success is None or not wp_result.success.any():
            failed_index = getattr(wp_result, "failed_index", None)
            message = f"规划失败: {wp_result.status}"
            if failed_index is not None:
                start, end = segment_collision_labels(int(failed_index), n_wp)
                message = f"规划失败: {start}到{end}；{wp_result.status}"
            segment_start = hold_state
            if failed_index not in (None, 0):
                segment_start = None
                segments = getattr(wp_result, "segment_trajectories", None)
                if segments:
                    previous = segments[-1]
                    position = previous.position.reshape(-1, previous.position.shape[-1])[-1:]
                    segment_start = JointState.from_position(position, joint_names=previous.joint_names)
            return self._fail(self._with_collision_detail(
                message,
                getattr(wp_result, "failed_result", None),
                segment_start,
                failed_index=failed_index,
                n_goals=n_wp,
            ))
        if wp_result.interpolated_trajectory is None:
            return self._fail("规划成功但轨迹为空")
        return self._make_plan_output(
            wp_result.interpolated_trajectory,
            hold_state,
            hold_joints,
            n_wp,
            float(getattr(wp_result, "planning_time", 0.0) or 0.0),
            wp_result.status,
        )

    def plan_pose(
        self,
        group_name: str,
        current: JointState,
        waypoints: Sequence[Dict[str, Pose]],
        max_attempts: int = 5,
    ) -> TrajResult:
        self.last_timing = {}
        bad = self._prepare(group_name, waypoints)
        if bad:
            return bad
        if group_name == "both_arms":
            required = set(both_arm_ee_links())
            if any(set(poses) != required for poses in waypoints):
                return self._fail("both_arms 的每个位姿路点必须同时提供左右臂目标；单臂目标请使用 left_arm 或 right_arm")
        motion = self._sync_locked_arm(group_name, current)
        active = self._active(current)
        goals = []
        idle_joints: List[str] = []
        current_fk = None
        for pose_by_ee in waypoints:
            pose_by_ee = dict(pose_by_ee)
            if len(pose_by_ee) == 1:
                ee, err = resolve_ee_link(group_name, next(iter(pose_by_ee)))
                if err:
                    return self._fail(err)
                if ee not in pose_by_ee:
                    pose_by_ee = {ee: pose_by_ee[next(iter(pose_by_ee))]}
            if current_fk is None and any(name not in pose_by_ee for name in motion.tool_frames):
                current_fk = self.fk_tool_poses(current)
            goal, idle_joints = self._goal_for_ee_poses(current, pose_by_ee, current_fk=current_fk)
            goals.append(goal)
        with vertical_tool_scope(motion,self.keep_tool_z_vertical,tuple(pose_by_ee),goal_poses=goals[-1]):
            if len(goals) == 1:
                result = motion.plan_pose(
                    goals[0],
                    active,
                    max_attempts=max_attempts,
                    finetune_attempts=self._finetune_attempts,
                )
            else:
                result = motion.plan_cartesian_waypoints(
                    goals,
                    active,
                    max_attempts=max_attempts,
                    finetune_attempts=self._finetune_attempts,
                )
        self.last_timing = dict(getattr(motion, "last_timing", {}) or {})
        if len(goals) == 1:
            return self._from_single_result(result, current, idle_joints)
        return self._from_waypoint_result(result, current, idle_joints, len(goals))

    def plan_joint(
        self,
        group_name: str,
        current: JointState,
        goals: Sequence[JointState],
        max_attempts: int = 5,
    ) -> TrajResult:
        self.last_timing = {}
        bad = self._prepare(group_name, goals)
        if bad:
            return bad
        group_joints = set(GROUP_JOINTS[group_name])
        motion = self._sync_locked_arm(group_name, current)
        active = self._active(current)
        # 单臂求解器的活动关节不含被锁的另一臂。目标仍按双臂关节名组装，FK 要求名字完全一致。
        goals = [self._active(goal) for goal in goals]
        hold_joints = [n for n in self.joint_names if n not in group_joints]
        collision_hits = []
        point_count = len(goals) + 1
        for index, goal in enumerate(goals):
            collision_error = self._check_mesh_endpoint(
                goal, label=waypoint_collision_label(index + 1, point_count)
            )
            if collision_error:
                collision_hits.append(collision_error)
        if collision_hits:
            return self._fail("规划失败: " + "；".join(collision_hits))
        frames=both_arm_ee_links() if group_name=='both_arms' else (resolve_ee_link(group_name,'')[0],)
        with vertical_tool_scope(motion,self.keep_tool_z_vertical,frames,goal_state=goals[-1]):
            if len(goals) == 1:
                result = motion.plan_cspace(
                    goals[0],
                    active,
                    max_attempts=max_attempts,
                    finetune_attempts=self._finetune_attempts,
                )
            else:
                result = motion.plan_joint_waypoints(
                    list(goals),
                    active,
                    max_attempts=max_attempts,
                    finetune_attempts=self._finetune_attempts,
                )
        self.last_timing = dict(getattr(motion, "last_timing", {}) or {})
        if len(goals) == 1:
            return self._from_single_result(result, current, hold_joints)
        return self._from_waypoint_result(
            result, current, hold_joints, len(goals)
        )

    def compute_ik(
        self,
        group_name: str,
        current: JointState,
        pose: Pose,
        ee_link: str = "",
    ) -> TrajResult:
        bad = self._prepare(group_name)
        if bad:
            return bad
        ee, err = resolve_ee_link(group_name, ee_link)
        if err:
            return self._fail(err)
        motion = self._sync_locked_arm(group_name, current)
        active = self._active(current)
        goal, _idle = self._goal_for_ee_poses(current, {ee: pose})
        ik = motion.ik_solver.solve_pose(
            goal, current_state=active, return_seeds=1
        )
        if ik is None or not ik.success.any():
            return self._fail(format_ik_failure(ik))
        return TrajResult(True, "IK 求解成功", solution=ik.js_solution)


CuroboK100Planner = CuroboPlanner
