"""cuRobo 规划封装：位姿规划、关节规划、IK。机器人由 ``robot`` YAML 决定。

只吃 / 吐 cuRobo 类型。ROS 消息和 RViz 绿线不进这里。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg, format_ik_failure, format_plan_failure
from curobo.types import GoalToolPose, JointState, Pose

from curobo_ros_python.planning.algorithms.common import squeeze_pose
from curobo_ros_python.planning.groups import (
    GROUP_JOINTS,
    apply_robot,
    joints_for_ee,
    resolve_ee_link,
    unknown_group,
)


@dataclass
class TrajResult:
    success: bool
    message: str
    trajectory: Optional[JointState] = None
    solution: Optional[JointState] = None
    planning_time: float = 0.0
    n_waypoints: int = 1
    status: Optional[str] = None


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
        collision_cache: Optional[Dict[str, int]] = None,
        logger=None,
    ):
        self._log = logger
        apply_robot(robot)
        if collision_cache is not None:
            cache = dict(collision_cache)
        elif scene_model is not None and getattr(scene_model, "voxel", None):
            cache = {}
        else:
            cache = {"mesh": 1}
        if cache and "mesh" not in cache and not getattr(scene_model, "voxel", None):
            cache.setdefault("mesh", 1)
        self._info(f"加载 cuRobo 机器人配置: {robot}")
        cfg = MotionPlannerCfg.create(
            robot=robot,
            scene_model=scene_model or None,
            collision_cache=cache,
            use_cuda_graph=use_cuda_graph,
            interpolation_dt=float(interpolation_dt),
            position_tolerance=position_tolerance,
            orientation_tolerance=orientation_tolerance,
        )
        self.planner = MotionPlanner(cfg)
        if warmup_iterations > 0:
            self._info(f"Warmup 规划器 ({warmup_iterations} 次)...")
            self.planner.warmup(enable_graph=True, num_warmup_iterations=warmup_iterations)
            self._info("Warmup 完成")

    def destroy(self):
        self.planner.destroy()

    def update_world_from_points(
        self,
        points,
        pitch: float = 0.02,
        name: str = "cam",
    ) -> int:
        from curobo._src.geom.types import Mesh, SceneCfg
        import numpy as np

        pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        mesh = Mesh.from_pointcloud(pts, pitch=float(pitch), name=name or "cam")
        self.planner.update_world(SceneCfg(mesh=[mesh]))
        n = int(pts.shape[0])
        self._info(f"已用点云更新碰撞场景: {n} 点, pitch={pitch:.3f} m")
        return n

    def update_world_from_voxel(self, voxel_grid) -> None:
        from curobo.scene import Scene

        self.planner.update_world(Scene(voxel=[voxel_grid]))
        self._info("已用 ESDF VoxelGrid 更新碰撞场景")

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

    def _fail(self, message: str) -> TrajResult:
        return TrajResult(False, message)

    def _prepare(self, group_name: str, waypoints=None) -> Optional[TrajResult]:
        err = unknown_group(group_name)
        if err:
            return self._fail(err)
        if waypoints is not None and not waypoints:
            return self._fail("waypoints 为空")
        return None

    def _active(self, current: JointState) -> JointState:
        return self.planner.kinematics.get_active_js(current.clone())

    def fk_tool_poses(self, joint_state: JointState) -> Dict[str, Pose]:
        active = self.planner.kinematics.get_active_js(joint_state.clone())
        kin = self.planner.compute_kinematics(active)
        if kin.tool_poses is None:
            return {}
        return {
            name: squeeze_pose(kin.tool_poses.get_link_pose(name))
            for name in self.tool_frames
        }

    def _goal_for_ee_poses(
        self,
        current_state: JointState,
        pose_by_ee: Dict[str, Pose],
    ) -> Tuple[GoalToolPose, List[str]]:
        if not pose_by_ee:
            raise ValueError("没有末端目标")
        missing = [name for name in pose_by_ee if name not in self.tool_frames]
        if missing:
            raise ValueError(
                f"末端 {missing} 不在 cuRobo tool_frames 中: {self.tool_frames}"
            )
        fk = self.fk_tool_poses(current_state)
        pose_dict = {}
        idle_frames = []
        for name in self.tool_frames:
            if name in pose_by_ee:
                pose_dict[name] = squeeze_pose(pose_by_ee[name])
            elif name in fk:
                pose_dict[name] = fk[name].clone()
                idle_frames.append(name)
            else:
                first = next(iter(pose_by_ee.values()))
                pose_dict[name] = squeeze_pose(first)
        goal = GoalToolPose.from_poses(
            pose_dict, ordered_tool_frames=self.tool_frames, num_goalset=1
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
        if hold_state is not None and hold_joints:
            traj = _hold_named_joints(traj, hold_state, hold_joints)
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

    def _from_single_result(
        self,
        result,
        hold_state: JointState,
        hold_joints: Sequence[str],
    ) -> TrajResult:
        if result is None or not result.success.any():
            return self._fail(
                f"规划失败: {format_plan_failure(result, self.planner.last_ik_result)}"
            )
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
            return self._fail(f"规划失败: {wp_result.status}")
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
        bad = self._prepare(group_name, waypoints)
        if bad:
            return bad
        active = self._active(current)
        goals = []
        idle_joints: List[str] = []
        for pose_by_ee in waypoints:
            pose_by_ee = dict(pose_by_ee)
            if len(pose_by_ee) == 1:
                ee, err = resolve_ee_link(group_name, next(iter(pose_by_ee)))
                if err:
                    return self._fail(err)
                if ee not in pose_by_ee:
                    pose_by_ee = {ee: pose_by_ee[next(iter(pose_by_ee))]}
            goal, idle_joints = self._goal_for_ee_poses(current, pose_by_ee)
            goals.append(goal)
        if len(goals) == 1:
            return self._from_single_result(
                self.planner.plan_pose(goals[0], active, max_attempts=max_attempts),
                active,
                idle_joints,
            )
        return self._from_waypoint_result(
            self.planner.plan_cartesian_waypoints(goals, active, max_attempts=max_attempts),
            active,
            idle_joints,
            len(goals),
        )

    def plan_joint(
        self,
        group_name: str,
        current: JointState,
        goals: Sequence[JointState],
        max_attempts: int = 5,
    ) -> TrajResult:
        bad = self._prepare(group_name, goals)
        if bad:
            return bad
        group_joints = set(GROUP_JOINTS[group_name])
        active = self._active(current)
        hold_joints = [n for n in self.joint_names if n not in group_joints]
        if len(goals) == 1:
            return self._from_single_result(
                self.planner.plan_cspace(goals[0], active, max_attempts=max_attempts),
                active,
                hold_joints,
            )
        return self._from_waypoint_result(
            self.planner.plan_joint_waypoints(list(goals), active, max_attempts=max_attempts),
            active,
            hold_joints,
            len(goals),
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
        active = self._active(current)
        goal, _idle = self._goal_for_ee_poses(current, {ee: pose})
        ik = self.planner.ik_solver.solve_pose(
            goal, current_state=active, return_seeds=1
        )
        if ik is None or not ik.success.any():
            return self._fail(format_ik_failure(ik))
        return TrajResult(True, "IK 求解成功", solution=ik.js_solution)


CuroboK100Planner = CuroboPlanner
