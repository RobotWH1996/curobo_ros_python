"""双臂 MPC：两个笛卡尔目标，缺失的一侧钉在当前末端。

只吃 / 吐 cuRobo 类型。ROS 消息和 RViz 不进这里。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from curobo._src.geom.collision.collision_scene import SceneCollisionCfg
from curobo.model_predictive_control import ModelPredictiveControl, ModelPredictiveControlCfg
from curobo.scene import Mesh, Scene
from curobo.types import DeviceCfg, GoalToolPose, JointState, Pose

from curobo_ros_python.planning.algorithms.common import (
    pose_error,
    pose_xyz_wxyz,
    robot_cfg_with_locked_joints,
    squeeze_pose,
)
from curobo_ros_python.planning.groups import both_arm_ee_links


def _rotvec_wxyz(q_from: torch.Tensor, q_to: torch.Tensor) -> torch.Tensor:
    """q_from 转到 q_to 的旋转向量，四元数 wxyz。"""
    if torch.dot(q_from, q_to) < 0:
        q_to = -q_to
    w1, x1, y1, z1 = q_to.unbind(-1)
    w2, x2, y2, z2 = q_from[0], -q_from[1], -q_from[2], -q_from[3]
    err = torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )
    )
    xyz = err[1:]
    norm = torch.linalg.norm(xyz)
    if float(norm) < 1e-8:
        return 2.0 * xyz
    angle = 2.0 * torch.atan2(norm, err[0].clamp(-1.0, 1.0))
    return xyz * (angle / norm)


def _multiple_of(value: int, step: int) -> int:
    """LBFGS 要求 num_iters 是 inner_iters 的整数倍，否则求解直接抛异常。"""
    step = max(1, int(step))
    count = max(step, int(value))
    return max(step, (count // step) * step)


@dataclass
class MpcStep:
    """一拍求解结果，全是 cuRobo / Python 类型。"""

    result: object = None
    used: Dict[str, Pose] = field(default_factory=dict)
    current_ee: Dict[str, Pose] = field(default_factory=dict)
    success: bool = False
    solve_time: float = 0.0
    message: str = ""


class DualArmMpc:
    def __init__(
        self,
        robot: str = "lycus.yml",
        device: str = "cuda:0",
        use_cuda_graph: bool = True,
        optimization_dt: float = 0.05,
        run_ik: bool = False,
        posture_weight: float = 5.0,
        warm_start_iters: int = 50,
        cold_start_iters: int = 100,
        lock_joint_values: Optional[Dict[str, float]] = None,
        extra_lock_names: Optional[Sequence[str]] = None,
        logger=None,
    ):
        self._log = logger
        self.run_ik = bool(run_ik)
        self.posture_weight = float(posture_weight)
        self.warm_start_iters = _multiple_of(warm_start_iters, 25)
        self.cold_start_iters = max(
            self.warm_start_iters, _multiple_of(cold_start_iters, 25)
        )
        self._posture_logged = False
        self._warn_at: Dict[str, float] = {}
        self.device_cfg = DeviceCfg(device=torch.device(device), dtype=torch.float32)
        self.optimization_dt = float(optimization_dt)
        self.interpolation_steps = 4
        robot_cfg, self.locked_joints = robot_cfg_with_locked_joints(
            robot, lock_joint_values, extra_lock_names=extra_lock_names
        )
        cfg = ModelPredictiveControlCfg.create(
            robot=robot_cfg,
            scene_model=None,
            collision_cache={"mesh": 1},
            device_cfg=self.device_cfg,
            use_cuda_graph=bool(use_cuda_graph),
            optimization_dt=self.optimization_dt,
            interpolation_steps=self.interpolation_steps,
            warm_start_optimization_num_iters=self.warm_start_iters,
            cold_start_optimization_num_iters=self.cold_start_iters,
        )
        # scene_model 为空时 cuRobo 会丢掉 collision_cache，检查器不会被创建。
        # 点云要靠 load_collision_model 装进去，所以这里按 TrajOpt 的方式补一个空场景。
        if cfg.core_cfg.scene_collision_cfg is None:
            cfg.core_cfg.scene_collision_cfg = SceneCollisionCfg(
                device_cfg=self.device_cfg,
                cache={"mesh": 1},
            )
        self.mpc = ModelPredictiveControl(cfg)
        self.left_ee, self.right_ee = both_arm_ee_links()
        missing = [n for n in (self.left_ee, self.right_ee) if n not in self.mpc.tool_frames]
        if missing:
            raise ValueError(
                f"MPC tool_frames={self.mpc.tool_frames} 缺少 {missing}，无法双臂跟踪"
            )
        self._setup_done = False
        self._last_goal_key = None
        self._last_cmd_vel = None
        self._last_cmd_acc = None
        self._info(
            f"MPC 已创建，tool_frames={self.mpc.tool_frames}，"
            f"dt={self.optimization_dt} command_dt={self.command_dt:.4f} "
            f"run_ik={self.run_ik} posture_weight={self.posture_weight:g}，"
            f"warm_iters={self.warm_start_iters} cold_iters={self.cold_start_iters}，"
            f"规划关节 {len(self.joint_names)} 个"
        )
        if self.locked_joints:
            self._info(
                "已锁定（不参与 MPC，也不下发）: "
                + ", ".join(f"{k}={v:.3f}" for k, v in self.locked_joints.items())
            )

    def _info(self, msg: str):
        if self._log is not None:
            self._log.info(msg)
        else:
            print(msg, flush=True)

    def _warn(self, msg: str):
        if self._log is not None:
            self._log.warn(msg)
        else:
            print(msg, flush=True)

    def _warn_throttled(self, key: str, msg: str, period: float = 2.0) -> None:
        now = time.monotonic()
        if now - self._warn_at.get(key, 0.0) < period:
            return
        self._warn_at[key] = now
        self._warn(msg)

    @property
    def joint_names(self) -> List[str]:
        return list(self.mpc.joint_names)

    @property
    def default_joint_position(self) -> torch.Tensor:
        return self.mpc.default_joint_state.position.reshape(-1)

    @property
    def command_dt(self) -> float:
        """相邻命令点的间隔：一个 action_sequence 正好覆盖 optimization_dt。"""
        return self.optimization_dt / float(self.interpolation_steps)

    def apply_command_continuity(self, state: JointState) -> JointState:
        """速度/加速度沿用上一拍命令，避免求解器每拍以为机器人急停。"""
        if self._last_cmd_vel is None:
            return state
        state.velocity = self._last_cmd_vel.clone()
        state.acceleration = self._last_cmd_acc.clone()
        return state

    def remember_command(self, result) -> None:
        """记住本拍最后一个命令点的速度/加速度，供下一拍接力。"""
        seq = getattr(result, "action_sequence", None)
        if seq is None or seq.position is None:
            return
        vel, acc = seq.velocity, seq.acceleration
        if vel is None or acc is None:
            return
        self._last_cmd_vel = vel.detach().reshape(-1, vel.shape[-1])[-1:].clone()
        self._last_cmd_acc = acc.detach().reshape(-1, acc.shape[-1])[-1:].clone()

    def forget_command(self) -> None:
        self._last_cmd_vel = None
        self._last_cmd_acc = None

    def update_world_from_points(self, points: np.ndarray, pitch: float, name: str = "cam") -> int:
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        mesh = Mesh.from_pointcloud(pts, pitch=float(pitch), name=name or "cam")
        checker = self.mpc.core.scene_collision_checker
        if checker is None:
            raise RuntimeError("MPC 没有 scene collision checker，无法更新点云环境")
        checker.load_collision_model(Scene(mesh=[mesh]))
        return int(pts.shape[0])

    def fk_tool_poses(self, joint_state: JointState) -> Dict[str, Pose]:
        kin = self.mpc.compute_kinematics(joint_state.clone())
        if kin.tool_poses is None:
            return {}
        return {
            name: squeeze_pose(kin.tool_poses.get_link_pose(name))
            for name in self.mpc.tool_frames
        }

    def pose_error(self, current: Pose, target: Pose) -> Tuple[float, float]:
        return pose_error(current, target)

    def _ensure_setup(self, current_state: JointState) -> None:
        if self._setup_done:
            return
        self.mpc.setup(current_state)
        self._setup_done = True
        self._info("MPC setup 完成（首次较慢）")

    def _goal_key(self, left_pose: Optional[Pose], right_pose: Optional[Pose]) -> tuple:
        items = []
        for pose in (left_pose, right_pose):
            if pose is None:
                items.append(None)
                continue
            xyz, wxyz = pose_xyz_wxyz(pose)
            items.append(tuple(round(v, 3) for v in xyz + wxyz))
        return tuple(items)

    def update_goals(
        self,
        current_state: JointState,
        left_pose: Optional[Pose],
        right_pose: Optional[Pose],
        fk: Optional[Dict[str, Pose]] = None,
    ) -> Dict[str, Pose]:
        self._ensure_setup(current_state)
        if fk is None:
            fk = self.fk_tool_poses(current_state)
        used = dict(fk)
        if left_pose is not None:
            used[self.left_ee] = left_pose
        if right_pose is not None:
            used[self.right_ee] = right_pose
        key = self._goal_key(left_pose, right_pose)
        if key == self._last_goal_key:
            return used
        goal = GoalToolPose.from_poses(
            used, ordered_tool_frames=self.mpc.tool_frames, num_goalset=1
        )
        # run_ik 会额外给一个关节空间吸引点。IK 解不好（或落在另一个肘部分支）时，
        # 它会和笛卡尔代价互相拉扯，表现为末端停在半路不动，所以默认关掉。
        if self.run_ik and self.mpc.update_goal_tool_poses(goal, run_ik=True):
            self._last_goal_key = key
            return used
        if self.run_ik:
            self._warn("MPC 目标 IK 未成功，退回纯笛卡尔跟踪")
        self.mpc.update_goal_tool_poses(goal, run_ik=False)
        self._last_goal_key = key
        return used

    def _apply_posture_goal(
        self,
        current_state: JointState,
        left_pose: Optional[Pose],
        right_pose: Optional[Pose],
    ) -> None:
        """沿末端雅可比的零空间，把关节往 default_joint_position（全 0）收。

        直接把全部关节的目标设成 0 时，手的位姿对不上，短时域会扎进点云，
        整段轨迹被判成不可行，send 变成 0。零空间方向不改变末端，只收多余自由度。
        """
        weight = self.posture_weight
        if weight <= 0.0:
            return
        goal_buffer = self.mpc.core._goal_buffer
        if goal_buffer is None or goal_buffer.goal_js is None:
            self._warn_throttled("no_goal_js", "MPC 还没有 goal_js，跳过构型代价")
            return
        q = current_state.position.detach().reshape(-1)
        default = self.default_joint_position.reshape(-1).to(device=q.device, dtype=q.dtype)
        goal_js = goal_buffer.goal_js.clone()
        if goal_js.position.shape[-1] != q.numel() or default.numel() != q.numel():
            self._warn_throttled(
                "posture_shape",
                f"构型维度对不上：q={q.numel()} default={default.numel()} "
                f"goal_js={tuple(goal_js.position.shape)}",
            )
            return
        links = []
        if left_pose is not None:
            links.append(self.left_ee)
        if right_pose is not None:
            links.append(self.right_ee)
        if not links:
            links = [self.left_ee, self.right_ee]
        q_ref, dof_w = self._nullspace_toward(current_state, q, default, links)
        if q_ref is None:
            self._warn_throttled("no_null", "当前构型没有末端零空间，本拍不附加构型代价")
            return
        lower, upper = self._joint_position_limits(q.device, q.dtype)
        if lower is not None:
            q_ref = torch.max(torch.min(q_ref, upper), lower)
        goal_js.position = goal_js.position.clone()
        goal_js.position[...] = q_ref
        self.mpc.update_goal_state(goal_js)
        self._write_posture_weight(weight)
        if not self._write_dof_weight(dof_w):
            return
        self.mpc.enable_joint_position_tracking()
        if self._posture_logged:
            return
        self._posture_logged = True
        self._info(
            f"构型沿零空间收向关节 0，权重 {weight:g}，每拍最多 "
            f"{self._POSTURE_STEP:.2f} rad"
        )

    _POSTURE_STEP = 0.04

    def _nullspace_toward(
        self,
        state: JointState,
        q: torch.Tensor,
        q_des: torch.Tensor,
        links: Sequence[str],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        jac = self._ee_jacobian(state, q, links)
        if jac is None:
            return None, None
        # 奇异值相对最大者太小的右奇异向量当成零空间。
        _u, singular, vh = torch.linalg.svd(jac, full_matrices=True)
        if singular.numel() == 0 or float(singular[0]) < 1e-8:
            return None, None
        rank = int(torch.sum(singular > singular[0] * 1e-3).item())
        if rank >= vh.shape[0]:
            return None, None
        null = vh[rank:]
        dq_null = null.transpose(0, 1) @ (null @ (q_des - q))
        norm = torch.linalg.norm(dq_null)
        if float(norm) > self._POSTURE_STEP:
            dq_null = dq_null * (self._POSTURE_STEP / norm)
        basis = null.abs().sum(dim=0)
        peak = basis.max().clamp(min=1e-6)
        return q + dq_null, basis / peak

    def _ee_jacobian(
        self, state: JointState, q: torch.Tensor, links: Sequence[str]
    ) -> Optional[torch.Tensor]:
        base = self._link_pose_rows(state, links)
        if base is None:
            return None
        dof = q.numel()
        eps = 1e-4
        jac = torch.zeros(
            (base.shape[0] * 6, dof), device=q.device, dtype=q.dtype
        )
        for i in range(dof):
            q2 = q.clone()
            q2[i] = q2[i] + eps
            nxt = state.clone()
            nxt.position = q2.reshape(state.position.shape)
            pert = self._link_pose_rows(nxt, links)
            if pert is None:
                return None
            for k in range(base.shape[0]):
                jac[6 * k : 6 * k + 3, i] = (pert[k, :3] - base[k, :3]) / eps
                jac[6 * k + 3 : 6 * k + 6, i] = (
                    _rotvec_wxyz(base[k, 3:], pert[k, 3:]) / eps
                )
        return jac

    def _link_pose_rows(
        self, state: JointState, links: Sequence[str]
    ) -> Optional[torch.Tensor]:
        poses = self.fk_tool_poses(state)
        rows = []
        for name in links:
            pose = poses.get(name)
            if pose is None or pose.position is None or pose.quaternion is None:
                return None
            pos = pose.position.detach().reshape(-1)[:3]
            quat = pose.quaternion.detach().reshape(-1)[:4]
            rows.append(torch.cat((pos, quat)))
        return torch.stack(rows)

    def _joint_position_limits(self, device, dtype):
        for rollout in self.mpc.core.get_all_rollout_instances(include_optimizer_rollouts=True):
            for cost in rollout.get_cost_component_by_name("cspace"):
                if cost is None or cost.config.joint_limits is None:
                    continue
                pos = cost.config.joint_limits.position
                if pos.shape[0] == 2:
                    return pos[0].to(device=device, dtype=dtype), pos[1].to(
                        device=device, dtype=dtype
                    )
        return None, None

    def _write_dof_weight(self, dof_w: torch.Tensor) -> bool:
        wrote = False
        for rollout in self.mpc.core.get_all_rollout_instances(include_optimizer_rollouts=True):
            for cost in rollout.get_cost_component_by_name("cspace"):
                if cost is None:
                    continue
                current = cost.config.cspace_target_dof_weight
                src = dof_w.reshape(-1).to(device=current.device, dtype=current.dtype)
                if current.shape != src.shape:
                    continue
                current.copy_(src)
                wrote = True
        if not wrote:
            self._warn_throttled("dof_weight", "构型自由度权重写不进去，本拍不附加构型代价")
        return wrote

    def _write_posture_weight(self, weight: float) -> None:
        value = torch.tensor(
            [float(weight)], device=self.device_cfg.device, dtype=self.device_cfg.dtype
        )
        for rollout in self.mpc.core.get_all_rollout_instances(include_optimizer_rollouts=True):
            for cost in rollout.get_cost_component_by_name("cspace"):
                if cost is None:
                    continue
                current = cost.config.cspace_target_weight
                if current.shape != value.shape or current.device != value.device:
                    cost.config.cspace_target_weight = value.clone()
                else:
                    current.copy_(value)

    def step(self, current_state: JointState):
        self._ensure_setup(current_state)
        return self.mpc.optimize_action_sequence(current_state)

    def run(
        self,
        current_state: JointState,
        left_pose: Optional[Pose],
        right_pose: Optional[Pose],
    ) -> MpcStep:
        self._ensure_setup(current_state)
        current_ee = self.fk_tool_poses(current_state)
        used = self.update_goals(current_state, left_pose, right_pose, fk=current_ee)
        self._apply_posture_goal(current_state, left_pose, right_pose)
        result = self.step(current_state)
        self.remember_command(result)

        success = False
        solve_time = 0.0
        message = ""
        if result is not None:
            solve_time = float(result.solve_time or 0.0)
            # result.success 看的是整段时域。后面的 knot 一擦到点云，整拍就被丢掉，
            # 日志里就是第一拍 send>0、后面每拍 send=0。本拍真正执行的是前面几个命令点，
            # 有动作序列就下发。不要在不可行时从当前关节重解，那会丢掉热启动，手臂来回抖。
            success = getattr(result, "action_sequence", None) is not None
            if not success:
                message = "MPC 无动作序列"
        else:
            message = "MPC 无结果"
        return MpcStep(
            result=result,
            used=used,
            current_ee=current_ee,
            success=success,
            solve_time=solve_time,
            message=message,
        )
