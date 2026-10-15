"""双臂 MPC：两个笛卡尔目标，缺失的一侧钉在当前末端。

只吃 / 吐 cuRobo 类型。ROS 消息和 RViz 不进这里。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

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
        lock_joint_values: Optional[Dict[str, float]] = None,
        extra_lock_names: Optional[Sequence[str]] = None,
        logger=None,
    ):
        self._log = logger
        self.run_ik = bool(run_ik)
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
            f"run_ik={self.run_ik}，规划关节 {len(self.joint_names)} 个"
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
        result = self.step(current_state)
        self.remember_command(result)

        success = False
        solve_time = 0.0
        message = ""
        if result is not None:
            solve_time = float(result.solve_time or 0.0)
            if result.success is not None:
                try:
                    success = bool(result.success.detach().reshape(-1)[0].item())
                except Exception:  # noqa: BLE001
                    success = bool(result.success)
            if not success:
                message = "MPC 本周期未找到可行解"
            elif result.action_sequence is None:
                message = "MPC 无动作序列"
                success = False
        else:
            message = "MPC 无结果"
        success = bool(success and result is not None and result.action_sequence is not None)
        return MpcStep(
            result=result,
            used=used,
            current_ee=current_ee,
            success=success,
            solve_time=solve_time,
            message=message,
        )
