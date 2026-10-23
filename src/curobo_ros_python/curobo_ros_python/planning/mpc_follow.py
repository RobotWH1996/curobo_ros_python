"""MPC 周期：只在 FollowMpc Action 激活时跑。算法在 algorithms.mpc。"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np

from geometry_msgs.msg import PoseStamped
from tf2_ros import TransformException

from curobo_ros_interfaces.action import FollowMpc
from curobo_ros_interfaces.msg import MpcPlan, MpcTarget
from curobo_ros_python.planning.convert import (
    joint_map_to_curobo,
    mpc_step_to_msgs,
    stamped_to_pose,
)
from curobo_ros_python.planning.groups import GROUP_JOINTS
from curobo_ros_python.planning.pointcloud import cloud_to_base_points, drop_boxes
from curobo_ros_python.visualization.publish import publish_collision_world, publish_mpc_preview


class MpcFollower:
    def __init__(self, node):
        self._node = node
        self._mpc = None
        self._cache = {}
        self._lock_key = None
        self._target: Optional[MpcTarget] = None
        self._joint_vel = {}
        self._last_joint_pos = {}
        self._last_joint_t = None
        self._last_warn_left = 0.0
        self._last_warn_right = 0.0
        self._cloud_note = ""
        self._loaded_pc_seq = None

    def on_target(self, msg: MpcTarget):
        self._target = msg

    def on_joint_state(self, names, positions, velocities, stamp_sec: float):
        pos_map = {str(name): float(p) for name, p in zip(names, positions)}
        reported = (
            velocities
            and len(velocities) == len(names)
            and any(abs(float(v)) > 1e-4 for v in velocities)
        )
        if reported:
            for name, vel in zip(names, velocities):
                self._joint_vel[str(name)] = float(vel)
        elif self._last_joint_pos and self._last_joint_t is not None:
            dt = float(stamp_sec) - float(self._last_joint_t)
            if dt <= 1e-4:
                pass
            elif dt >= 0.5:
                for name in pos_map:
                    self._joint_vel[name] = 0.0
            else:
                for name, pos in pos_map.items():
                    prev = self._last_joint_pos.get(name)
                    if prev is None:
                        continue
                    dq = pos - prev
                    if abs(dq) < 1e-5:
                        self._joint_vel[name] = 0.0
                        continue
                    vel = dq / dt
                    if vel > 10.0:
                        vel = 10.0
                    elif vel < -10.0:
                        vel = -10.0
                    self._joint_vel[name] = vel
        else:
            for name in pos_map:
                self._joint_vel.setdefault(name, 0.0)
        self._last_joint_pos = pos_map
        self._last_joint_t = float(stamp_sec)

    def _unused_arm_joints(self) -> list:
        has_left = bool(self._target is not None and self._target.has_left)
        has_right = bool(self._target is not None and self._target.has_right)
        extra = []
        if has_left and not has_right:
            extra.extend(GROUP_JOINTS.get("right_arm") or [])
        elif has_right and not has_left:
            extra.extend(GROUP_JOINTS.get("left_arm") or [])
        return extra

    def ensure(self, extra_lock_names=None):
        """取出该模式的求解器。每种模式只在第一次创建，之后锁定关节变了也不重建。"""
        extra_lock_names = list(extra_lock_names or [])
        key = tuple(sorted(extra_lock_names))
        if self._mpc is not None and self._lock_key == key:
            return self._mpc
        from curobo_ros_python.planning.algorithms.mpc import DualArmMpc

        node = self._node
        cached = self._cache.get(key)
        if cached is not None:
            switched = self._mpc is not cached
            if switched:
                self._loaded_pc_seq = None
                cached.forget_command()
            self._mpc = cached
            self._lock_key = key
            if switched:
                self.reset()
            return cached
        node.get_logger().info("正在创建 ModelPredictiveControl（首次较慢）...")
        joint_map = dict(node._joint_map)
        self._mpc = DualArmMpc(
            robot=node._robot_yaml,
            device=str(node._p("mpc_device") or "cuda:0"),
            use_cuda_graph=bool(node._p("use_cuda_graph")),
            optimization_dt=float(node._p("mpc_optimization_dt")),
            run_ik=bool(node._p("mpc_run_ik")),
            posture_weight=float(node._p("mpc_posture_weight")),
            posture=str(node._p("mpc_posture") or "l"),
            warm_start_iters=int(node._p("mpc_warm_start_iters")),
            cold_start_iters=int(node._p("mpc_cold_start_iters")),
            num_control_points=int(node._p("mpc_num_control_points")),
            lock_joint_values=joint_map,
            extra_lock_names=extra_lock_names,
            logger=node.get_logger(),
        )
        self._lock_key = key
        self._cache[key] = self._mpc
        # 序号记的是上一份求解器。新实例的碰撞场景是空的，必须重新加载当前帧。
        self._loaded_pc_seq = None
        return self._mpc

    def prewarm_next(self, modes) -> bool:
        """每次只预热一个模式。调用者持有节点 GPU 锁；订阅回调继续收状态。"""
        node = self._node
        needed = set(GROUP_JOINTS.get("both_arms") or [])
        needed.update(node.planner.gripper_joint_names)
        if not needed or not needed.issubset(node._joint_map):
            return False
        if node.use_pointcloud and node.pointcloud_topic:
            with node._pc_lock:
                if node._pc_msg is None:
                    return False
        locks = {
            "left_arm": GROUP_JOINTS["right_arm"],
            "right_arm": GROUP_JOINTS["left_arm"],
            "both_arms": [],
        }
        for mode in modes:
            key = tuple(sorted(locks[mode]))
            cached = self._cache.get(key)
            if cached is not None and cached.prewarmed:
                continue
            started = time.monotonic()
            mpc = self.ensure(locks[mode])
            mpc.update_gripper_state(dict(node._joint_map))
            error = self._sync_world(mpc, publish=False)
            if error:
                node.get_logger().warn(f"MPC 预热等待点云：{error}", throttle_duration_sec=5.0)
                return False
            current, missing = joint_map_to_curobo(
                mpc.joint_names, dict(node._joint_map), mpc.device_cfg.device,
                default_position=mpc.default_joint_position, joint_vel=dict(self._joint_vel),
            )
            if missing:
                return False
            mpc.warmup(current, mode)
            node.get_logger().info(
                f"MPC {mode} 预热完成，耗时 {time.monotonic() - started:.2f}s（未下发运动）"
            )
            return False
        return True

    def reset(self):
        """每次 follow_mpc 开始时调用：丢掉上一轮的命令接力状态。"""
        if self._mpc is not None:
            self._mpc.update_gripper_state(dict(self._node._joint_map))
            current, _ = joint_map_to_curobo(
                self._mpc.joint_names, dict(self._node._joint_map), self._mpc.device_cfg.device,
                default_position=self._mpc.default_joint_position, joint_vel=dict(self._joint_vel),
            )
            self._mpc.prepare_tracking(current)
            # 即使点云序号没变，也重新应用最新 TF；预热场景不直接用于跟踪。
            self._loaded_pc_seq = None

    def _arm_target_stamped(self, has_flag: bool, pose: PoseStamped) -> Optional[PoseStamped]:
        if has_flag:
            return pose
        return None

    def _curobo_pose_from_stamped(self, stamped: PoseStamped):
        pose = self._node._pose_in_base(stamped)
        return stamped_to_pose(pose, self._mpc.device_cfg)

    def _warn_missing(self, arm: str, last_ts: float) -> float:
        node = self._node
        now = time.monotonic()
        period = float(node._p("mpc_missing_target_warn_period"))
        if now - last_ts >= period:
            if arm == "left":
                node.get_logger().warn("左手目标缺失，左臂保持原位")
            else:
                node.get_logger().warn("右手目标缺失，右臂保持原位")
            return now
        return last_ts

    def _sync_world(self, mpc, publish: bool = True) -> Optional[str]:
        node = self._node
        self._cloud_note = ""
        if not node.use_pointcloud or not node.pointcloud_topic:
            return None
        with node._pc_lock:
            msg = node._pc_msg
            seq = int(node._pc_seq)
        if msg is None:
            return f"无点云：尚未收到 {node.pointcloud_topic}"
        if seq == self._loaded_pc_seq:
            self._cloud_note = f"点云未更新 seq={seq}"
            return None
        src = (msg.header.frame_id or "").lstrip("/")
        try:
            translation, quat_wxyz = node._lookup_cloud_tf(src)
        except TransformException as exc:
            return f"没有点云 TF: {exc}"
        try:
            pts, _n_raw = cloud_to_base_points(
                msg,
                translation,
                quat_wxyz,
                min_range=node.pointcloud_min_range,
                max_range=node.pointcloud_max_range,
                pitch=node.pointcloud_pitch,
                max_points=node.pointcloud_max_points,
                filter_cloud=node.filter_pointcloud,
                bounds_min=node.pointcloud_bounds_min,
                bounds_max=node.pointcloud_bounds_max,
            )
        except Exception as exc:  # noqa: BLE001
            return f"解析点云失败: {exc}"
        if pts.shape[0] == 0:
            self._clear_loaded_world(mpc, seq, "这一帧没有可用点", publish=publish)
            return None
        pts = drop_boxes(pts, node.static_exclude_boxes())
        if pts.shape[0] == 0:
            self._clear_loaded_world(mpc, seq, "去掉固定夹爪盒子后没有点", publish=publish)
            return None
        self._cloud_note = _cloud_grid_note(pts, node.pointcloud_pitch)
        mpc.update_world_from_points(pts, pitch=node.pointcloud_pitch)
        if publish:
            publish_collision_world(node, pts)
        self._loaded_pc_seq = seq
        return None

    def _clear_loaded_world(self, mpc, seq: int, reason: str, publish: bool = True) -> None:
        """这一帧已经解析成功，只是没有障碍。清掉旧网格，别把它当成传感器失效。"""
        mpc.clear_world()
        if publish:
            publish_collision_world(self._node, np.zeros((0, 3), dtype=np.float32))
        self._loaded_pc_seq = seq
        self._cloud_note = f"点云 n=0 seq={seq} {reason}，已清除碰撞网格"

    def tick(self) -> Tuple[MpcPlan, FollowMpc.Feedback]:
        node = self._node
        t_lap = time.monotonic()
        laps = {}

        def lap(name: str) -> None:
            nonlocal t_lap
            now = time.monotonic()
            laps[name] = (now - t_lap) * 1000.0
            t_lap = now

        mpc = self.ensure(self._unused_arm_joints())
        # 夹爪不在 mpc.joint_names 中，必须独立刷新，不能靠 active state 带入。
        joint_map = dict(node._joint_map)
        mpc.update_gripper_state(joint_map)
        current, missing = joint_map_to_curobo(
            mpc.joint_names,
            joint_map,
            mpc.device_cfg.device,
            default_position=mpc.default_joint_position,
            joint_vel=self._joint_vel,
        )
        if missing:
            node.get_logger().warn(
                f"joint_states 缺少 {len(missing)} 个主动关节，用默认值: {missing[:6]}",
                throttle_duration_sec=5.0,
            )
        current = mpc.apply_command_continuity(current)
        node._mpc_state_read_t = time.monotonic()
        lap("state")
        world_err = self._sync_world(mpc)
        lap("world")
        if world_err:
            node.get_logger().warn(world_err + "，本周期沿用上一环境", throttle_duration_sec=2.0)
        elif not node.use_pointcloud or not node.pointcloud_topic:
            node.get_logger().info("无点云", throttle_duration_sec=2.0)

        left_msg = None
        right_msg = None
        if self._target is not None:
            left_msg = self._arm_target_stamped(bool(self._target.has_left), self._target.left)
            right_msg = self._arm_target_stamped(bool(self._target.has_right), self._target.right)
        left_pose = None
        right_pose = None
        left_note = ""
        right_note = ""
        if left_msg is None:
            self._last_warn_left = self._warn_missing("left", self._last_warn_left)
            left_note = "左手目标缺失，左臂保持原位"
        else:
            try:
                left_pose = self._curobo_pose_from_stamped(left_msg)
            except Exception as exc:  # noqa: BLE001
                left_note = f"左手目标 TF/位姿无效: {exc}"
                node.get_logger().warn(left_note)
        if right_msg is None:
            self._last_warn_right = self._warn_missing("right", self._last_warn_right)
            right_note = "右手目标缺失，右臂保持原位"
        else:
            try:
                right_pose = self._curobo_pose_from_stamped(right_msg)
            except Exception as exc:  # noqa: BLE001
                right_note = f"右手目标 TF/位姿无效: {exc}"
                node.get_logger().warn(right_note)

        lap("target")
        mpc.set_posture(str(node._p("mpc_posture") or "l"))
        step = mpc.run(current, left_pose, right_pose)
        lap("solve")
        publish_mpc_preview(node, mpc, step)
        lap("preview")

        stamp = node.get_clock().now().to_msg()
        out, fb = mpc_step_to_msgs(
            step,
            left_ee=mpc.left_ee,
            right_ee=mpc.right_ee,
            left_valid=left_pose is not None,
            right_valid=right_pose is not None,
            left_note=left_note,
            right_note=right_note,
            joint_map=node._joint_map,
            command_dt=mpc.command_dt,
            velocity_scale=float(node.mpc_velocity_scale),
            frame_id=node.base_frame,
            stamp=stamp,
            joint_names=mpc.joint_names,
        )
        lap("pack")
        timing = "MPC tick 分段(ms): " + " ".join(f"{k}={v:.1f}" for k, v in laps.items())
        if self._cloud_note:
            timing = f"{timing} {self._cloud_note}"
        node.get_logger().info(timing, throttle_duration_sec=2.0)
        return out, fb


def _cloud_grid_note(pts, pitch: float) -> str:
    """和 Mesh.from_pointcloud 同一张格子：包围盒有多大，CPU 数组就有多少格。"""
    arr = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    keep = np.isfinite(arr).all(axis=1)
    arr = arr[keep]
    if arr.shape[0] == 0:
        return "点云 n=0"
    lo = arr.min(axis=0)
    span = arr.max(axis=0) - lo
    pitch = max(float(pitch), 1e-4)
    ijk = np.floor((arr - (lo - pitch)) / pitch).astype(np.int64)
    cells = ijk.max(axis=0) + 3
    n_cells = int(cells[0]) * int(cells[1]) * int(cells[2])
    far = float(np.linalg.norm(arr, axis=1).max())
    return (
        f"点云 n={arr.shape[0]} far={far:.2f}m "
        f"span=({span[0]:.2f},{span[1]:.2f},{span[2]:.2f})m "
        f"cells={int(cells[0])}x{int(cells[1])}x{int(cells[2])}={n_cells}"
    )
