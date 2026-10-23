"""cuRobo ROS 2 运动规划节点。

节点与 cuRobo 跑在同一个 Python 3.10 解释器里（conda 环境需同时有 rclpy 和 torch）。
机器人由参数 ``robot``（cuRobo YAML，默认 lycus.yml）选择。
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional, Sequence, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState, PointCloud2
from tf2_ros import Buffer, TransformException, TransformListener
from trajectory_msgs.msg import JointTrajectory

from curobo_ros_interfaces.action import FollowMpc
from curobo_ros_interfaces.msg import MpcPlan, MpcTarget
from curobo_ros_interfaces.srv import ComputeIK, PlanJointGoal, PlanPoseGoal
from curobo_ros_python.planning.geom import transform_pose_wxyz
from curobo_ros_python.planning.groups import (
    GROUP_JOINTS,
    apply_robot,
    both_arm_ee_links,
    resolve_ee_link,
    unknown_group,
)
from curobo_ros_python.planning.algorithms.trajopt import CuroboPlanner, TrajResult
from curobo_ros_python.planning.convert import (
    curobo_traj_to_msg,
    group_goal_state,
    ik_group_angles,
    joint_map_to_curobo,
    stamped_to_pose,
)
from curobo_ros_python.planning.mpc_follow import MpcFollower
from curobo_ros_python.planning.pointcloud import (
    cloud_to_base_points,
    drop_boxes,
    exclude_boxes_from_param,
)
from curobo_ros_python.planning.trajectory import (
    TrajectoryExecutor,
    copy_joint_trajectory,
    drop_passed_prefix,
    joint_map_from_msg,
    trajectory_is_finite,
)

_PARAM_DEFAULTS: Dict[str, Any] = {
    "robot": "lycus.yml",
    "scene_model": "",
    "warmup_iterations": 3,
    "use_cuda_graph": True,
    "interpolation_dt": 0.02,
    "trajopt_num_seeds": 2,
    "trajopt_finetune_attempts": 1,
    "trajopt_num_iters": 50,
    "position_tolerance": 0.005,
    "orientation_tolerance": 0.05,
    "waypoint_joint_tolerance": 0.1,
    "trajopt_velocity_scale": 0.5,
    "mpc_velocity_scale": 0.5,
    "execute_trajectory": True,
    "switch_controllers": True,
    "joint_states_topic": "/joint_states",
    "base_frame": "base_link",
    "execute_timeout": 60.0,
    "robot_model_id": "",
    "planned_trajectory_topic": "/curobo/planned_trajectory",
    "trajectory_marker_topic": "",
    "use_pointcloud": True,
    "visualize_trajectory": True,
    "compute_ee_paths": True,
    "visualize_mesh_obstacles": True,
    "visualize_goal_waypoints": True,
    "pointcloud_topic": "/camera/depth_registered/points",
    "pointcloud_collision": "mesh",
    "pointcloud_voxel_dims": [2.0, 2.0, 2.0],
    "pointcloud_voxel_center": [0.2, 0.0, 1.0],
    "pointcloud_pitch": 0.02,
    "pointcloud_max_points": 40000,
    "pointcloud_min_range": 0.2,
    "pointcloud_max_range": 3.0,
    "filter_pointcloud": False,
    # 躯干锁在 0 时，肩高 1.24 m、肩距 ±0.08 m，臂长 0.87 m，再留 0.18 m 给手。
    "pointcloud_bounds_min": [0.00, -1.15, 0.20],
    "pointcloud_bounds_max": [1.05, 1.15, 1.90],
    "pointcloud_exclude_boxes": "",
    "require_pointcloud": False,
    "pointcloud_parent_frame": "base_link",
    "pointcloud_child_frame": "camera_depth_optical_frame",
    "mpc_device": "cuda:0",
    "mpc_optimization_dt": 0.14,
    "mpc_control_period": 0.10,
    "mpc_run_ik": False,
    "mpc_posture_weight": 5.0,
    "mpc_posture": "l",
    "mpc_warm_start_iters": 50,
    "mpc_cold_start_iters": 100,
    "mpc_num_control_points": 6,
    "mpc_prewarm": True,
    "mpc_prewarm_modes": ["left_arm", "right_arm", "both_arms"],
    "mpc_target_topic": "/curobo/target",
    "mpc_plan_topic": "/curobo/mpc_plan",
    "mpc_missing_target_warn_period": 2.0,
}


def _pose_xyz_xyzw(pose: PoseStamped) -> Tuple[list, list]:
    p = pose.pose.position
    q = pose.pose.orientation
    return [p.x, p.y, p.z], [q.x, q.y, q.z, q.w]


def _fmt_named_joints(names: Sequence[str], values: Sequence[float]) -> str:
    pairs = [f"{name}={float(val):.4f}" for name, val in zip(names, values)]
    extra = len(values) - len(names)
    if extra > 0:
        pairs.append(f"(多 {extra} 个数)")
    elif extra < 0:
        pairs.append(f"(少 {-extra} 个数)")
    return ", ".join(pairs)


def _fmt_pose(xyz: Sequence[float], xyzw: Sequence[float], frame: str = "") -> str:
    frame_s = f" frame={frame}" if frame else ""
    return (
        f"xyz=({xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f}) "
        f"xyzw=({xyzw[0]:.4f}, {xyzw[1]:.4f}, {xyzw[2]:.4f}, {xyzw[3]:.4f})"
        f"{frame_s}"
    )


class _LapTimer:
    def __init__(self):
        now = time.monotonic()
        self._t = now
        self._t0 = now
        self.laps: Dict[str, float] = {}

    def lap(self, name: str) -> None:
        now = time.monotonic()
        self.laps[name] = (now - self._t) * 1000.0
        self._t = now

    def total_ms(self) -> float:
        return (time.monotonic() - self._t0) * 1000.0

    def summary(self) -> str:
        segs = " ".join(f"{k}={v:.1f}" for k, v in self.laps.items())
        return f"{segs} total={self.total_ms():.1f}"


class CuroboMotionPlanningNode(Node):
    def __init__(self):
        super().__init__("curobo_motion_planning_node")
        self._service_group = MutuallyExclusiveCallbackGroup()
        self._sub_group = ReentrantCallbackGroup()
        self._action_group = ReentrantCallbackGroup()
        self._gpu_lock = threading.Lock()
        self._mpc_start_lock = threading.Lock()
        for name, default in _PARAM_DEFAULTS.items():
            self.declare_parameter(name, default)

        self.execute_trajectory = bool(self._p("execute_trajectory"))
        self.visualize_trajectory = bool(self._p("visualize_trajectory"))
        self.visualize_goal_waypoints = bool(self._p("visualize_goal_waypoints"))
        self.compute_ee_paths = bool(self._p("compute_ee_paths"))
        self.base_frame = self._p("base_frame") or "base_link"
        self.execute_timeout = self._p("execute_timeout")
        self.trajopt_velocity_scale = self._p("trajopt_velocity_scale")
        self.mpc_velocity_scale = self._p("mpc_velocity_scale")

        self.get_logger().info("========== 初始化 cuRobo Motion Planning Node ==========")
        robot_yaml = self._p("robot")
        self._robot_yaml = robot_yaml
        self.robot_id = apply_robot(robot_yaml)
        explicit_model = (self._p("robot_model_id") or "").strip()
        self.robot_model_id = explicit_model or self.robot_id
        marker_topic = (self._p("trajectory_marker_topic") or "").strip()
        self._marker_topic = marker_topic or f"/{self.robot_id}_motion_trajectory"
        self.get_logger().info(
            f"规划组配置: {self.robot_id} ({robot_yaml})，RViz model_id={self.robot_model_id}"
        )
        self.get_logger().info(
            f"可视化 {'开' if self.visualize_trajectory else '关'}，"
            f"目标路点 {'开' if self.visualize_goal_waypoints else '关'}，"
            f"仿真/调试执行 {'开' if self.execute_trajectory else '关'}"
        )
        self.pointcloud_collision = self._collision_mode()
        self.pointcloud_voxel_dims, self.pointcloud_voxel_center = self._voxel_box()
        collision_cache = None
        if self.pointcloud_collision == "voxel":
            from curobo_ros_python.planning.pointcloud import snap_voxel_dims

            snapped = snap_voxel_dims(self.pointcloud_voxel_dims, self._p("pointcloud_pitch"))
            self.pointcloud_voxel_dims = list(snapped)
            collision_cache = {
                "mesh": 1,
                "voxel": {
                    "layers": 1,
                    "dims": self.pointcloud_voxel_dims,
                    "voxel_size": float(self._p("pointcloud_pitch")),
                },
            }
            self.get_logger().info(
                "TrajOpt 点云碰撞: voxel（ESDF），"
                f"dims={tuple(round(v, 3) for v in self.pointcloud_voxel_dims)} m，"
                f"center={tuple(round(v, 3) for v in self.pointcloud_voxel_center)} m，"
                f"pitch={float(self._p('pointcloud_pitch')):.3f} m"
            )
        else:
            self.get_logger().info("TrajOpt 点云碰撞: mesh")
        self._world_timing = {"cloud": 0.0, "scene": 0.0}
        self.planner = CuroboPlanner(
            robot=robot_yaml,
            scene_model=self._p("scene_model") or None,
            warmup_iterations=int(self._p("warmup_iterations")),
            use_cuda_graph=self._p("use_cuda_graph"),
            interpolation_dt=self._p("interpolation_dt"),
            position_tolerance=self._p("position_tolerance"),
            orientation_tolerance=self._p("orientation_tolerance"),
            waypoint_joint_tolerance=self._p("waypoint_joint_tolerance"),
            collision_cache=collision_cache,
            num_trajopt_seeds=int(self._p("trajopt_num_seeds")),
            trajopt_num_iters=int(self._p("trajopt_num_iters")),
            finetune_attempts=int(self._p("trajopt_finetune_attempts")),
            logger=self.get_logger(),
        )
        self.executor_helper = None
        if self.execute_trajectory:
            self.executor_helper = TrajectoryExecutor(
                self, switch_controllers=self._p("switch_controllers")
            )

        self._joint_map: Dict[str, float] = {}
        self._mpc_following = False
        self._mpc_preparing = False
        self._mpc_ready_logged = False
        self._last_mpc_command: Optional[Tuple[Optional[JointTrajectory], Optional[JointTrajectory]]] = None
        self._mpc = MpcFollower(self)
        self.create_subscription(
            JointState,
            self._p("joint_states_topic"),
            self._on_joint_state,
            50,
            callback_group=self._sub_group,
        )
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._setup_pointcloud()

        traj_topic = self._p("planned_trajectory_topic") or "/curobo/planned_trajectory"
        self._traj_topic = traj_topic
        self._traj_pub = self.create_publisher(JointTrajectory, traj_topic, 1)
        self._display_pub = None
        self._marker_pub = None
        self._world_marker_pub = None
        self._goal_marker_pub = None
        if self.visualize_trajectory:
            from moveit_msgs.msg import DisplayTrajectory
            from visualization_msgs.msg import MarkerArray

            self._display_pub = self.create_publisher(
                DisplayTrajectory, "/display_planned_path", 1
            )
            self._marker_pub = self.create_publisher(MarkerArray, self._marker_topic, 1)
        if self.visualize_mesh_obstacles:
            from visualization_msgs.msg import MarkerArray

            latch = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self._world_marker_pub = self.create_publisher(
                MarkerArray, "/curobo/collision_world", latch
            )
        if self.visualize_goal_waypoints:
            from visualization_msgs.msg import MarkerArray

            latch = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self._goal_marker_pub = self.create_publisher(
                MarkerArray, "/curobo/goal_waypoints", latch
            )
            self.get_logger().info("目标路点: /curobo/goal_waypoints（规划失败也保留）")
        for srv_type, name, cb in (
            (PlanJointGoal, "plan_joint_goal", self._on_plan_joint),
            (PlanPoseGoal, "plan_pose_goal", self._on_plan_pose),
            (ComputeIK, "compute_ik", self._on_compute_ik),
        ):
            self.create_service(srv_type, name, cb, callback_group=self._service_group)
        qos_latest = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(
            MpcTarget,
            self._p("mpc_target_topic"),
            self._mpc.on_target,
            qos_latest,
            callback_group=self._sub_group,
        )
        self._mpc_plan_pub = self.create_publisher(
            MpcPlan, self._p("mpc_plan_topic"), 1
        )
        self._follow_mpc = ActionServer(
            self,
            FollowMpc,
            "follow_mpc",
            execute_callback=self._execute_follow_mpc,
            goal_callback=self._follow_mpc_goal,
            cancel_callback=self._follow_mpc_cancel,
            callback_group=self._action_group,
        )
        self._mpc_prewarm_timer = None
        self._initialization_logged = False
        if bool(self._p("mpc_prewarm")):
            modes = list(self._p("mpc_prewarm_modes"))
            if not modes or any(mode not in ("left_arm", "right_arm", "both_arms") for mode in modes):
                raise ValueError("mpc_prewarm_modes 必须包含 left_arm / right_arm / both_arms 中的模式")
            self._mpc_prewarm_timer = self.create_timer(
                1.0, self._prewarm_mpc, callback_group=self._service_group
            )
            self.get_logger().info(f"MPC 将在收到关节状态和点云后提前预热：{modes}")
            self.get_logger().info("基础节点已启动，等待 MPC 预热完成...")
        else:
            self._log_initialization_complete()

    def _log_initialization_complete(self) -> None:
        if self._initialization_logged:
            return
        self._initialization_logged = True
        self.get_logger().info(
            "服务已就绪: plan_joint_goal / plan_pose_goal / compute_ik；"
            "Action: follow_mpc（目标 /curobo/target，结果 /curobo/mpc_plan）"
        )
        self.get_logger().info("========== cuRobo Motion Planning Node 初始化完成 ==========")

    def _p(self, name: str):
        return self.get_parameter(name).value

    def _collision_mode(self) -> str:
        mode = str(self._p("pointcloud_collision") or "mesh").strip().lower()
        if mode not in ("mesh", "voxel"):
            raise ValueError(
                f"pointcloud_collision 只能是 mesh 或 voxel，收到 {mode!r}"
            )
        return mode

    def _vec3(self, name: str) -> list:
        raw = self._p(name)
        vals = [float(v) for v in list(raw)]
        if len(vals) != 3:
            raise ValueError(f"{name} 需要 3 个数，收到 {vals}")
        return vals

    def _voxel_box(self) -> tuple:
        return self._vec3("pointcloud_voxel_dims"), self._vec3("pointcloud_voxel_center")

    def _setup_pointcloud(self):
        self.use_pointcloud = bool(self._p("use_pointcloud"))
        self.visualize_mesh_obstacles = bool(self._p("visualize_mesh_obstacles"))
        self.pointcloud_topic = self._p("pointcloud_topic")
        self.pointcloud_pitch = self._p("pointcloud_pitch")
        self.pointcloud_max_points = int(self._p("pointcloud_max_points"))
        self.pointcloud_min_range = self._p("pointcloud_min_range")
        self.pointcloud_max_range = self._p("pointcloud_max_range")
        self.filter_pointcloud = self._p("filter_pointcloud")
        self.pointcloud_bounds_min = self._vec3("pointcloud_bounds_min")
        self.pointcloud_bounds_max = self._vec3("pointcloud_bounds_max")
        if any(lo >= hi for lo, hi in zip(self.pointcloud_bounds_min, self.pointcloud_bounds_max)):
            raise ValueError(
                "pointcloud_bounds_min 必须小于 pointcloud_bounds_max，收到 "
                f"{self.pointcloud_bounds_min} / {self.pointcloud_bounds_max}"
            )
        self.require_pointcloud = self._p("require_pointcloud")
        self.pointcloud_parent_frame = (
            self._p("pointcloud_parent_frame") or self.base_frame
        ).lstrip("/")
        self.pointcloud_child_frame = (self._p("pointcloud_child_frame") or "").lstrip("/")
        self._pc_lock = threading.Lock()
        self._pc_msg: Optional[PointCloud2] = None
        self._pc_seq = 0
        self._last_world_pts = None

        if not self.use_pointcloud:
            self.get_logger().info("点云碰撞未启用（use_pointcloud:=false），规划按空场景")
            return

        if self.pointcloud_parent_frame != self.base_frame.lstrip("/"):
            self.get_logger().warn(
                f"pointcloud_parent_frame={self.pointcloud_parent_frame} "
                f"与 base_frame={self.base_frame} 不一致，仍把点云变到 parent_frame"
            )

        if not self.pointcloud_topic:
            self.get_logger().info("点云碰撞未启用（pointcloud_topic 为空）")
            return
        self.create_subscription(
            PointCloud2,
            self.pointcloud_topic,
            self._on_pointcloud,
            1,
            callback_group=self._sub_group,
        )
        child = self.pointcloud_child_frame or "header.frame_id"
        self.get_logger().info(
            f"点云碰撞: 订阅 {self.pointcloud_topic}，"
            f"TrajOpt 用 {self.pointcloud_collision}"
            f"（滤波 {'开' if self.filter_pointcloud else '关'}，"
            f"体素可视化 {'开' if self.visualize_mesh_obstacles else '关'}）"
        )
        lo = self.pointcloud_bounds_min
        hi = self.pointcloud_bounds_max
        self.get_logger().info(
            f"点云坐标系 {child} -> {self.pointcloud_parent_frame} 规划前查 TF；"
            f"只保留 base_link 盒子 "
            f"x[{lo[0]:.2f},{hi[0]:.2f}] y[{lo[1]:.2f},{hi[1]:.2f}] z[{lo[2]:.2f},{hi[2]:.2f}] m"
        )

    def destroy_node(self):
        try:
            self.planner.destroy()
        except Exception:  # noqa: BLE001
            pass
        super().destroy_node()

    def _on_joint_state(self, msg: JointState):
        self._joint_map.update(joint_map_from_msg(msg.name, msg.position))
        stamp = msg.header.stamp
        t = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        if t <= 0.0:
            t = self.get_clock().now().nanoseconds * 1e-9
        self._mpc.on_joint_state(msg.name, msg.position, msg.velocity, t)

    def _on_pointcloud(self, msg: PointCloud2):
        with self._pc_lock:
            self._pc_msg = msg
            self._pc_seq += 1

    def _pc_skip_or_fail(self, err: str, *, empty_scene_note: bool = False) -> Optional[str]:
        if self.require_pointcloud:
            return err
        self.get_logger().warn(err + ("，本次按空场景规划" if empty_scene_note else ""))
        return None

    def _lookup_cloud_tf(self, src_frame: str):
        src = self.pointcloud_child_frame or (src_frame or self.base_frame).lstrip("/")
        dst = self.pointcloud_parent_frame or self.base_frame.lstrip("/")
        if not src or src == dst:
            return [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]
        tf = self._tf_buffer.lookup_transform(dst, src, rclpy.time.Time())
        t = tf.transform.translation
        r = tf.transform.rotation
        return [t.x, t.y, t.z], [r.w, r.x, r.y, r.z]

    def static_exclude_boxes(self):
        """base_link 下的固定剔除盒。仿真 bag 里的夹爪用，真机留空。"""
        if getattr(self, "_exclude_boxes_ready", False):
            return self._exclude_boxes
        self._exclude_boxes_ready = True
        self._exclude_boxes = None
        try:
            self._exclude_boxes = exclude_boxes_from_param(self._p("pointcloud_exclude_boxes"))
        except (TypeError, ValueError) as exc:
            self.get_logger().warn(f"pointcloud_exclude_boxes 无法解析: {exc}")
        if self._exclude_boxes is not None:
            self.get_logger().info(
                f"点云去掉 {int(self._exclude_boxes.shape[0])} 个固定盒子（base_link，仿真夹爪）"
            )
        return self._exclude_boxes

    def _mark_world_timing(self, started: float, scene_started: Optional[float] = None) -> None:
        now = time.perf_counter()
        if scene_started is None:
            self._world_timing = {"cloud": now - started, "scene": 0.0}
        else:
            self._world_timing = {
                "cloud": scene_started - started,
                "scene": now - scene_started,
            }

    def _planning_timing_note(self, timer: _LapTimer) -> str:
        world = self._world_timing
        stages = getattr(self.planner, "last_timing", {}) or {}

        def ms(source: Dict[str, float], key: str) -> float:
            return float(source.get(key, 0.0)) * 1000.0

        parts = (
            f"cloud={ms(world, 'cloud'):.0f}ms "
            f"scene={ms(world, 'scene'):.0f}ms "
            f"ik={ms(stages, 'ik'):.0f}ms "
            f"prm={ms(stages, 'prm'):.0f}ms "
            f"trajopt={ms(stages, 'trajopt'):.0f}ms "
            f"check={ms(stages, 'check'):.0f}ms "
            f"wall={timer.total_ms():.0f}ms"
        )
        return f"计时 {parts}"

    def _sync_world_from_pointcloud(self, group_name: Optional[str] = None) -> Optional[str]:
        started = time.perf_counter()
        if not self.use_pointcloud or not self.pointcloud_topic:
            self._world_timing = {"cloud": 0.0, "scene": 0.0}
            return None
        with self._pc_lock:
            msg = self._pc_msg
        if msg is None:
            self._mark_world_timing(started)
            return self._pc_skip_or_fail(
                f"尚未收到点云 {self.pointcloud_topic}，无法更新碰撞场景",
                empty_scene_note=True,
            )
        src = (msg.header.frame_id or "").lstrip("/")
        child = self.pointcloud_child_frame or src
        parent = self.pointcloud_parent_frame
        try:
            translation, quat_wxyz = self._lookup_cloud_tf(src)
        except TransformException as exc:
            err = (
                f"没有 TF {child} -> {parent}，点云无法进规划坐标系。详情: {exc}"
            )
            self._mark_world_timing(started)
            return self._pc_skip_or_fail(err, empty_scene_note=True)
        try:
            pts, n_raw = cloud_to_base_points(
                msg,
                translation,
                quat_wxyz,
                min_range=self.pointcloud_min_range,
                max_range=self.pointcloud_max_range,
                pitch=self.pointcloud_pitch,
                max_points=self.pointcloud_max_points,
                filter_cloud=self.filter_pointcloud,
                bounds_min=self.pointcloud_bounds_min,
                bounds_max=self.pointcloud_bounds_max,
            )
        except Exception as exc:  # noqa: BLE001
            self._mark_world_timing(started)
            return self._pc_skip_or_fail(f"解析点云失败: {exc}")
        if pts.shape[0] == 0:
            empty_msg = (
                "点云滤波后为空，未更新碰撞场景"
                if self.filter_pointcloud
                else "点云为空，未更新碰撞场景"
            )
            self._mark_world_timing(started)
            return self._pc_skip_or_fail(empty_msg)
        pts = drop_boxes(pts, self.static_exclude_boxes())
        if pts.shape[0] == 0:
            self._mark_world_timing(started)
            return self._pc_skip_or_fail("点云去掉固定夹爪盒子后为空，未更新碰撞场景")
        scene_started = time.perf_counter()
        if self.pointcloud_collision == "voxel":
            n_used, n_inside = self.planner.update_world_from_esdf(
                pts,
                pitch=self.pointcloud_pitch,
                dims=self.pointcloud_voxel_dims,
                center=self.pointcloud_voxel_center,
            )
            if n_inside == 0:
                self.get_logger().warn(
                    "点云全部落在 ESDF 盒子外，本次按盒子内无障碍规划",
                    throttle_duration_sec=2.0,
                )
        else:
            n_used = self.planner.update_world_from_points(
                pts, pitch=self.pointcloud_pitch, name="cam", group_name=group_name
            )
        self._mark_world_timing(started, scene_started)
        self._last_world_pts = pts
        viz_note = ""
        from curobo_ros_python.visualization.publish import publish_collision_world

        if publish_collision_world(self, pts):
            viz_note = "；碰撞体素已发布 /curobo/collision_world"
        child = self.pointcloud_child_frame or src
        self.get_logger().info(
            f"已导入点云到 cuRobo: 原始 {n_raw} 点 -> {n_used} 点 "
            f"(frame {child} -> {self.pointcloud_parent_frame}, "
            f"pitch={self.pointcloud_pitch:.3f} m，外参来自 TF){viz_note}"
        )
        return None

    def _require_group(self, group_name: str, waypoints=None) -> Optional[str]:
        err = unknown_group(group_name)
        if err:
            return err
        if not self._joint_map:
            return "尚未收到 /joint_states，无法规划"
        if waypoints is not None and not waypoints:
            return "waypoints 为空"
        return None

    def _busy_mpc(self) -> Optional[str]:
        if self._mpc_following:
            return "MPC 跟踪进行中，请先取消 /follow_mpc"
        return None

    def _prewarm_mpc(self) -> None:
        """空闲时串行预热三个模式，与 TrajOpt / FollowMpc 共用 GPU 锁。"""
        with self._mpc_start_lock:
            if self._mpc_following or not self._gpu_lock.acquire(blocking=False):
                return
            self._mpc_preparing = True
        try:
            ready = self._mpc.prewarm_next(self._p("mpc_prewarm_modes"))
            if ready and not self._mpc_ready_logged:
                self.get_logger().info("MPC ready：指定模式均已预热，可以启动 /follow_mpc")
                self._log_initialization_complete()
            self._mpc_ready_logged = ready
            if ready and self._mpc_prewarm_timer is not None:
                self._mpc_prewarm_timer.cancel()
                self._mpc_prewarm_timer = None
        except Exception as exc:  # noqa: BLE001
            self._mpc_ready_logged = False
            self.get_logger().error(f"MPC 提前预热失败，将重试：{exc}", throttle_duration_sec=5.0)
        finally:
            self._gpu_lock.release()
            with self._mpc_start_lock:
                self._mpc_preparing = False

    def _follow_mpc_goal(self, _request):
        with self._mpc_start_lock:
            if self._mpc_preparing:
                self.get_logger().warn("MPC 正在提前预热，请等待 MPC ready 后再启动跟踪")
                return GoalResponse.REJECT
            if self._mpc_following:
                self.get_logger().warn("已有 follow_mpc 在跑，拒绝新 Goal")
                return GoalResponse.REJECT
            self._mpc_following = True
            return GoalResponse.ACCEPT

    def _follow_mpc_cancel(self, _handle):
        self.get_logger().info("收到 follow_mpc 取消")
        return CancelResponse.ACCEPT

    def _mpc_reached(self, fb: FollowMpc.Feedback, pos_tol: float, ori_tol: float) -> bool:
        if pos_tol <= 0.0 and ori_tol <= 0.0:
            return False
        checked = False
        if fb.left_target_valid:
            checked = True
            if pos_tol > 0.0 and fb.left_position_error > pos_tol:
                return False
            if ori_tol > 0.0 and fb.left_orientation_error > ori_tol:
                return False
        if fb.right_target_valid:
            checked = True
            if pos_tol > 0.0 and fb.right_position_error > pos_tol:
                return False
            if ori_tol > 0.0 and fb.right_orientation_error > ori_tol:
                return False
        return checked

    def _send_mpc_to_controllers(self, plan: MpcPlan) -> None:
        if not self.execute_trajectory or self.executor_helper is None:
            return
        if not plan.success:
            return
        left = plan.left.trajectory if plan.left.target_valid else None
        right = plan.right.trajectory if plan.right.target_valid else None
        if left is None and right is None:
            return
        finite = (left is None or trajectory_is_finite(left)) and (
            right is None or trajectory_is_finite(right)
        )
        if not finite:
            cached = self._last_mpc_command
            if cached is None:
                self.get_logger().warn(
                    "本拍轨迹含 NaN，尚无有效轨迹，本拍不下发",
                    throttle_duration_sec=2.0,
                )
                return
            left, right = cached
            self.get_logger().warn(
                "本拍轨迹含 NaN，改下发上一拍有效轨迹",
                throttle_duration_sec=2.0,
            )
        else:
            self._last_mpc_command = (
                None if left is None else copy_joint_trajectory(left),
                None if right is None else copy_joint_trajectory(right),
            )
        # 不要再把轨迹整体后移一个 knot：delay 等于 tick 时，
        # 第一个点永远在「下一拍」，每拍顶替后命令根本执行不到。
        # 第一个点已经在读数之后一个 knot。算得比这更慢时，丢掉超前的前缀，
        # 至少留 0.35 s，下一拍再慢也不会把指令裁没。
        read_t = getattr(self, "_mpc_state_read_t", None)
        elapsed = time.monotonic() - read_t if read_t else 0.0
        algo = getattr(self._mpc, "_mpc", None)
        lookahead = float(getattr(algo, "optimization_dt", 0.14) or 0.14)
        skip = max(0.0, elapsed - lookahead)
        actual = 0.0
        if skip > 1e-3:
            if left is not None:
                left, actual = drop_passed_prefix(left, skip, 0.35)
            if right is not None:
                right, right_actual = drop_passed_prefix(right, skip, 0.35)
                actual = max(actual, right_actual)
            kept = left if left is not None else right
            if actual > 1e-3 and kept is not None and algo is not None:
                algo.note_command_skip(actual)
                last = kept.points[-1].time_from_start
                dur = float(last.sec) + float(last.nanosec) * 1e-9
                self.get_logger().info(
                    f"本拍 {elapsed*1000:.0f} ms，指令丢掉前 {actual*1000:.0f} ms，"
                    f"剩下 {len(kept.points)} 点 {dur:.3f} s",
                    throttle_duration_sec=2.0,
                )
        self.executor_helper.send_mpc(left, right, start_delay=0.0)

    def _execute_follow_mpc(self, goal_handle):
        result = FollowMpc.Result()
        held_gpu = False
        try:
            if not self._joint_map:
                result.success = False
                result.message = "尚未收到 /joint_states，无法跟踪"
                goal_handle.abort()
                return result
            if not self._gpu_lock.acquire(blocking=False):
                result.success = False
                result.message = "TrajOpt 规划进行中，无法启动 MPC"
                goal_handle.abort()
                return result
            held_gpu = True
            goal = goal_handle.request
            timeout = float(goal.timeout)
            pos_tol = float(goal.position_tolerance)
            ori_tol = float(goal.orientation_tolerance)
            period = max(1e-3, float(self._p("mpc_control_period")))
            started = time.monotonic()
            self.get_logger().info(
                f"follow_mpc 开始：timeout={timeout:.1f}s，"
                f"pos_tol={pos_tol:.4f}，ori_tol={ori_tol:.4f}，周期={period:.3f}s，"
                f"下发控制器={'开' if self.execute_trajectory and self.executor_helper else '关'}"
            )
            from curobo_ros_python.visualization.publish import clear_trajopt_preview

            clear_trajopt_preview(self)
            self._last_mpc_command = None
            self._mpc.ensure(self._mpc._unused_arm_joints())
            self._mpc.reset()
            while rclpy.ok() and goal_handle.is_active:
                cycle_t0 = time.monotonic()
                t_lap = cycle_t0
                laps = {}

                def lap(name: str) -> None:
                    nonlocal t_lap
                    now = time.monotonic()
                    laps[name] = (now - t_lap) * 1000.0
                    t_lap = now

                if goal_handle.is_cancel_requested:
                    result.success = False
                    result.message = "已取消"
                    self.get_logger().info("follow_mpc 已取消")
                    goal_handle.canceled()
                    return result
                if timeout > 0.0 and (time.monotonic() - started) >= timeout:
                    result.success = False
                    result.message = f"超时 {timeout:.1f}s"
                    self.get_logger().warn(result.message)
                    goal_handle.abort()
                    return result
                try:
                    plan, feedback = self._mpc.tick()
                except Exception as exc:  # noqa: BLE001
                    result.success = False
                    result.message = f"MPC 周期异常: {exc}"
                    self.get_logger().error(result.message)
                    goal_handle.abort()
                    return result
                lap("tick")
                self._mpc_plan_pub.publish(plan)
                lap("publish")
                self._send_mpc_to_controllers(plan)
                lap("send")
                goal_handle.publish_feedback(feedback)
                lap("feedback")
                if self._mpc_reached(feedback, pos_tol, ori_tol):
                    result.success = True
                    result.message = "已到达目标容差"
                    self.get_logger().info(result.message)
                    goal_handle.succeed()
                    return result
                total_ms = (time.monotonic() - cycle_t0) * 1000.0
                segs = " ".join(f"{k}={v:.1f}" for k, v in laps.items())
                self.get_logger().info(
                    f"MPC 周期(ms): {segs} total={total_ms:.1f}",
                    throttle_duration_sec=2.0,
                )
                remain = period - (time.monotonic() - cycle_t0)
                if remain > 0.0:
                    time.sleep(remain)
                else:
                    self.get_logger().warn(
                        f"MPC 本周期超时: 耗时 {total_ms:.1f} ms，"
                        f"设定 {period*1000:.1f} ms（{segs}），无法维持目标频率",
                        throttle_duration_sec=2.0,
                    )
            result.success = False
            result.message = "节点关闭"
            if goal_handle.is_active:
                goal_handle.abort()
            return result
        finally:
            with self._mpc_start_lock:
                self._mpc_following = False
            if self.executor_helper is not None:
                self.executor_helper.cancel_sent()
            try:
                from curobo_ros_python.visualization.publish import clear_mpc_preview

                clear_mpc_preview(self)
            except Exception:  # noqa: BLE001
                pass
            if held_gpu:
                self._gpu_lock.release()


    def _reject(self, response, message: str, *, log: bool = False, angles: bool = False):
        response.success = False
        response.message = message
        if angles:
            response.joint_angles = []
        if log:
            self.get_logger().error(message)
        return response

    def _log_group_current(self, group_name: str):
        joints = GROUP_JOINTS[group_name]
        cur = [self._joint_map.get(n, float("nan")) for n in joints]
        self.get_logger().info(f"  当前 {group_name}: {_fmt_named_joints(joints, cur)}")

    def _pose_in_base(self, pose: PoseStamped) -> PoseStamped:
        src = (pose.header.frame_id or self.base_frame).lstrip("/")
        dst = self.base_frame.lstrip("/")
        out = PoseStamped()
        out.header.frame_id = dst
        if not src or src == dst:
            out.pose = pose.pose
            return out
        try:
            tf = self._tf_buffer.lookup_transform(dst, src, rclpy.time.Time())
        except TransformException as exc:
            raise RuntimeError(f"TF {src} -> {dst} 失败: {exc}") from exc
        t, r = tf.transform.translation, tf.transform.rotation
        p, o = pose.pose.position, pose.pose.orientation
        (px, py, pz), (qw, qx, qy, qz) = transform_pose_wxyz(
            (t.x, t.y, t.z),
            (r.w, r.x, r.y, r.z),
            (p.x, p.y, p.z),
            (o.w, o.x, o.y, o.z),
        )
        out.pose.position.x = px
        out.pose.position.y = py
        out.pose.position.z = pz
        out.pose.orientation.w = qw
        out.pose.orientation.x = qx
        out.pose.orientation.y = qy
        out.pose.orientation.z = qz
        return out

    def _log_laps(self, kind: str, group_name: str, timer: _LapTimer) -> None:
        self.get_logger().info(f"{kind} [{group_name}] 分段(ms): {timer.summary()}")

    def _current_curobo_state(self):
        current, missing = joint_map_to_curobo(
            self.planner.joint_names,
            self._joint_map,
            self.planner.device,
            default_position=self.planner.default_joint_position,
        )
        if missing:
            self.get_logger().warn(
                f"joint_states 缺少 {len(missing)} 个主动关节，用默认值: {missing[:6]}"
            )
        return current

    def _finish(self, group_name: str, planned: TrajResult, response, timer: Optional[_LapTimer] = None):
        response.success = planned.success
        note = self._planning_timing_note(timer) if timer is not None else ""
        response.message = f"{planned.message}；{note}" if note else planned.message
        planned.message = response.message
        if not planned.success:
            self.get_logger().error(f"规划失败 [{group_name}]: {planned.message}")
            if timer is not None:
                self._log_laps("TrajOpt", group_name, timer)
            return
        if planned.trajectory is None or planned.trajectory.position is None:
            self.get_logger().error("规划成功但无法生成 JointTrajectory")
            if timer is not None:
                self._log_laps("TrajOpt", group_name, timer)
            return
        n_pts = int(
            planned.trajectory.position.reshape(-1, planned.trajectory.position.shape[-1]).shape[0]
        )
        self.get_logger().info(
            f"规划成功 [{group_name}]: {planned.message}（{n_pts} 点）"
        )
        traj = curobo_traj_to_msg(
            planned.trajectory,
            group_name,
            dt=self.planner.interpolation_dt,
            velocity_scale=self.trajopt_velocity_scale,
        )
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.header.frame_id = self.base_frame
        self._traj_pub.publish(traj)
        published = [self._traj_topic]
        if timer is not None:
            timer.lap("pack")
        from curobo_ros_python.visualization.publish import (
            ee_paths_from_joint_state,
            publish_trajopt,
            trajopt_ee_frames,
        )

        ee_paths = {}
        if self.visualize_trajectory and self.compute_ee_paths:
            frames = trajopt_ee_frames(group_name, self.planner.tool_frames)
            try:
                traj_js = planned.trajectory.clone()
                if traj_js.joint_names and list(traj_js.joint_names) != list(
                    self.planner.joint_names
                ):
                    traj_js = self.planner.planner.kinematics.get_active_js(traj_js)
                ee_paths = ee_paths_from_joint_state(
                    self.planner.planner.compute_kinematics,
                    traj_js,
                    frames,
                )
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f"末端轨迹 FK 失败，RViz 绿线可能没有: {exc}")
        published.extend(publish_trajopt(self, traj, ee_paths))
        self.get_logger().info("已发布 " + "、".join(published))
        if timer is not None:
            timer.lap("preview")
        if not self.execute_trajectory or self.executor_helper is None:
            if not self.execute_trajectory:
                response.message = planned.message + "（未执行，execute_trajectory:=false）"
                self.get_logger().info("未下发控制器（execute_trajectory:=false）")
            if timer is not None:
                self._log_laps("TrajOpt", group_name, timer)
            return
        self.get_logger().info(f"开始执行轨迹 [{group_name}] ...")
        ok, msg = self.executor_helper.execute(
            group_name, traj, timeout_sec=self.execute_timeout
        )
        if timer is not None:
            timer.lap("execute")
            self._log_laps("TrajOpt", group_name, timer)
        response.success = ok
        response.message = msg if not ok else planned.message + "；" + msg
        if ok:
            self.get_logger().info(f"执行成功 [{group_name}]: {response.message}")
        else:
            self.get_logger().error(f"执行失败 [{group_name}]: {msg}")

    def _on_plan_joint(self, request, response):
        err = self._require_group(request.group_name, request.waypoints)
        if err:
            return self._reject(response, err)
        busy = self._busy_mpc()
        if busy:
            return self._reject(response, busy)
        if not self._gpu_lock.acquire(blocking=False):
            return self._reject(response, "规划器正忙")
        try:
            self.get_logger().info(
                f"收到 PlanJointGoal: group={request.group_name}, waypoints={len(request.waypoints)}"
            )
            timer = _LapTimer()
            self._world_timing = {"cloud": 0.0, "scene": 0.0}
            self.planner.last_timing = {}
            joints = GROUP_JOINTS[request.group_name]
            self._log_group_current(request.group_name)
            for i, wp in enumerate(request.waypoints):
                self.get_logger().info(f"  目标路点[{i}]: {_fmt_named_joints(joints, wp.positions)}")
            try:
                current = self._current_curobo_state()
                goals = [
                    group_goal_state(
                        self.planner.joint_names,
                        request.group_name,
                        list(wp.positions),
                        current,
                    )
                    for wp in request.waypoints
                ]
                timer.lap("state")
                self._publish_joint_goal_waypoints(request.group_name, goals)
                pc_err = self._sync_world_from_pointcloud(request.group_name)
                timer.lap("world")
                if pc_err:
                    self._log_laps("TrajOpt", request.group_name, timer)
                    return self._reject(
                        response, f"{pc_err}；{self._planning_timing_note(timer)}", log=True
                    )
                planned = self.planner.plan_joint(
                    group_name=request.group_name,
                    current=current,
                    goals=goals,
                )
                timer.lap("solve")
            except Exception as exc:  # noqa: BLE001
                timer.lap("solve")
                self._log_laps("TrajOpt", request.group_name, timer)
                return self._reject(
                    response,
                    f"规划异常: {exc}；{self._planning_timing_note(timer)}",
                    log=True,
                )
            self._finish(request.group_name, planned, response, timer)
            return response
        finally:
            self._gpu_lock.release()


    def _pose_to_xyz_xyzw(self, pose: PoseStamped) -> Tuple[PoseStamped, list, list, str]:
        src_frame = pose.header.frame_id or self.base_frame
        pose = self._pose_in_base(pose)
        xyz, xyzw = _pose_xyz_xyzw(pose)
        extra = (
            f" (from {src_frame})"
            if src_frame.lstrip("/") != self.base_frame.lstrip("/")
            else ""
        )
        return pose, xyz, xyzw, extra

    def _on_plan_pose(self, request, response):
        err = self._require_group(request.group_name, request.waypoints)
        if err:
            return self._reject(response, err)
        busy = self._busy_mpc()
        if busy:
            return self._reject(response, busy)
        if not self._gpu_lock.acquire(blocking=False):
            return self._reject(response, "规划器正忙")
        try:
            return self._on_plan_pose_locked(request, response)
        finally:
            self._gpu_lock.release()

    def _raw_goal_items(self, request):
        items = [("goal", pose) for pose in request.waypoints]
        other = list(getattr(request, "other_waypoints", []) or [])
        items.extend(("other", pose) for pose in other)
        return items

    def _publish_goal_waypoints(self, items) -> None:
        from curobo_ros_python.visualization.publish import publish_goal_waypoints

        if publish_goal_waypoints(self, items):
            self.get_logger().info(f"目标路点已发布 /curobo/goal_waypoints（{len(items)} 个）")

    def _publish_joint_goal_waypoints(self, group_name: str, goals) -> None:
        from curobo_ros_python.visualization.publish import joint_goal_marker_items

        try:
            items = joint_goal_marker_items(self, group_name, goals)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"关节路点 FK 失败，RViz 不显示目标: {exc}")
            return
        self._publish_goal_waypoints(items)

    def _build_pose_waypoints(self, request, other, other_ee):
        """返回 (规划路点, 可视化位姿, 错误)。可视化位姿在变到 base 之后；校验失败则用原始位姿。"""
        if other:
            if request.group_name != "both_arms":
                return None, self._raw_goal_items(request), "other_waypoints 只用于 group_name=both_arms"
            if len(other) != len(request.waypoints):
                return (
                    None,
                    self._raw_goal_items(request),
                    f"waypoints 与 other_waypoints 数量须相同，"
                    f"收到 {len(request.waypoints)} 和 {len(other)}",
                )
            left_ee, right_ee = both_arm_ee_links()
            ee = (request.ee_link or "").strip() or left_ee
            other_ee = other_ee or right_ee
            if ee == other_ee:
                return None, self._raw_goal_items(request), "ee_link 与 other_ee_link 不能相同"
            waypoints = []
            viz = []
            for i, (pose, pose_other) in enumerate(zip(request.waypoints, other)):
                pose, xyz, xyzw, extra = self._pose_to_xyz_xyzw(pose)
                pose_other, xyz2, xyzw2, extra2 = self._pose_to_xyz_xyzw(pose_other)
                self.get_logger().info(
                    f"  路点[{i}] {ee}: {_fmt_pose(xyz, xyzw, pose.header.frame_id)}{extra}"
                )
                self.get_logger().info(
                    f"  路点[{i}] {other_ee}: {_fmt_pose(xyz2, xyzw2, pose_other.header.frame_id)}{extra2}"
                )
                viz.append((ee, pose))
                viz.append((other_ee, pose_other))
                waypoints.append(
                    {
                        ee: stamped_to_pose(pose, self.planner.device_cfg),
                        other_ee: stamped_to_pose(pose_other, self.planner.device_cfg),
                    }
                )
            return waypoints, viz, None
        ee, ee_err = resolve_ee_link(request.group_name, request.ee_link)
        if ee_err:
            return None, self._raw_goal_items(request), ee_err
        waypoints = []
        viz = []
        for i, pose in enumerate(request.waypoints):
            pose, xyz, xyzw, extra = self._pose_to_xyz_xyzw(pose)
            self.get_logger().info(
                f"  目标路点[{i}]: {_fmt_pose(xyz, xyzw, pose.header.frame_id)}{extra}"
            )
            viz.append((ee, pose))
            waypoints.append({ee: stamped_to_pose(pose, self.planner.device_cfg)})
        return waypoints, viz, None

    def _on_plan_pose_locked(self, request, response):
        other = list(getattr(request, "other_waypoints", []) or [])
        other_ee = (getattr(request, "other_ee_link", "") or "").strip()
        self.get_logger().info(
            f"收到 PlanPoseGoal: group={request.group_name}, waypoints={len(request.waypoints)}, "
            f"ee_link={request.ee_link or '(默认)'}"
            + (f", other_waypoints={len(other)}" if other else "")
        )
        self._log_group_current(request.group_name)
        timer = _LapTimer()
        self._world_timing = {"cloud": 0.0, "scene": 0.0}
        self.planner.last_timing = {}
        viz_published = False
        try:
            waypoints, viz_items, err = self._build_pose_waypoints(request, other, other_ee)
            if viz_items:
                self._publish_goal_waypoints(viz_items)
                viz_published = True
            timer.lap("convert")
            if err:
                self._log_laps("TrajOpt", request.group_name, timer)
                return self._reject(response, err, log=True)
            pc_err = self._sync_world_from_pointcloud(request.group_name)
            timer.lap("world")
            if pc_err:
                self._log_laps("TrajOpt", request.group_name, timer)
                return self._reject(
                    response, f"{pc_err}；{self._planning_timing_note(timer)}", log=True
                )
            planned = self.planner.plan_pose(
                group_name=request.group_name,
                current=self._current_curobo_state(),
                waypoints=waypoints,
            )
            timer.lap("solve")
        except Exception as exc:  # noqa: BLE001
            timer.lap("solve")
            if not viz_published:
                self._publish_goal_waypoints(self._raw_goal_items(request))
            self._log_laps("TrajOpt", request.group_name, timer)
            return self._reject(
                response,
                f"规划异常: {exc}；{self._planning_timing_note(timer)}",
                log=True,
            )
        self._finish(request.group_name, planned, response, timer)
        return response

    def _on_compute_ik(self, request, response):
        err = self._require_group(request.group_name)
        if err:
            return self._reject(response, err, angles=True)
        busy = self._busy_mpc()
        if busy:
            return self._reject(response, busy, angles=True)
        if not self._gpu_lock.acquire(blocking=False):
            return self._reject(response, "规划器正忙", angles=True)
        try:
            self.get_logger().info(f"收到 ComputeIK: group={request.group_name}")
            self._log_group_current(request.group_name)
            timer = _LapTimer()
            try:
                pc_err = self._sync_world_from_pointcloud(request.group_name)
                timer.lap("world")
                if pc_err:
                    self._log_laps("IK", request.group_name, timer)
                    return self._reject(response, pc_err, log=True, angles=True)
                pose = self._pose_in_base(request.pose)
                xyz, xyzw = _pose_xyz_xyzw(pose)
                self.get_logger().info(f"  目标位姿: {_fmt_pose(xyz, xyzw, pose.header.frame_id)}")
                current = self._current_curobo_state()
                goal_pose = stamped_to_pose(pose, self.planner.device_cfg)
                timer.lap("convert")
                planned = self.planner.compute_ik(
                    group_name=request.group_name,
                    current=current,
                    pose=goal_pose,
                    ee_link=request.ee_link,
                )
                timer.lap("solve")
            except Exception as exc:  # noqa: BLE001
                timer.lap("solve")
                self._log_laps("IK", request.group_name, timer)
                return self._reject(response, f"IK 异常: {exc}", log=True, angles=True)
            response.success = planned.success
            response.message = planned.message
            response.joint_angles = []
            if planned.success and planned.solution is not None:
                response.joint_angles = ik_group_angles(
                    planned.solution, request.group_name, self._joint_map
                )
            timer.lap("pack")
            self._log_laps("IK", request.group_name, timer)
            return response
        finally:
            self._gpu_lock.release()
