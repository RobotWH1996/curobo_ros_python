#!/usr/bin/env python3
"""从 /obj_poses 选一个目标，发一次 /pick。

选目标和 obj_pose_to_plan.py 相同：label 过滤、变到 base_link、取最左（+Y 最大）。
检测位姿写入 tool_pose。--object-in-tool 反推成相对抓取位姿，接近和抬升也写成相对 tool_pose 的位姿。

跟踪脚本的偏移 (-0.4, 0, 0.10) 是悬停点，这里不使用。

返回已拼接的本体+夹爪轨迹，可用 --output 保存 JSON。
规划节点按 debug/deploy 启动配置决定是否执行；本脚本只提交 Action 目标。
"""

import argparse
import json
import math

import rclpy
from geometry_msgs.msg import Pose, PoseStamped
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from tf2_ros import Buffer, TransformException, TransformListener

from curobo_ros_interfaces.action import Pick
from curobo_ros_python.planning.geom import rotate_vec_wxyz, transform_pose_wxyz
from perception_msgs.msg import Pose6D, Pose6DArray
from pick_place_client import save_result

LABEL_NAMES = {0: "BACKGROUND", 5: "BOX", 11: "ROD", 12: "HOLE"}


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="/obj_poses")
    parser.add_argument("--action", default="/pick")
    parser.add_argument("--group", default="left_arm", choices=("left_arm", "right_arm"))
    parser.add_argument("--base-frame", default="base_link", help="用来比较左右的机器人坐标系")
    parser.add_argument(
        "--label",
        "--class-id",
        type=int,
        default=5,
        help="处理该类别，默认 5=BOX；11=ROD，12=HOLE",
    )
    parser.add_argument(
        "--offset",
        type=float,
        nargs=3,
        metavar=("DX", "DY", "DZ"),
        default=[0.0, 0.0, 0.0],
        help="加到检测位姿上的 base_link 平移（米）。默认不偏移",
    )
    parser.add_argument(
        "--object-in-tool",
        type=float,
        nargs=7,
        metavar=("X", "Y", "Z", "QX", "QY", "QZ", "QW"),
        default=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        help="工件中心相对工具的位姿，米和 xyzw。Lycus 工具系已在夹指中点；工件中心重合时用 0 0 0 0 0 0 1",
    )
    parser.add_argument(
        "--object-size",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=[0.00, 0.00, 0.00],
        help="工件包围盒（米）。默认全零读取规划节点默认尺寸；可传三个实测正值",
    )
    parser.add_argument("--approach-axis", default="z", choices=("x", "y", "z"))
    parser.add_argument("--approach-offset", type=float, default=-0.10, help="prepick 相对抓取点的位移（米）")
    parser.add_argument(
        "--approach-in-tool-frame",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="接近轴是否在工具坐标系",
    )
    parser.add_argument("--lift-distance", type=float, default=0.10, help="抬升距离，沿 base +Z（米）")
    parser.add_argument("--open-position", type=float, default=0.0, help="URDF 夹爪关节位置，Lycus 0m 全开")
    parser.add_argument("--close-position", type=float, default=-0.042, help="URDF 夹爪关节位置，Lycus -0.042m 全闭")
    parser.add_argument("--gripper-duration", type=float, default=1.0, help="每次开/闭合插补时长，秒")
    parser.add_argument("--timeout", type=float, default=5.0, help="等第一帧和 Action 服务器的秒数")
    parser.add_argument("--task-timeout", type=float, default=180.0, help="抓取任务总超时（秒）")
    parser.add_argument("--output", help="保存完整本体+夹爪轨迹及阶段元数据 JSON")
    parser.add_argument("--dry-run", action="store_true", help="只打印选中的抓取目标，不调用 /pick")
    return parser.parse_args()


def _matched(msg: Pose6DArray, label):
    return [pose for pose in msg.poses if pose.valid == 1 and pose.label == label]


def _to_stamped(msg: Pose6DArray, pose: Pose6D) -> PoseStamped:
    stamped = PoseStamped()
    stamped.header = msg.header
    stamped.pose.position.x = float(pose.x)
    stamped.pose.position.y = float(pose.y)
    stamped.pose.position.z = float(pose.z)
    stamped.pose.orientation.x = float(pose.qx)
    stamped.pose.orientation.y = float(pose.qy)
    stamped.pose.orientation.z = float(pose.qz)
    stamped.pose.orientation.w = float(pose.qw)
    return stamped


def _in_base(node: "Picker", stamped: PoseStamped) -> PoseStamped:
    """变到 base_link。该系 +Y 为机器人左侧。"""
    src = (stamped.header.frame_id or node._args.base_frame).lstrip("/")
    dst = node._args.base_frame.lstrip("/")
    out = PoseStamped()
    out.header.frame_id = dst
    out.header.stamp = stamped.header.stamp
    if not src or src == dst:
        out.pose = stamped.pose
        return out
    deadline = node.get_clock().now().nanoseconds + int(node._args.timeout * 1e9)
    last_exc = None
    tf = None
    while rclpy.ok() and node.get_clock().now().nanoseconds <= deadline:
        try:
            tf = node._tf_buffer.lookup_transform(dst, src, rclpy.time.Time())
            break
        except TransformException as exc:
            last_exc = exc
            rclpy.spin_once(node, timeout_sec=0.05)
    if tf is None:
        raise RuntimeError(f"TF {src} -> {dst} 失败: {last_exc}")
    t, r = tf.transform.translation, tf.transform.rotation
    p, o = stamped.pose.position, stamped.pose.orientation
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


def _apply_offset(stamped: PoseStamped, offset) -> PoseStamped:
    dx, dy, dz = offset
    out = PoseStamped()
    out.header = stamped.header
    p = stamped.pose.position
    q = stamped.pose.orientation
    out.pose.position.x = float(p.x) + float(dx)
    out.pose.position.y = float(p.y) + float(dy)
    out.pose.position.z = float(p.z) + float(dz)
    out.pose.orientation.x = float(q.x)
    out.pose.orientation.y = float(q.y)
    out.pose.orientation.z = float(q.z)
    out.pose.orientation.w = float(q.w)
    return out


def _pose_line(stamped: PoseStamped) -> str:
    p = stamped.pose.position
    q = stamped.pose.orientation
    frame = stamped.header.frame_id or "(empty)"
    return (
        f"frame={frame} xyz=({p.x:.4f}, {p.y:.4f}, {p.z:.4f}) "
        f"xyzw=({q.x:.4f}, {q.y:.4f}, {q.z:.4f}, {q.w:.4f})"
    )


def _unit_wxyz(xyzw):
    qx, qy, qz, qw = (float(v) for v in xyzw)
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm < 1e-8:
        raise ValueError("四元数为零")
    return (qw / norm, qx / norm, qy / norm, qz / norm)


def _tool_pose(object_pose: PoseStamped, object_in_tool) -> PoseStamped:
    """T_base_tool = T_base_object * inv(T_tool_object)。"""
    x, y, z, qx, qy, qz, qw = object_in_tool
    in_tool_q = _unit_wxyz((qx, qy, qz, qw))
    inv_q = (in_tool_q[0], -in_tool_q[1], -in_tool_q[2], -in_tool_q[3])
    inv_t = rotate_vec_wxyz(inv_q, (-float(x), -float(y), -float(z)))
    src = object_pose.pose
    src_q = _unit_wxyz((src.orientation.x, src.orientation.y, src.orientation.z, src.orientation.w))
    tool_t, tool_q = transform_pose_wxyz(
        (src.position.x, src.position.y, src.position.z),
        src_q,
        inv_t,
        inv_q,
    )
    out = PoseStamped()
    out.header = object_pose.header
    out.pose.position.x, out.pose.position.y, out.pose.position.z = tool_t
    out.pose.orientation.w = tool_q[0]
    out.pose.orientation.x = tool_q[1]
    out.pose.orientation.y = tool_q[2]
    out.pose.orientation.z = tool_q[3]
    return out


def _invert_xyzw(values):
    """Invert an xyzw pose. object-in-tool becomes the tool pose relative to the object."""
    x, y, z, qx, qy, qz, qw = values
    rotation = _unit_wxyz((qx, qy, qz, qw))
    inverse = (rotation[0], -rotation[1], -rotation[2], -rotation[3])
    translation = rotate_vec_wxyz(inverse, (-float(x), -float(y), -float(z)))
    return (*translation, inverse[1], inverse[2], inverse[3], inverse[0])


def _shift_relative(pick_xyzw, axis, distance, in_tool, reference_xyzw):
    """Move the grasp pose along one axis, returning another pose relative to tool_pose."""
    px, py, pz, pqx, pqy, pqz, pqw = pick_xyzw
    delta = [0.0, 0.0, 0.0]
    delta["xyz".index(axis)] = float(distance)
    if in_tool:
        rotation = _unit_wxyz((pqx, pqy, pqz, pqw))
    else:
        rx, ry, rz, rw = reference_xyzw
        reference = _unit_wxyz((rx, ry, rz, rw))
        rotation = (reference[0], -reference[1], -reference[2], -reference[3])
    shift = rotate_vec_wxyz(rotation, tuple(delta))
    return (px + shift[0], py + shift[1], pz + shift[2], pqx, pqy, pqz, pqw)


def _fill_pose(target: Pose, values) -> None:
    x, y, z, qx, qy, qz, qw = values
    target.position.x = float(x)
    target.position.y = float(y)
    target.position.z = float(z)
    target.orientation.x = float(qx)
    target.orientation.y = float(qy)
    target.orientation.z = float(qz)
    target.orientation.w = float(qw)


class Picker(Node):
    def __init__(self, args):
        super().__init__("obj_pose_to_pick")
        self._args = args
        self._msg = None
        self._object_pose = None
        self._tool_pose = None
        self._track_id = None
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(Pose6DArray, args.topic, self._on_msg, qos)
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._action = ActionClient(self, Pick, args.action)

    def _on_msg(self, msg: Pose6DArray):
        self._msg = msg

    def wait_selection(self) -> int:
        deadline = self.get_clock().now().nanoseconds + int(self._args.timeout * 1e9)
        while rclpy.ok() and self._tool_pose is None:
            if self.get_clock().now().nanoseconds > deadline:
                self.get_logger().error(f"{self._args.timeout:.1f}s 内没有可用的 {self._args.topic}")
                return 1
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._msg is not None:
                self._select(self._msg)
                self._msg = None
        return 0 if self._tool_pose is not None else 1

    def _select(self, msg: Pose6DArray) -> None:
        poses = _matched(msg, self._args.label)
        if not poses:
            return
        bases = [_apply_offset(_in_base(self, _to_stamped(msg, pose)), self._args.offset) for pose in poses]
        chosen_i = max(range(len(bases)), key=lambda i: bases[i].pose.position.y)
        self._track_id = int(poses[chosen_i].track_id)
        self._object_pose = bases[chosen_i]
        self._tool_pose = _tool_pose(self._object_pose, self._args.object_in_tool)
        name = LABEL_NAMES.get(self._args.label, "")
        self.get_logger().info(
            f"最左 track={self._track_id} label={self._args.label} {name} "
            f"工件 {_pose_line(self._object_pose)}"
        )
        self.get_logger().info(f"抓取位姿 {_pose_line(self._tool_pose)}")

    def _goal(self) -> Pick.Goal:
        args = self._args
        size = tuple(float(v) for v in args.object_size)
        if any(not math.isfinite(v) for v in size) or (any(size) and any(v <= 0.0 for v in size)):
            raise ValueError("object_size 必须全零（读节点默认）或三个有限正数")
        goal = Pick.Goal()
        goal.timeout = float(args.task_timeout)
        task = goal.task
        reference = self._object_pose.pose.orientation
        reference_xyzw = (reference.x, reference.y, reference.z, reference.w)
        pick_xyzw = _invert_xyzw(args.object_in_tool)
        task.group_name = args.group
        task.tool_pose = self._object_pose
        _fill_pose(task.pick_pose, pick_xyzw)
        _fill_pose(task.prepick_in_target, _shift_relative(
            pick_xyzw, args.approach_axis, args.approach_offset, args.approach_in_tool_frame, reference_xyzw))
        _fill_pose(task.lift_in_target, _shift_relative(
            pick_xyzw, args.approach_axis, args.lift_distance, args.approach_in_tool_frame, reference_xyzw))
        task.object_size.x, task.object_size.y, task.object_size.z = size
        task.open_position = float(args.open_position)
        task.close_position = float(args.close_position)
        task.gripper_duration = float(args.gripper_duration)
        return goal

    def send(self) -> int:
        goal = self._goal()
        size = goal.task.object_size
        self.get_logger().info(
            f"size=({size.x:.3f}, {size.y:.3f}, {size.z:.3f}) "
            "执行方式由规划节点的 debug/deploy 启动配置决定"
        )
        if not self._action.wait_for_server(timeout_sec=self._args.timeout):
            self.get_logger().error(f"Action {self._args.action} 不可用")
            return 1
        send_future = self._action.send_goal_async(goal, feedback_callback=self._on_feedback)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=self._args.timeout)
        if not send_future.done() or send_future.result() is None:
            self.get_logger().error("发送 /pick 超时")
            return 1
        handle = send_future.result()
        if not handle.accepted:
            self.get_logger().error("/pick 被拒绝。已有抓放任务、MPC 或参数不合法时请先处理")
            return 1
        result_future = handle.get_result_async()
        try:
            rclpy.spin_until_future_complete(self, result_future)
        except KeyboardInterrupt:
            self.get_logger().info("取消本次 /pick 任务")
            cancel = handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self, cancel, timeout_sec=5.0)
            return 1
        value = result_future.result().result
        if self._args.output:
            save_result(value, "pick", self._args.output)
        print(
            json.dumps(
                dict(
                    success=value.success,
                    stage=value.stage,
                    message=value.message,
                    joint_names=list(value.trajectory.joint_names),
                    points=len(value.trajectory.points),
                    output=self._args.output,
                    trajectories=len(value.trajectories),
                    trajectory_stages=list(value.trajectory_stages),
                ),
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0 if value.success else 1

    def _on_feedback(self, msg):
        self.get_logger().info(f"{msg.feedback.stage} {msg.feedback.message}")


def main():
    args = _parse_args()
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = Picker(args)
    code = 1
    try:
        code = node.wait_selection()
        if code == 0 and args.dry_run:
            node.get_logger().info(f"dry-run 工件 {_pose_line(node._object_pose)}")
            node.get_logger().info(f"dry-run 工具 {_pose_line(node._tool_pose)}")
        elif code == 0:
            code = node.send()
    except KeyboardInterrupt:
        node.get_logger().info("停止")
        code = 1
    except Exception as exc:
        node.get_logger().error(f"{type(exc).__name__}: {exc}")
        code = 1
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
