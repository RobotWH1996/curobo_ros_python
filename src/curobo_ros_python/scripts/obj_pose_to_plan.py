#!/usr/bin/env python3
"""读一帧 /obj_poses，按 label 过滤后，把 base_link 下最左边的目标发给 /plan_pose_goal。"""

import argparse

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros import Buffer, TransformException, TransformListener

from curobo_ros_interfaces.srv import PlanPoseGoal
from curobo_ros_python.planning.geom import transform_pose_wxyz
from perception_msgs.msg import Pose6D, Pose6DArray

LABEL_NAMES = {0: "BACKGROUND", 5: "BOX", 11: "ROD", 12: "HOLE"}


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="/obj_poses")
    parser.add_argument("--service", default="/plan_pose_goal")
    parser.add_argument("--group", default="left_arm")
    parser.add_argument("--ee-link", default="", help="空则由规划节点按 group 选默认末端")
    parser.add_argument("--base-frame", default="base_link", help="用来比较左右的机器人坐标系")
    parser.add_argument(
        "--offset",
        type=float,
        nargs=3,
        metavar=("DX", "DY", "DZ"),
        default=[-0.4, 0.0, 0.10],
        help="选中目标在 base_link 下的位置偏移（米），加到 xyz 后再下发",
    )
    parser.add_argument(
        "--label",
        "--class-id",
        type=int,
        default=5,
        help="处理该类别，默认 5=BOX；11=ROD，12=HOLE",
    )
    parser.add_argument("--timeout", type=float, default=5.0, help="等话题的秒数")
    parser.add_argument("--dry-run", action="store_true", help="只打印服务请求，不调用")
    return parser.parse_args()


def _matched(msg: Pose6DArray, label):
    poses = [pose for pose in msg.poses if pose.valid == 1 and pose.label == label]
    if not poses:
        name = LABEL_NAMES.get(label, "")
        raise RuntimeError(f"这一帧没有有效的 label={label} {name}")
    return poses


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


def _in_base(node: "Once", stamped: PoseStamped) -> PoseStamped:
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


def _leftmost(node: "Once", msg: Pose6DArray, poses):
    bases = [_in_base(node, _to_stamped(msg, pose)) for pose in poses]
    chosen_i = max(range(len(bases)), key=lambda i: bases[i].pose.position.y)
    return bases, bases[chosen_i]


def _describe(poses, bases, chosen) -> str:
    lines = []
    for i, (pose, base) in enumerate(zip(poses, bases)):
        name = LABEL_NAMES.get(pose.label, "")
        mark = "  <- 最左" if base is chosen else ""
        b = base.pose.position
        lines.append(
            f"  [{i}] label={pose.label} {name} track={pose.track_id} score={pose.score:.3f} "
            f"msg_xyz=({pose.x:.3f}, {pose.y:.3f}, {pose.z:.3f}) "
            f"base_xyz=({b.x:.3f}, {b.y:.3f}, {b.z:.3f}){mark}"
        )
    return "\n".join(lines)


class Once(Node):
    def __init__(self, args):
        super().__init__("obj_pose_to_plan")
        self._args = args
        self._msg = None
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(Pose6DArray, args.topic, self._on_msg, qos)
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._client = self.create_client(PlanPoseGoal, args.service)

    def _on_msg(self, msg: Pose6DArray):
        if self._msg is None:
            self._msg = msg


def main():
    args = _parse_args()
    rclpy.init()
    node = Once(args)
    deadline = node.get_clock().now().nanoseconds + int(args.timeout * 1e9)
    while rclpy.ok() and node._msg is None:
        if node.get_clock().now().nanoseconds > deadline:
            node.get_logger().error(f"{args.timeout:.1f}s 内没有收到 {args.topic}")
            node.destroy_node()
            rclpy.shutdown()
            return 1
        rclpy.spin_once(node, timeout_sec=0.1)

    msg = node._msg
    try:
        poses = _matched(msg, args.label)
        bases, stamped = _leftmost(node, msg, poses)
    except RuntimeError as exc:
        node.get_logger().error(str(exc))
        node.destroy_node()
        rclpy.shutdown()
        return 1

    chosen_i = bases.index(stamped)
    raw = stamped
    stamped = _apply_offset(raw, args.offset)
    frame = stamped.header.frame_id or "(empty)"
    src = (msg.header.frame_id or "(empty)").lstrip("/")
    dx, dy, dz = args.offset
    node.get_logger().info(
        f"{args.topic} {src} -> {frame}，label={args.label} 共 {len(poses)} 个，"
        f"选用最左 track={poses[chosen_i].track_id} "
        f"base_y={raw.pose.position.y:.3f} "
        f"offset=({dx:.3f}, {dy:.3f}, {dz:.3f})\n{_describe(poses, bases, raw)}"
    )

    request = PlanPoseGoal.Request()
    request.group_name = args.group
    request.ee_link = args.ee_link
    request.waypoints = [stamped]

    p = stamped.pose.position
    q = stamped.pose.orientation
    node.get_logger().info(
        "PlanPoseGoal:\n"
        f"  group_name: {request.group_name}\n"
        f"  ee_link: {request.ee_link or '(默认)'}\n"
        f"  waypoints[0] frame={frame} "
        f"xyz=({p.x:.4f}, {p.y:.4f}, {p.z:.4f}) "
        f"xyzw=({q.x:.4f}, {q.y:.4f}, {q.z:.4f}, {q.w:.4f})"
    )
    if args.dry_run:
        node.destroy_node()
        rclpy.shutdown()
        return 0

    if not node._client.wait_for_service(timeout_sec=args.timeout):
        node.get_logger().error(f"服务 {args.service} 不可用")
        node.destroy_node()
        rclpy.shutdown()
        return 1
    future = node._client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=120.0)
    if not future.done() or future.result() is None:
        node.get_logger().error("调用规划服务超时或失败")
        code = 1
    else:
        result = future.result()
        node.get_logger().info(f"success={result.success} message={result.message}")
        code = 0 if result.success else 1
    node.destroy_node()
    rclpy.shutdown()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
