#!/usr/bin/env python3
"""持续跟踪 /obj_poses：按 label 选 base_link 下最左目标，发 /curobo/target 并打开 /follow_mpc。"""

import argparse

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros import Buffer, TransformException, TransformListener

from curobo_ros_interfaces.action import FollowMpc
from curobo_ros_interfaces.msg import MpcTarget
from curobo_ros_python.planning.geom import transform_pose_wxyz
from perception_msgs.msg import Pose6D, Pose6DArray

LABEL_NAMES = {0: "BACKGROUND", 5: "BOX", 11: "ROD", 12: "HOLE"}


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="/obj_poses")
    parser.add_argument("--target-topic", default="/curobo/target")
    parser.add_argument("--action", default="/follow_mpc")
    parser.add_argument("--group", default="left_arm", choices=("left_arm", "right_arm"))
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
    parser.add_argument("--timeout", type=float, default=5.0, help="等第一帧和 Action 的秒数")
    parser.add_argument("--rate", type=float, default=10.0, help="重发 /curobo/target 的频率")
    parser.add_argument("--dry-run", action="store_true", help="只打印第一帧目标，不跟踪")
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


def _in_base(node: "Tracker", stamped: PoseStamped) -> PoseStamped:
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


class Tracker(Node):
    def __init__(self, args):
        super().__init__("obj_pose_to_plan")
        self._args = args
        self._msg = None
        self._target = None
        self._goal_handle = None
        self._action_message = None
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(Pose6DArray, args.topic, self._on_msg, qos)
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._pub = self.create_publisher(MpcTarget, args.target_topic, qos)
        self._action = ActionClient(self, FollowMpc, args.action)

    def _on_msg(self, msg: Pose6DArray):
        self._msg = msg

    def wait_first(self) -> int:
        deadline = self.get_clock().now().nanoseconds + int(self._args.timeout * 1e9)
        while rclpy.ok() and self._target is None:
            if self.get_clock().now().nanoseconds > deadline:
                self.get_logger().error(f"{self._args.timeout:.1f}s 内没有可用的 {self._args.topic}")
                return 1
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._msg is not None:
                self._refresh(self._msg, announce=True)
        return 0 if self._target is not None else 1

    def _refresh(self, msg: Pose6DArray, announce: bool = False) -> None:
        poses = _matched(msg, self._args.label)
        if not poses:
            if self._target is not None:
                self.get_logger().warn(
                    f"label={self._args.label} 本帧没有目标，沿用上一目标",
                    throttle_duration_sec=2.0,
                )
            return
        bases = [_in_base(self, _to_stamped(msg, pose)) for pose in poses]
        chosen_i = max(range(len(bases)), key=lambda i: bases[i].pose.position.y)
        track_id = int(poses[chosen_i].track_id)
        self._target = _apply_offset(bases[chosen_i], self._args.offset)
        if announce:
            name = LABEL_NAMES.get(self._args.label, "")
            dx, dy, dz = self._args.offset
            self.get_logger().info(
                f"最左 track={track_id} label={self._args.label} {name} "
                f"offset=({dx:.3f}, {dy:.3f}, {dz:.3f}) {_pose_line(self._target)}"
            )
        else:
            self.get_logger().info(
                f"最左 track={track_id} {_pose_line(self._target)}",
                throttle_duration_sec=1.0,
            )

    def _mpc_target(self) -> MpcTarget:
        msg = MpcTarget()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.has_left = self._args.group == "left_arm"
        msg.has_right = self._args.group == "right_arm"
        stamped = PoseStamped()
        stamped.header.stamp = msg.header.stamp
        stamped.header.frame_id = self._target.header.frame_id
        stamped.pose = self._target.pose
        if msg.has_left:
            msg.left = stamped
        else:
            msg.right = stamped
        return msg

    def publish_latest(self) -> None:
        if self._msg is not None:
            self._refresh(self._msg)
            self._msg = None
        if self._target is None:
            return
        self._pub.publish(self._mpc_target())

    def start_action(self) -> int:
        if not self._action.wait_for_server(timeout_sec=self._args.timeout):
            self.get_logger().error(f"Action {self._args.action} 不可用")
            return 1
        goal = FollowMpc.Goal()
        goal.timeout = 0.0
        goal.position_tolerance = 0.0
        goal.orientation_tolerance = 0.0
        send_future = self._action.send_goal_async(goal, feedback_callback=self._on_feedback)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=self._args.timeout)
        if not send_future.done() or send_future.result() is None:
            self.get_logger().error("发送 /follow_mpc 超时")
            return 1
        self._goal_handle = send_future.result()
        if not self._goal_handle.accepted:
            self.get_logger().error("follow_mpc 被拒绝。已有跟踪在跑时请先取消")
            return 1
        self._goal_handle.get_result_async().add_done_callback(self._on_result)
        self.get_logger().info(f"{self._args.action} 已开始，Ctrl-C 停止")
        return 0

    def _on_feedback(self, msg):
        fb = msg.feedback
        if self._args.group == "left_arm":
            pos, ori = fb.left_position_error, fb.left_orientation_error
        else:
            pos, ori = fb.right_position_error, fb.right_orientation_error
        self.get_logger().info(
            f"跟踪误差 pos={pos:.4f} m ori={ori:.4f} rad solve={fb.solve_time:.3f}s "
            f"plan_success={fb.plan_success} {fb.message}",
            throttle_duration_sec=2.0,
        )

    def _on_result(self, future):
        wrapped = future.result()
        result = getattr(wrapped, "result", None)
        message = getattr(result, "message", "") or "follow_mpc 结束"
        self._action_message = message
        self.get_logger().info(message)

    def cancel_action(self) -> None:
        handle = self._goal_handle
        if handle is None:
            return
        future = handle.cancel_goal_async()
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)


def main():
    args = _parse_args()
    rclpy.init()
    node = Tracker(args)
    code = node.wait_first()
    if code != 0 or args.dry_run:
        if code == 0:
            node.get_logger().info(f"dry-run {_pose_line(node._target)}")
        node.destroy_node()
        rclpy.shutdown()
        return code
    code = node.start_action()
    if code != 0:
        node.destroy_node()
        rclpy.shutdown()
        return code
    period = 1.0 / max(float(args.rate), 1e-3)
    try:
        while rclpy.ok() and node._action_message is None:
            node.publish_latest()
            rclpy.spin_once(node, timeout_sec=period)
    except KeyboardInterrupt:
        node.get_logger().info("停止跟踪")
    finally:
        node.cancel_action()
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
