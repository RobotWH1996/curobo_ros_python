#!/usr/bin/env python3
"""按固定路点循环，发 /curobo/target 并打开 /follow_mpc。

不订 /obj_poses。点云关掉时用来测空场景跟踪。路点在 base_link，姿态共用。
"""

import argparse

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from curobo_ros_interfaces.action import FollowMpc
from curobo_ros_interfaces.msg import MpcTarget

# 上次空旷处能收到 1–2 cm 的末端姿态，xyzw。
ORIENTATION_XYZW = (0.045, -0.020, -0.574, 0.817)

# base_link，米。中心是上次能收到的点，其余沿同一串台阶。
FIXED_POINTS = (
    (0.545, 0.230, 1.182),
    (0.695, 0.230, 1.182),
    (0.695, 0.350, 1.232),
    (0.545, 0.350, 1.282),
    (0.445, 0.230, 1.232),
    (0.545, 0.130, 1.182),
)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-topic", default="/curobo/target")
    parser.add_argument("--action", default="/follow_mpc")
    parser.add_argument("--group", default="left_arm", choices=("left_arm", "right_arm"))
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--timeout", type=float, default=5.0, help="等 Action 的秒数")
    parser.add_argument("--rate", type=float, default=10.0, help="重发 /curobo/target 的频率")
    parser.add_argument("--waypoint-period", type=float, default=3.0, help="每个固定路点停留的秒数")
    parser.add_argument("--dry-run", action="store_true", help="只打印第一点，不跟踪")
    return parser.parse_args()


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
        self._index = 0
        self._next_ns = 0
        self._target = self._make_pose(0)
        self._goal_handle = None
        self._action_message = None
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._pub = self.create_publisher(MpcTarget, args.target_topic, qos)
        self._action = ActionClient(self, FollowMpc, args.action)

    def _make_pose(self, index: int) -> PoseStamped:
        x, y, z = FIXED_POINTS[index]
        qx, qy, qz, qw = ORIENTATION_XYZW
        stamped = PoseStamped()
        stamped.header.frame_id = self._args.base_frame
        stamped.pose.position.x = float(x)
        stamped.pose.position.y = float(y)
        stamped.pose.position.z = float(z)
        stamped.pose.orientation.x = float(qx)
        stamped.pose.orientation.y = float(qy)
        stamped.pose.orientation.z = float(qz)
        stamped.pose.orientation.w = float(qw)
        return stamped

    def _advance(self) -> None:
        now = self.get_clock().now().nanoseconds
        if self._next_ns != 0 and now < self._next_ns:
            return
        if self._next_ns != 0:
            self._index = (self._index + 1) % len(FIXED_POINTS)
        self._target = self._make_pose(self._index)
        period = max(float(self._args.waypoint_period), 0.1)
        self._next_ns = now + int(period * 1e9)
        self.get_logger().info(
            f"路点 {self._index + 1}/{len(FIXED_POINTS)} {_pose_line(self._target)}"
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
        self._advance()
        self._target.header.stamp = self.get_clock().now().to_msg()
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
    if args.dry_run:
        node.get_logger().info(f"dry-run {_pose_line(node._target)}")
        node.destroy_node()
        rclpy.shutdown()
        return 0
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
