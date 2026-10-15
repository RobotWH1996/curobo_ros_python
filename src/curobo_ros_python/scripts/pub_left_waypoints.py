#!/usr/bin/env python3
"""循环发三个左臂目标，每个 2 秒；不退出脚本就一直轮。位姿是 2026-09-17 从 TF 记下的。"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from curobo_ros_interfaces.msg import MpcTarget
from geometry_msgs.msg import PoseStamped

# (hold_sec, xyz, xyzw)
WAYPOINTS = [
    (1.0, (0.242, 0.767, 0.990), (-0.206, 0.676, 0.205, 0.677)),
    (1.0, (0.242, 0.767, 1.282), (-0.356, 0.611, 0.356, 0.611)),
    (5.0, (0.449, 0.469, 1.154), (0.276, 0.651, -0.276, 0.651)),
]
RATE_HZ = 10.0
TOPIC = "/curobo/target"


def make_target(xyz, xyzw) -> MpcTarget:
    msg = MpcTarget()
    msg.has_left = True
    msg.has_right = False
    left = PoseStamped()
    left.header.frame_id = "base_link"
    left.pose.position.x, left.pose.position.y, left.pose.position.z = xyz
    left.pose.orientation.x, left.pose.orientation.y, left.pose.orientation.z, left.pose.orientation.w = xyzw
    msg.left = left
    return msg


def main():
    rclpy.init()
    node = Node("pub_left_waypoints")
    pub = node.create_publisher(
        MpcTarget, TOPIC, QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
    )
    period = 1.0 / RATE_HZ
    cycle = 0
    try:
        while rclpy.ok():
            cycle += 1
            for i, (hold, xyz, xyzw) in enumerate(WAYPOINTS, start=1):
                msg = make_target(xyz, xyzw)
                node.get_logger().info(
                    f"第 {cycle} 轮 路点 {i}/{len(WAYPOINTS)} 发 {hold:.1f}s: xyz={xyz} xyzw={xyzw}"
                )
                started = node.get_clock().now()
                while rclpy.ok():
                    msg.header.stamp = node.get_clock().now().to_msg()
                    msg.left.header.stamp = msg.header.stamp
                    pub.publish(msg)
                    elapsed = (node.get_clock().now() - started).nanoseconds * 1e-9
                    if elapsed >= hold:
                        break
                    rclpy.spin_once(node, timeout_sec=period)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
