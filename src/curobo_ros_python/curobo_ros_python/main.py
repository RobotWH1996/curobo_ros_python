"""cuRobo 运动规划节点入口。"""

from __future__ import annotations

import rclpy
from rclpy.executors import MultiThreadedExecutor

from curobo_ros_python.motion_planning_node import CuroboMotionPlanningNode


def main(args=None):
    rclpy.init(args=args)
    node = CuroboMotionPlanningNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
