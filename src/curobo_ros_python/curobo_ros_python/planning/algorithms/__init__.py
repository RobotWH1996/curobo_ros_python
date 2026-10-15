"""规划算法后端：TrajOpt 与 MPC 分文件实现。

只吃 / 吐 cuRobo 类型。ROS 消息转换在 ``planning.convert``，RViz 在 ``visualization``。

- ``algorithms.trajopt.CuroboPlanner``：启动时加载，一次规划整条轨迹
- ``algorithms.mpc.DualArmMpc``：首次 ``follow_mpc`` 时加载，周期滚动优化
"""
