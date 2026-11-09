# OPT Pick / Place / PickPlace

[工程 README](../README.md#opt-抓取与放置pick--place--pickplace) 包含当前参数、碰撞规则和调用方式。本页说明分块执行边界。

## 输入与规划状态

- `/pick` 使用 Pick.Goal.task；`/place` 使用 Place.Goal.task；`/pick_place` 使用 PickPlace.Goal.pick/place。
- 是否下发沿用节点 execute_trajectory：debug 默认 true，deploy 默认 false，请求不再包含 execute。timeout 限制本次规划及提交，不等待返回后的运动完成。
- left_arm/right_arm 仍为 7 自由度 OPT；夹爪不加入手臂优化组。原关节角/笛卡尔多目标 Service、双臂规划、MPC 保留。
- 本体和左右夹爪的初始实际状态来自 `/joint_states`，后续段使用计划终点继续，不向该话题发布命令。
- 工件几何、混合路点、end_pose、全开碰撞假设及 keep_tool_z_vertical 等参数含义保持原有规则。

## 分块与连续通过

| 接口 | 首块 | 后续块 |
| --- | --- | --- |
| Pick | 起点/路点 → prepick → pick → 闭合及等待 | lift → 可选 end |
| Place | 起点/路点 → preplace → place → 张开及等待 | retreat → 可选 end |
| PickPlace | 起点/路点 → prepick → pick → 闭合及等待 | lift → 可选 Pick end → 携物路点 → preplace → place → 张开及等待；最后 retreat → 可选 Place end |

首块完成合并验收即可提交，后续规划与运动重叠。每块内部保留原时间轴、局部平滑、动态限位、GPU/FCL 分阶段验收以及接触段 FK 检查。预抓取、提升、预放置、撤退和中间 end 不作为停顿边界。关键边界通过速度为零时拒绝提交，不能用停顿掩盖衔接失败。

抓取前尽快张开夹爪，与接近运动重叠；若接近路程太短，则在起点补足张开时间，避免 prepick 停顿。闭合或释放的自动插补及额外 gripper_wait_time（默认 1 秒）都写入轨迹，不轮询设备反馈。

本体滤除继续按运动臂执行，off/observe/apply 规则不变。启用碰撞检查的区间仍使用运动相关配对，保留运动臂与另一臂/躯干的自碰撞检查。接触区间沿用现有免碰撞策略，未增加豁免范围。携物附着、提升后启用工件碰撞、释放后静态工件检查继续按阶段切换。

## 控制器与话题

- debug 使用标准 FollowJointTrajectory，手臂和同侧夹爪独立控制器共用起跑时间（首块留 0.1 秒提交提前量）。后续版追加到累计前缀，不能修改已经提交的点或时间。
- 两个控制器都确认接收后才提交下一版；适配器处理目标拒绝、确认超时、迟到确认和取消。新版目标替换旧版产生的正常取消不算规划失败。
- 后续规划/确认若超过已提交前缀的续接窗口，会失败并取消。后段提前完成时不在自然停顿处增加控制器往返等待。
- 规划失败或取消会取消已提交目标；此时前缀可能已经执行。执行未结束时拒绝新任务。
- deploy 不调用控制器。两种模式均在 `/curobo/planned_trajectory` 发布累计已验收前缀，各消息共用 header.stamp 和全任务 time_from_start。实机适配器应续接同一时间轴，不能逐条从头重放。

规划侧不依赖设备消息或 SDK。仿真由 ros2_control/joint_state_broadcaster 统一输出本体及夹爪实际状态；实机由既有 SDK 转接脚本统一输出 `/joint_states`。RViz 可选，只显示反馈。

## 结果与日志

Result.trajectory 是最终完整轨迹，包含模型所有非固定、非 mimic 独立关节及左右夹爪；读取顺序以 joint_names 为准。Result.trajectories 与 trajectory_stages 为局部时间的分段结果。失败时可能保留已提交前缀，必须检查 success。

success 表示规划及提交阶段成功，不表示实际夹到物体或运动完成。debug 的 execution_requested=true、execution_completed=false；deploy 两者均 false。规划结果返回后，控制器适配器仍跟踪实际执行终态。

`event=轨迹块提交` 记录 block、累计阶段、block_ready_ms、ready_ms 和累计运动时长。任务结束日志记录 blocks、first_ready_ms、first_submit_ms、elapsed_ms。first_submit_ms 是首次提交调用返回的时刻，不是机器人已到位；起跑还受预留提前量和轨迹自身起步加速影响。

2026-11-06 已在独立 ROS domain 197 用保存点云、label 5 目标和匹配关节/TF 快照验证三个 Action，连接真实 ros2_control mock 控制器核对时间轴、接缝和连续通过速度；未向用户运行域下发运动。部署仓库不保留测试及结果文件。
