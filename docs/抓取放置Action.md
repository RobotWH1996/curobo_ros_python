# OPT Pick / Place / PickPlace

[工程 README](../README.md) 包含完整参数与调用方式。抓放 Action 先将所有手臂运动和夹爪插补拼成完整轨迹，再跟随节点启动配置：debug 默认分段下发通用控制器，deploy 只发布和返回轨迹。规划不读取设备专用消息，执行仅等待标准 FollowJointTrajectory 结果。

## 输入、初始状态与模型

- `/pick`：Pick.Goal.task；`/place`：Place.Goal.task；`/pick_place`：PickPlace.Goal.pick/place。
- 请求不再包含 execute；是否下发沿用节点 execute_trajectory（debug=true / deploy=false）。timeout 包含规划及可选执行的整次任务超时。
- left_arm/right_arm 仍为 7 自由度 OPT；原 both_arms、关节角多目标和笛卡尔多目标 Service、MPC 均保留。夹爪不进手臂优化组。
- 本体和左右夹爪的初始实际状态统一来自 joint_states_topic（默认 `/joint_states`）。后续段使用本次计划状态，不覆盖真实反馈，也不向 `/joint_states` 发布计划位置。
- Pick 使用请求的 `object_size`，工件相对工具由 `pick_pose` 反推。独立 Place 在当前工具位姿建立初始携物模型，使用自己的 `object_size` 和 `place_pose`，不依赖上一次 Pick。PickPlace 的 Place 复用本次 Pick 几何。
- 返回前后恢复真实关节几何、临时碰撞开关和场景；不跨请求保存携物假设。

## 轨迹组成

```text
Pick：open → waypoints → prepick → pick → close → attach → lift → enable_payload → end_pose
Place：初始携物 → waypoints → preplace → place → release → detach → retreat → end_pose
```

open/close/release 生成夹爪关节插补，期间手臂保持不动；手臂轨迹段保持当前计划夹爪位置。attach/detach/enable_payload 是规划碰撞状态切换，不需要单独运动轨迹。end_pose 与 waypoints 均支持 type=joint/pose；额外终点为空则停在 lift/retreat。

输出 `Result.trajectory` 为完整时间轴的 JointTrajectory，包含机器人模型所有非固定、非 mimic 独立关节与左右夹爪。未运动关节保持初始值，读取顺序以 joint_names 为准。`Result.trajectories` 与 `trajectory_stages` 是用于检查的分段结果，段内使用局部时间。动作失败时可能保留部分段，外部执行器必须检查 success。

夹爪位置采用 URDF 单位（Lycus prismatic 为米，范围 [-0.042,0]），open_position 默认 0，Pick.close_position 默认 -0.042，gripper_duration 默认每次 1 秒。不存在归一化行程/驱动力字段。该时长定义离线插补，不能代替执行到位或物体夹持确认。

## 碰撞规则

| 区间 | GPU 两片夹指 | GPU 工件 | FCL |
| --- | --- | --- | --- |
| Pick 路点、prepick | 正常 | 尚未启用 | 检查关键目标点 |
| prepick → pick → lift | 忽略 | 提升完成前不启用 | 跳过接触段 |
| lift 后 → Pick end | 恢复 | 随工具运动 | 检查 end |
| Place 携物路点、preplace | 正常 | 正常 | 检查关键目标点 |
| preplace → place → retreat | 忽略 | 忽略，含释放后盒体/对应点云 | 跳过接触段 |
| retreat 后 → Place end | 恢复 | 已放置工件为静态障碍 | 检查 end |

夹爪底座、手腕和其他手臂部件保持 GPU 检查；接触段跳过 FCL IK 候选和输出终点检查。GPU 仍做轨迹避障。FCL 只检查关键点，不提供整条轨迹连续碰撞保证。接近/抬升/撤退的直线与姿态约束保留。

携物点云按已配置包围盒 OBB 去重，不自动估计工件尺寸。机器人本体点云过滤 off/observe/apply 及按规划手臂选择几何的机制保留，与接触豁免各自生效。

## 仿真与实机统一边界

仿真由 Lycus 的 ros2_control 和一个 joint_state_broadcaster 统一发布本体/夹爪实际状态；实机由已有 SDK 转接脚本统一输出相同 `/joint_states`。RViz 只是 `/joint_states → robot_state_publisher → TF` 的显示端，可不启动。

```bash
# 无 MoveIt/RViz 的本体+夹爪 SDK 模拟；可 publish_tf:=false
ros2 launch lycus_moveit_config sdk_sim.launch.py
# 原 MoveIt 仿真仍可用
ros2 launch lycus_moveit_config demo.launch.py
```

仿真左右夹爪各自提供标准 FollowJointTrajectory：`/left_gripper_controller/follow_joint_trajectory` 与 `/right_gripper_controller/follow_joint_trajectory`。手臂与夹爪控制器独立，位置直接使用 URDF 单位。debug 在整次规划成功后复用 TrajectoryExecutor 逐段下发，手臂段只选手臂列，开合段只选夹爪列，并保留每段计划时间；当前段成功后才发下一段，失败/取消停止后续段。段间控制器处理开销不计入计划轨迹时间轴。deploy 不调用控制器。两种模式均发布 `/curobo/planned_trajectory`；只把反馈写入内部实测状态，不向 `/joint_states` 发命令。

## 调用与导出

完整例子在 `src/curobo_ros_python/config/pick_place_example.yaml`。PickPlace 顶层用 pick/place，独立 Action 顶层用 task。调用脚本从 YAML 设置字段。

```bash
python3 src/curobo_ros_python/scripts/pick_place_client.py \
  --action pick_place --goal-file /tmp/pick_place_goal.yaml --output /tmp/pick_place_plan.json
```

导出 JSON schema 为 curobo_task_v2，含完整 Action Result、关节单位、整段/分段时间说明，以及 execution_requested / execution_completed；deploy 两者均 false，debug 全部控制器成功后两者均 true。`obj_pose_to_pick.py` 把检测位姿写入 `tool_pose`，并把夹爪偏移、接近和抬升换成相对位姿。

接口变更后需重新构建 curobo_ros_interfaces / curobo_ros_python，使用新消息环境并重启旧规划节点。必要的接口构建与针对性验证由本次实现记录单独报告，不要求全量离线回归。

## 本次执行改动验证（2026-11-04）

- 已构建 curobo_ros_interfaces、curobo_ros_python、lycus_moveit_config。
- 独立 ROS domain 195 的 mock 环境中，真实 OPT 模型完整 PickPlace 规划为 341 点、22 个独立关节、15 段；全部规划完成后再下发，12 个手臂段与 3 个夹爪段均执行成功。
- 同一隔离环境验证 deploy 对应分支：Pick 返回 101 点完整轨迹，执行器调用次数不增加，execution_requested / execution_completed 均为 false。
- 统一 /joint_states 只有 joint_state_broadcaster 一个发布者；夹爪反馈从 0 到 -0.021 m 再张开，另一侧保持不变，四片夹指的 TF 正常更新。
- 本次只做上述针对性验证，未运行全量回归，未操作实机或重启已有业务进程。
