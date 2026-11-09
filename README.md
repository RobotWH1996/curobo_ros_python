# curobo_ros_python

ROS 2 Humble 双臂规划节点，支持 Lycus（`lycus.yml`）和 K100（`k100.yml`）。OPT 提供多目标规划 Service 和抓放任务 Action，MPC 通过 Action 跟踪末端目标。两者在同一节点中互斥使用 GPU。

## OPT 抓取与放置（Pick / Place / PickPlace）

三个 Action 的外段复用现有 `plan_joint` / `plan_pose`，中段用连续 IK 生成笛卡尔插补，按抓放阶段规划，并在返回前拼接好**完整本体关节 + 左右夹爪关节**的 `JointTrajectory`。手臂仍按原来的规划组优化，夹爪不加入 OPT 自由度；抓取前的张开叠加到接近运动中，按夹爪能力尽快张开，提前完成后保持；接近路程不足时，先在起点补足张开时间，再连续通过 prepick。抓取闭合和放置释放期间保持手臂静止。普通路点、预置点、抬升点、撤退点和中间 end 不再逐段停顿，整次任务起终点仍平稳启停。

三个接口按抓取/放置的自然停顿点分块：首块规划并验收通过后即可下发，后续规划与首块运动重叠。debug.yaml 的 execute_trajectory=true，按同一时间轴续接手臂和夹爪控制器，不等待运动完成；deploy.yaml 为 false，只分块发布并最终返回完整轨迹，供实机转接脚本使用。请求中不再设置 execute。夹爪使用通用关节轨迹接口，不依赖设备专用消息或 SDK。

### 接口、阶段与完整轨迹

| 调用 | 目标字段 | 用途 |
| --- | --- | --- |
| `/pick`，`Pick` | `task.tool_pose` | 从当前状态规划张开、抓取、闭合、抬升及额外终点 |
| `/place`，`Place` | `task.tool_pose` | 假定起点已夹持工件，规划携物、释放、撤退及额外终点 |
| `/pick_place`，`PickPlace` | `pick.tool_pose`、`place.tool_pose` | 一次性规划完整抓、搬、放；两部分必须使用同一手臂 |

抓放只支持 `left_arm` 或 `right_arm` 单臂 OPT，使用 mesh 点云碰撞。左/右工具系分别为 `left_arm_ee_link` / `right_arm_ee_link`。原有单臂 7 关节、双臂 14 关节优化，多关节角目标、多笛卡尔目标、混合路点、IK、MPC、本体点云过滤均保留。

```text
Pick：接近时同步张开〔混合路点 → prepick〕→ pick〔停顿闭合〕→ lift → 启用工件碰撞 → end_pose
Place：携物混合路点 → preplace → place → 张开插补 → 解除附着 → retreat → end_pose
PickPlace：完整 Pick → 完整 Place（包括各自可选的 end_pose）
```

每次请求从统一 `/joint_states` 取初始状态，后续各段使用上一段的计划终点继续规划，不把计划位置写回真实反馈。未运动手臂、躯干、头部等关节保持起始位置。输出关节覆盖机器人模型的非固定、非 mimic 独立关节，包含 `left_gripper` 和 `right_gripper`；按 `trajectory.joint_names` 读取，不硬编码数组索引。mimic 由 URDF/robot_state_publisher 计算，不单独控制。

| 公共输入 | 默认 | 含义 |
| --- | --- | --- |
| `timeout` | 0.0 | 本次规划及异步提交的总超时，秒；0 使用节点默认 180 秒，不含返回后的执行时间 |

| 结果 | 含义 |
| --- | --- |
| `success`、`stage`、`message` | 当前模式任务是否成功、结束/失败阶段及原因；不表示实际夹到物体 |
| `execution_requested`、`execution_completed` | 前者表示 debug 请求下发；规划侧不等待执行，后者保持 false，不能据此判断实机完成。deploy 下两者均为 false |
| `trajectory` | 返回前完成拼接的整条轨迹，统一关节顺序、全任务时间轴 |
| `trajectories` | 各阶段的完整关节轨迹，每段从自己的局部时间开始 |
| `trajectory_stages` | 与 trajectories 一一对应；连续运动段用 + 连接原阶段名，close/release 单独保留 |

整条轨迹已经包含夹爪位置和必要停留，不需要控制方再猜开合时刻。分块如下：

| 接口 | 第 1 块 | 第 2 块 | 第 3 块 |
| --- | --- | --- | --- |
| Pick | 起点/路点 → prepick → pick → 闭合及等待 | lift → 可选 end | — |
| Place | 起点/路点 → preplace → place → 张开及等待 | retreat → 可选 end | — |
| PickPlace | 起点/路点 → prepick → pick → 闭合及等待 | lift → 可选 Pick end → 携物路点 → preplace → place → 张开及等待 | retreat → 可选 Place end |

块内保留连续平滑与原分阶段验收。预抓取、提升、预放置和撤退不是下发边界，不额外停顿。块间在抓取/放置的既定静止段衔接；已提交前缀的位置、速度、加速度和时间不再修改。后续规划或确认错过续接窗口时明确失败并取消，不能保证任意计算延迟下仍连续。

debug 分别向手臂、夹爪的标准 FollowJointTrajectory 提交相同起跑时间的累计前缀；更新保留全部旧前缀，控制器在当前时刻继续跟踪。左右控制器确认后才能提交下一版，防止异步乱序。适配器处理拒绝、确认超时、取消和迟到确认；执行尚未结束时拒绝新任务。规划 Action 的 success 表示规划及提交阶段成功，execution_completed 保持 false，不表示实际运动或夹持完成。取消或后续规划失败会取消已经提交的控制轨迹；此时首块可能已经开始执行。

两种模式的 `/curobo/planned_trajectory` 都逐次发布**累计的已验收前缀**，各消息共用 header.stamp，旧前缀不变，点时间从同一起点累计。最终 Result.trajectory 返回完整轨迹。deploy 不调用控制器；外部适配器若消费中间消息，应按同一时间轴续接，不能每收到一条就从第一个点重新执行。失败结果可能带已提交前缀，应检查 success。

### PickTask 与 PlaceTask 参数

| 字段 | Pick | Place | 含义 |
| --- | --- | --- | --- |
| `group_name` | 必填 | 必填 | left_arm / right_arm |
| `tool_pose` | 必填 | 必填 | 参考位姿 PoseStamped。下面三段位姿相对它 |
| `in_base_frame` | false | false | true：三段位姿在 base_link 下变换。false：在末端坐标系下，结果 = tool_pose × 相对位姿 |
| `pick_pose` | 单位位姿 | — | 相对于 `tool_pose` 的抓取位姿，没有 header |
| `prepick_in_target` / `lift_in_target` | 必填 | — | 相对于 `tool_pose` 的预抓取和抬升，与抓取位姿相距 1 mm–0.5 m |
| `place_pose` | — | 单位位姿 | 相对于 `tool_pose` 的放置位姿，没有 header |
| `preplace_in_target` / `retreat_in_target` | — | 必填 | 相对于 `tool_pose` 的预放置和撤退，与放置位姿相距 1 mm–0.5 m |
| `waypoints` | [] | [] | 到预置点之前依次经过的混合路点 |
| `end_pose` | 空 type | 空 type | 抬升/撤退之后的额外终点，可填关节角或工具位姿 |
| `object_size` | 可省略 | 可省略 | 工件包围盒三边，米；全零读取节点默认，显式值须三个有限正数 |
| `open_position` | 0.0 | 0.0 | 夹爪 URDF 关节位置。Lycus 单位米，0 全开 |
| `close_position` | -0.042 | — | Pick 闭合的计划关节位置；Lycus -0.042 m 全闭，按工件尺寸调整 |

夹爪值直接使用 URDF 关节单位，Lycus 范围为 `[-0.042, 0.0]` m。无需输入开闭时长或速度。RG75-300 以满行程约 0.55 s 为标称能力，按模型行程比例和 URDF 速度约束生成平滑开合轨迹；75 mm 不直接作为单指关节行程。夹爪起步即张开，提前完成则保持，路程过短则在起点补足张开时间，避免在 prepick 停顿。闭合/释放时间也自动规划。旧消息中的 gripper_duration 仅保留兼容，不再读取；YAML 和客户端已移除该输入。close_position 仍由调用方给出预计夹持位置。

`config/debug.yaml` / `config/deploy.yaml` 的节点参数 `gripper_wait_time: 1.0` 控制额外稳定等待（秒），供 Pick、Place、PickPlace 共用。抓取点先完成闭合，再保持手臂和夹爪静止该时长后抬升；放置点先完成张开，再保持该时长后撤退。它不包含夹爪开合时间，不用于抓取前提前张开，也不轮询反馈。`0.0` 关闭额外等待；必须为有限非负数。每次规划开始时固定本次使用的值。修改节点 YAML 后重启生效，也可用 `ros2 param set /curobo_motion_planning_node gripper_wait_time 0.5` 修改后续任务的等待。此参数不改变 Action 消息格式。


单独 Place 用自己的 `object_size`，工件相对工具由 `place_pose` 反推。它从初始 `/joint_states` 对应工具位姿建立附着模型，夹爪起始开度也来自该反馈，不依赖上一次 Pick。PickPlace 使用本次 Pick 的几何。任务结束恢复本次临时附着、放置场景和真实反馈状态，不跨请求保留携物记录。

`object_size` 全零时读取 `config/debug.yaml` / `config/deploy.yaml` 的 `pick_place_default_object_size`。该默认值使用前须填三个正尺寸；请求里的显式尺寸也必须三个都为正。

### 位姿、混合路点和额外终点

PoseStamped 可以使用 base_link 或能通过 TF 转到 base 的坐标系，空 frame_id 按 base 处理。抓取、预抓取、抬升，以及放置、预放置、撤退，都是相对 `tool_pose` 的位姿，实际位姿为两者相乘。工件中心相对工具取抓取或放置相对位姿的逆。例如 `pick_pose` 为 `(0, -0.15, 0)` 且姿态相同，工具在参考点 -Y 15 cm，工件中心则在工具 +Y 15 cm。`object_size` 不会自动从 CAD 或点云估计。

```yaml
waypoints:
  - type: joint
    joint_positions: [0.1, 0.2, -0.1, 0.4, 0.0, 0.2, 0.0]
  - type: pose
    pose:
      header: {frame_id: base_link}
      pose:
        position: {x: 0.55, y: 0.25, z: 1.30}
        orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
end_pose:
  type: joint
  joint_positions: [0.0, 0.2, -0.1, 0.4, 0.0, 0.2, 0.0]
```

关节角是当前手臂七关节顺序的弧度，与 PlanJointGoal 相同。一个路点只填写 joint 或 pose 一种目标；type 为空仅用于不需要的 end_pose。路点各自复用原 OPT，随后去自动生成的 prepick/preplace。Pick 的 end 在 lift 后，Place 的 end 在 retreat 后，数组为空或 end.type 为空时不增加对应段。示例值只展示格式，需要按任务修改。

### 碰撞与本体点云过滤

起点经混合路点到预置点、抬升/撤退后到可选 end_pose，复用现有 OPT：GPU 做整段轨迹避障，FCL 检查关键点。普通 OPT Service 的检查保留。

启用碰撞检查的阶段，OPT/IK 的 FCL 与 OPT/MPC 的 GPU 都只检查涉及本规划组运动部件的碰撞：环境检查只查询运动臂、工具和随其运动的工件；自碰撞保留运动臂内部、运动臂与另一臂/躯干的配对，排除两侧都静止的配对，并继续遵守已有忽略表。静止本体几何保留用于避让；其与环境的既有碰撞不单独否决本次运动。本体点云仍仅滤除对应运动臂，不启用全本体滤除。

| 区间 | 规划方法 | 碰撞策略 |
| --- | --- | --- |
| Pick 起点、路点 → prepick | OPT | 按当前手臂夹爪全开（URDF upper）做 GPU 避障与 FCL 关键点检查 |
| prepick → pick → lift | 平移直线 + 姿态 SLERP + 连续 IK | 不检查环境、自碰撞或 FCL；保留 IK、关节位置/速度/加速度/jerk 限位 |
| lift 后 → Pick end | OPT | 恢复正常碰撞，附着工件随工具运动 |
| Place 携物起点、路点 → preplace | OPT | 正常检查机器人和附着工件 |
| preplace → place → retreat | 平移直线 + 姿态 SLERP + 连续 IK | 不检查环境、自碰撞或 FCL，包括工件 |
| retreat 后 → Place end | OPT | 恢复正常碰撞，放下的工件作为静态障碍 |

中段不调用 OPT。批量 IK 使用缓存求解器，局部失败时用连续求解和步长细分；不能保持连续到达时明确失败，不跳过路点。预置点与抬升/撤退位姿相同时，复用同一路径反向返回，避免重复 IK。轨迹重定时后再用 FK 检查直线、姿态及终点误差。只有闭合/释放段保持手臂不动。合并保留原始 OPT/插补时间轴，保留正常巡航段的时序；根据两侧实际运动速度识别衔接处的减速/静止尾部，局部压缩并平滑通过，动态超限时才局部减速；保留各阶段准确边界；随后按控制器的五次插值加密采样，复查位置/速度/加速度/jerk 限位。普通阶段恢复其原点云场景、夹爪开度和携物模型做 GPU 路径及 FCL 关键点检查；免碰撞接触阶段仍验收 FK 直线、姿态和终点。每块验收失败时不提交该块，并取消已提交轨迹；此前已验收块可能已开始运动。

全开碰撞模型是抓取前整段的规划假设，不覆盖从闭合到张开的动态扫掠体积；输出夹爪仍从实际反馈开度变化。接近路程不足时，起点的等待由规划器提前写入轨迹，不轮询夹爪反馈。中段免碰撞适用于调用方已确认安全的接触区间；外段不会因中段失败而自动关闭碰撞。工件使用包围盒/球簇，附着后按 OBB 对携物点云去重。

本体过滤按 left_arm/right_arm/both_arms 处理对应手臂。`self_filter_mode`：off 关闭，observe 只分类和可视化，apply 删除本体候选后参与规划。debug/deploy 配置当前均为 apply。点云若有单个整数 `label` 字段，正标签点作为感知工件保留，不因本体余量被删；没有标签时仍执行原来的几何过滤。过滤失败保留整帧，并记录原因。详情见 [本体点云过滤](docs/本体点云过滤.md)。

### 统一反馈与仿真/实机边界

```text
OPT/插补 → 已验收累计前缀 → debug：同时间轴续接 → 仿真控制器
                           → deploy：发布/返回 → 实机统一适配器
                                                   ↓ 实际状态
                                      /joint_states → cuRobo
                                                    → TF / RViz（可选）
```

`/joint_states` 是实际状态输入，不用命令值冒充反馈。仿真由 Lycus 的 ros2_control 驱动本体和夹爪，一个 joint_state_broadcaster 统一发布实际独立关节；实机由现有本体/夹爪 SDK 适配脚本统一打包相同消息。规划侧只认识关节名和 JointTrajectory，不区分底层设备。RViz 通过 `/joint_states → robot_state_publisher → TF` 显示，进程可选，不参与状态刷新。

规划程序不创建设备专用话题或加载设备消息包，夹爪不加入手臂规划组。debug 通过 TrajectoryExecutor 控制适配器分块续接同一手臂、夹爪时间轴；规划器不等待运动完成，不进行设备专用物体夹持检测。

Lycus 无界面模拟可运行 `ros2 launch lycus_moveit_config sdk_sim.launch.py`（默认发布 TF，可加 `publish_tf:=false`）；原 `demo.launch.py` 继续带 MoveIt/RViz。左右夹爪采用 `control_msgs/action/FollowJointTrajectory`，地址 `/left_gripper_controller/follow_joint_trajectory`、`/right_gripper_controller/follow_joint_trajectory`，各自只接收同侧实际夹爪关节，位置单位米，按轨迹时间插补。

### 构建、调用和保存结果

```bash
cd /home/ai/wh/curobo_ros_python
source /opt/ros/humble/setup.bash
colcon build --packages-select curobo_ros_interfaces curobo_ros_python \
  --symlink-install --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash
ros2 launch curobo_ros_python curobo_motion_planning.launch.py
```

接口修改后调用方须重新编译/source 并重启旧规划进程；Lycus 控制器配置也需重新构建并在下次仿真启动时加载。debug 启动默认执行，deploy 启动默认只输出；沿用原 execute_trajectory 配置，不新增任务执行开关。可编辑模板见 [pick_place.yaml](src/curobo_ros_python/config/pick_place.yaml)，填写实测坐标和尺寸后调用：

```bash
python3 src/curobo_ros_python/scripts/pick_place_client.py \
  --action pick_place --goal-file /tmp/pick_place_goal.yaml --output /tmp/pick_place_plan.json
```

YAML 顶层为 `timeout`、`pick`、`place`；旧 YAML 的 execute 必须删除。独立 Pick/Place 改成 `task`：

```yaml
timeout: 180.0
task:
  group_name: left_arm
  tool_pose: # 独立 Place 同样填写 tool_pose
    header: {frame_id: base_link}
    pose:
      position: {x: 0.60, y: 0.25, z: 1.10}
      orientation: {w: 1.0}
  object_size: {x: 0.04, y: 0.04, z: 0.06}
  # 其他字段使用上表默认值；独立 Place 应按实际携物设置几何。
```

```bash
python3 src/curobo_ros_python/scripts/pick_place_client.py \
  --action pick --goal-file /tmp/pick_goal.yaml --output /tmp/pick_plan.json
python3 src/curobo_ros_python/scripts/pick_place_client.py \
  --action place --goal-file /tmp/place_goal.yaml --output /tmp/place_plan.json
```

客户端打印完整关节名、总点数、时长与阶段名。`--output` 保存 `curobo_task_v2` JSON，含 Action Result 的完整轨迹、关节单位/时间轴及 execution_requested / execution_completed；保存文件本身不产生额外执行。`obj_pose_to_pick.py` 同样支持 `--output`，从检测工件位姿换算工具目标。它的 `--open-position/--close-position` 使用米，开闭时间由规划自动生成。

| 节点配置 | 默认 | 用途 |
| --- | --- | --- |
| `pick_place_timeout` | 180 s | 请求 timeout=0 的总规划超时 |
| `pick_place_joint_state_max_age` | 1 s | 初始关节反馈最大年龄 |
| `pick_place_cloud_max_age` | 2 s | 初始环境点云最大年龄 |
| `pick_place_path_tolerance` | 0.01 m | 接近、抬升、撤退的直线路径偏差 |
| `pick_place_max_placed_objects` | 16 | 本次任务已放置物体容量 |

规划失败先看 `stage/message`：目标不可达时检查工具/工件坐标换算和接近方向；关键点碰撞看 FCL 的连杆对；接触段仍失败时检查底座/手腕等未豁免的 GPU 碰撞球；初始状态过期则检查统一 `/joint_states` 是否包含本体和左右夹爪。更详细的阶段规则见 [抓取放置Action](docs/抓取放置Action.md)。

## 环境安装与构建

推荐目录结构：

```text
workspace/
  curobo/                 # 已配置 CUDA/Torch 的 Python 3.10 .venv
  curobo_ros_python/      # 本仓库
  Lycus_ros/             # 机器人驱动、控制器和描述（按部署需要）
```

前提是 NVIDIA GPU、可工作的 CUDA/Torch、ROS 2 Humble，以及 `../curobo/.venv/bin/python` 能导入 cuRobo。本仓库不安装驱动、Torch 或 cuRobo 本体。

OPT 必需依赖为 `python-fcl==0.7.0.11`、`trimesh>=4.0`，见 `requirements-fcl.txt`。用 cuRobo 的 Python 安装：

```bash
../curobo/.venv/bin/python -m pip install -r requirements-fcl.txt
```

Jetson 若已配置 `libcusolver_batched_stub.so`，launch 会使用现有兼容库。遇到 Torch/cuSOLVER 导入错误，先修复 cuRobo 环境。

构建：

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

zsh 交互终端使用 `setup.zsh`。构建使用 ROS 系统 Python；规划节点使用带 Torch/cuRobo 的 Python 3.10。修改源码或配置后重新构建并重启节点。

## 启动与 TF

机器人驱动先发布真实 `/joint_states`；执行轨迹还需要对应控制器。

```bash
source install/setup.bash

# 调试：可视化开启，默认允许控制器执行
ros2 launch curobo_ros_python curobo_motion_planning.launch.py

# 调试但不执行
ros2 launch curobo_ros_python curobo_motion_planning.launch.py execute_trajectory:=false

# 部署：默认只发布规划结果，关闭可视化及控制器切换
ros2 launch curobo_ros_python curobo_motion_planning_deploy.launch.py
```

实际参数以 `src/curobo_ros_python/config/debug.yaml`、`deploy.yaml` 为准。当前两者插值步长均为 `0.02 s`，TrajOpt 速度缩放为 `0.5`；MPC 周期为 `0.10 s`、速度缩放为 `1.0`，启动预热默认关闭。调试 launch 提供多个命令行规划参数覆盖，部署 launch 的覆盖项主要是 `config`、`robot`、Python 和 TF 参数。用 `--show-args` 查看对应入口支持项。

当前 Debug launch 只启动规划节点，需要外部机器人 TF。Deploy 默认增加 `robot_state_publisher`，使用 cuRobo robot YAML 对应 URDF，并订阅配置的 `joint_states_topic`；活动关节 TF 随反馈更新，固定关节发布到 `/tf_static`。已由 MoveIt/机器人 bringup 发布同一套 TF 时，部署 launch 加 `publish_robot_tf:=false`。

Lycus 相机命名统一为：

```text
base_link -> ... -> HeadRGBD_link -> camera_01_link -> camera_01_color_optical_frame
```

cuRobo 模型须包含相机改名提交 `1ebff3b` 或后续版本。最后一段光学 TF 由相机驱动或标定外参提供。Deploy 可用 `publish_camera_tf:=true` 发布；同时传入实际的 `camera_parent_frame`、`camera_frame`、`camera_xyz`（m）、`camera_rpy`（rad）。该开关默认关闭，默认外参只是示例，启用前应核对标定。

## OPT 定版流程

1. 从当前反馈建立规划起点，原地更新夹爪/mimic、躯干和头部几何；这些反馈关节不参与手臂优化或下发。
2. 点云通过 TF 转到 `base_link`，更新 GPU 环境及 CPU FCL mesh 环境；相同点云的 FCL 场景跨规划组缓存复用。
3. 关节目标：提取规划组活动关节，每个路点先全身 FCL；碰撞则整次请求失败。单点走关节空间规划，多点先整体优化，失败再分段。
4. 位姿目标：转到 `base_link`，IK 的成功候选逐个做全身 FCL，剔除碰撞候选，剩余候选才进入 TrajOpt。单点有重试及 PRM 种子路径，多点先整体路点优化，失败后分段衔接。
5. 优化内部仍使用 GPU 球碰撞及 L-BFGS。完整双臂位姿请求不做多余当前 FK；单臂多路点复用当前 FK。
6. 输出补回双臂关节顺序，未规划侧保持当前角度、导数为零，保留规划侧速度/加速度/jerk；实际输出终点再全身 FCL，通过后发布并按执行开关下发。

`both_arms` 的每个位姿路点必须同时提供左右目标；仅给一侧会报错，单臂目标应使用 `left_arm` 或 `right_arm`。双臂关节路点为 14 个角度（左 7、右 7），单臂为 7 个；单位 rad。Lycus 关节前缀为 `AR5-5_08L/R-W4C1C5-ZY2_`。ROS 四元数顺序为 xyzw。

### 碰撞配置与边界

- GPU 球模型使用 cuRobo 的 `lycus.yml/self_collision_ignore`。
- Lycus 的 FCL 基础配对使用 `config/lycus_fcl_collision.yaml` 的 `self_collision_ignore` 独立完整表。此表检查左右 link5/link7；GPU 暂时忽略这两对以避免粗球误报。代码没有五七特例，也没有独立腕部模块。
- 其他机器人尚未提供独立 FCL 表，沿用各自机器人配置中的配对。
- FCL 在 CPU 上检查 mesh 表面相交，覆盖 IK 候选、输入关节目标和实际输出终点；不覆盖整条轨迹，不是连续碰撞或封闭物体包含检测。中间路径及拼接检查仍使用球模型。
- URDF 没有 collision 几何的连杆会在启动日志列出，不进行 mesh 检查。

日志示例：

```text
OPT: IK 候选已启用运动相关 FCL 检查，保留运动臂与静止本体碰撞检查
FCL IK筛选: 检查=2, 通过=2, 剔除=0, FCL耗时=...ms; 环境=...
FCL 输出终点: 通过, 自碰撞通过; 环境=..., 耗时=...ms
FCL 关节目标[1/3]: 通过, ...
```

候选 FCL 耗时已包含在 IK 阶段计时中，不能重复相加；`check` 是底层规划器原有验收计时，不代表 FCL。`wall` 是 ROS 请求处理到规划完成的实际耗时，含状态同步、目标准备和输出整理，不含后续轨迹执行等待。

## 日志输出与排查

调试、部署 launch 都支持 `log_level`，默认 `info`。只调整规划节点的日志级别，不改变规划或执行参数：

```bash
# 日常查看阶段、结果、碰撞策略和耗时
ros2 launch curobo_ros_python curobo_motion_planning.launch.py log_level:=info
# 排查目标输入：额外显示当前关节、目标关节/位姿、mesh 连杆和发布话题
ros2 launch curobo_ros_python curobo_motion_planning.launch.py log_level:=debug
# 部署入口同样支持
ros2 launch curobo_ros_python curobo_motion_planning_deploy.launch.py log_level:=info
```

| 级别 | 输出内容 |
| --- | --- |
| INFO | OPT/IK 请求与结果、FCL 筛选结果、抓放阶段与碰撞策略、控制器执行阶段、轨迹点数与耗时、夹爪插补段 |
| DEBUG | 当前关节、目标关节值/位姿、过滤 mesh 连杆列表、轨迹发布话题 |
| WARN | 请求拒绝、任务取消、本体过滤跳过/候选比例过高、MPC 周期超时 |
| ERROR | 规划失败、执行失败、抓放阶段异常或恢复碰撞状态失败 |

本体过滤统计和 MPC 周期统计按 2 秒节流，不逐帧打印；抓放夹爪阶段记录计划位置和时长。ROS 默认还会在每行前加时间戳、节点名和级别。抓放结束日志的 `reason` 使用引号，异常文本中的换行会被转义。

### 抓放日志

每个任务使用 Action Goal UUID 标识。按 task 查找同一次请求的阶段、碰撞策略、求解耗时和最终结果；phase 区分 Pick/Place 中同名阶段。

- `method=cartesian_interpolation` 表示中段用连续 IK 插补；`self_collision=off world_collision=off fcl=off` 表示该段不检查碰撞，仍检查运动学和关节动态限位。
- `method=opt fcl=keypoints gpu_collision=on` 表示外段使用正常 GPU 轨迹避障和 FCL 关键点检查。
- `state_ms/world_ms/solve_ms` 分别为状态同步、场景更新、OPT 求解耗时；segment_ms 为本段总耗时，不能与子项重复相加。
- `trajectory_s` 表示计划运动时长；任务 elapsed_ms 包含规划、验收、可选的异步下发及清理耗时，不表示实际运动时长。
- 夹爪张开/闭合/释放的规划日志说明生成插补轨迹；debug 下发日志表示向控制器提交，规划侧不等待执行结果。success 表示规划及提交阶段成功，不表示实际夹到物体。

输入、初始反馈和忙状态拒绝发生在阶段计算前。失败时先看最后的 stage/reason，再看对应 FCL 或 OPT 日志。普通 OPT/MPC 的控制器执行日志仍按原接口输出。

失败明细区分初始点、路点和目标点，并列出碰撞类型及双方连杆/障碍名称。环境点云生成的网格统一命名为 `environment`，该名字代表整份环境点云，不代表摄像头部件。

IK 未到达目标时写作“目标点的IK候选碰撞”；失败优化轨迹的末端写作“目标点的失败轨迹末端候选碰撞”，不把候选构型当作实际目标状态。后段失败以已完成上一段的终点作为起点。失败轨迹最多抽查8个点、输出4条碰撞明细；这是失败诊断，不是整条轨迹的 FCL 验收，未报告碰撞也不证明轨迹无碰撞。诊断耗时不计入后续 IK 候选筛选统计，成功规划不运行这段诊断。


### 本体点云过滤与 MPC

```text
[本体过滤] mode=off，不做本体分类；仍执行常规点云预处理
[本体过滤] mode=observe group=left_arm input=12000 self_candidates=3200 removed=0 output=12000 cache=miss elapsed_ms=24.1
[本体过滤] mode=apply group=left_arm input=12000 self_candidates=3200 removed=3200 output=8800 cache=hit elapsed_ms=0.6
[本体过滤] mode=apply group=left_arm status=跳过 kept=12000 reason='本体过滤超出 100ms 预算，整帧不删除'
[MPC] event=周期 timing_ms=(tick=12.0 publish=0.2 send=0.3 feedback=0.1) total_ms=12.6 budget_ms=50.0
```

`self_candidates` 是被判为本体的点数，`removed` 是这一步实际删除数，`output` 是本体过滤步骤的输出数，后面仍可能下采样。observe 的 `removed=0`；off 不运行分类。`cache=hit` 表示同帧、同关节状态和同规划组复用掩码，耗时不能当作重新分类的耗时。`status=跳过` 保留输入点，`reason` 说明缺失关节、几何异常或超时等原因。MPC 的 `budget_ms` 是目标周期预算，`total_ms` 是本周期处理时间，不含补足周期的休眠；MPC 仍不使用 OPT 的 FCL。

可保存启动终端输出，再按任务或类别筛选：

```bash
ros2 launch curobo_ros_python curobo_motion_planning.launch.py log_level:=info 2>&1 | tee /tmp/curobo_runtime.log
grep -F 'task=<实际UUID>' /tmp/curobo_runtime.log
grep -E '\[抓放\]|\[夹爪\]|\[本体过滤\]|\[OPT\]|FCL|\[MPC\]' /tmp/curobo_runtime.log
```

## 接口与示例

| 接口 | 类型 | 用途 |
|---|---|---|
| `/plan_joint_goal` | `curobo_ros_interfaces/srv/PlanJointGoal` | 关节路点规划 |
| `/plan_pose_goal` | `curobo_ros_interfaces/srv/PlanPoseGoal` | 位姿路点规划 |
| `/compute_ik` | `curobo_ros_interfaces/srv/ComputeIK` | IK，成功候选也经 FCL |
| `/follow_mpc` | `curobo_ros_interfaces/action/FollowMpc` | 周期跟踪；取消停止 |

```bash
ros2 service call /plan_joint_goal curobo_ros_interfaces/srv/PlanJointGoal \
  "{group_name: left_arm, waypoints: [{positions: [0, 0, 0, 0, 0, 0, 0]}]}"
```

位姿请求使用 `waypoints`（PoseStamped 列表）、`ee_link`；双臂同时设置等长 `other_waypoints`、`other_ee_link`。末端名为 `left_arm_ee_link`、`right_arm_ee_link`。具体字段以 `msg/curobo_ros_interfaces/srv/` 为准。

输出包括 `/curobo/planned_trajectory`、`/curobo/mpc_plan`，调试可视化包括 `/display_planned_path`、`/lycus_motion_trajectory`、`/curobo/goal_waypoints` 和 `/curobo/collision_world`。

控制器对应 `/left_arm_controller/follow_joint_trajectory`、`/right_arm_controller/follow_joint_trajectory`、`/both_arms_controller/follow_joint_trajectory`。`execute_trajectory` 决定是否执行，`switch_controllers` 决定是否切换控制器。

## 点云与 MPC

默认订阅 `/all_points`，按 `base_link <- camera_01_color_optical_frame` 变换；`pointcloud_child_frame` 为空时采用消息 `header.frame_id`。相机跟随头部运动，因此每次获取 TF。

`require_pointcloud: false` 时缺少点云/TF 可能按空场景规划；FCL 日志会写明 `空场景（未检查环境障碍）`。需要强制环境保护时在 YAML 中设为 `true`。可用以下命令诊断：

```bash
ros2 topic echo /all_points --once --field header
ros2 run tf2_ros tf2_echo base_link camera_01_color_optical_frame
```

MPC 仍使用原有 GPU 球碰撞，不接入本轮 OPT FCL。目标由 `/curobo/target`（`MpcTarget`）输入，启动 `/follow_mpc` 后滚动优化并发布 `MpcPlan`。当前命令序列覆盖约 0.5 s；`mpc_prewarm` 开启时缓存指定模式，没点云可空场景预热，有点云但 TF 缺失仍需等待。运行 MPC 时拒绝并发 OPT/IK 请求。

部署调用脚本位于 `src/curobo_ros_python/scripts/`：`call_pick_place.sh` 与 `pick_place_client.py` 调用抓放 Action，`obj_pose_to_pick.py` 与 `obj_pose_to_plan.py` 接入感知目标；`perception_pick_target.py` 为感知读取共用模块。使用各入口 `--help` 查看参数。

## 部署源码范围

本仓库仅保留规划节点及模块、ROS 消息/Service/Action、运行配置、启动文件、实际调用脚本和使用文档。OPT/MPC、多目标、抓放、本体点云过滤、GPU/FCL 验收及调试启动入口继续保留。`deploy.yaml` 仍只发布规划结果；`debug.yaml` 保留原仿真执行行为。

抓放调用参数统一放在 `src/curobo_ros_python/config/pick_place.yaml`，沿用清理前正在使用的参数；只调整文件名，不自动修改工件尺寸、相对位姿、夹爪开度或 end_pose。默认感知选取逻辑也保持原样。

```bash
./src/curobo_ros_python/scripts/call_pick_place.sh pick --label 5
./src/curobo_ros_python/scripts/call_pick_place.sh place
./src/curobo_ros_python/scripts/call_pick_place.sh pick_place --label 5
```

测试、基准脚本、结果输出、历史分析报告和 CAD 草稿不随部署仓库发布。生成结果请输出到 `/tmp` 或仓库外的指定目录；构建产物、日志和虚拟环境继续忽略。

### 抓放插补 IK 启动预热

`pick_place_ik_prewarm: true` 默认在节点初始化时同步预热左右臂终点 IK 和 33/65/129 点批量 IK，完成后才输出服务和 Action 就绪日志。预热使用模型默认姿态，不依赖点云/真实关节反馈，不发布轨迹或下发控制器。正常规划仍检查真实反馈、点云和原来的碰撞策略。

`pick_place_ik_prewarm_modes` 可只选择需要的手臂；`pick_place_ik_prewarm_samples` 可选 33/65/129/257/513。默认三种覆盖当前常见接近/撤退段。预热复用运行时相同缓存，改变目标位姿不会重建同规格 CUDA 图。未预热规格、其他手臂或罕见备用 IK 首次使用仍可能初始化；重启节点后缓存重建。设 false 可缩短启动，但首次使用延迟会回到任务内。

日志 `[插补预热] event=完成 ready=true` 表示所选规格已完成，逐项日志记录手臂、样本数和耗时。已有 TrajOpt/MPC 预热配置继续独立生效；这项预热只将插补 IK 初始化移到启动阶段，不省略每条任务的 IK、重定时或 FK 校验。

路点连续通过会记录 `event=路点通过速度`：不仅要求速度非零，还检查连接处是否仍保留原段末端的长时间低速。接触段的通过方向沿原直线路径，普通段仍按阶段恢复 GPU/FCL 验收。夹爪未张开时在起点补足等待；连续通过的关键边界若合并后速度为零，则拒绝提交该块。

预置点衔接采用两侧约 45% 峰值速度附近的运动区间，适度提高通过速度；全局速度倍率不变，仍按实际关节速度、加速度和 jerk 限位局部调整，并做原 GPU/FCL 与接触段直线验收。

### 工具 Z 轴竖直偏好

节点参数 `keep_tool_z_vertical`：默认值为 `false`，实际启用状态以启动 YAML 或运行时参数为准；设为 `true` 开启。它对 OPT 的关节目标、笛卡尔目标及复用这些接口的 Pick/Place/PickPlace 中间运动加入软代价，抑制工具 Z 轴偏离 base 竖直轴；朝上或朝下取当前段目标工具姿态对应的方向，绕竖直轴旋转不受这项代价限制。输入的关节目标、末端位置和终点四元数仍按原约束求解，不会被改写。

这是“尽量保持”的软偏好，不保证每一帧绝对竖直；非竖直起点、非竖直目标、避障和关节限位可能要求倾斜。接触段仍遵守已指定的直线/SLERP 位姿，不覆盖用户给定的抓取、放置姿态。合并后记录 `event=工具Z竖直检查` 的实际平均/最大倾角，仍做原碰撞、直线和动态限位验收。MPC 不受此参数影响。

初始化时在 CUDA Graph 捕获前安装代价；每次 OPT 求解只原地更新选中工具的权重和朝向，求解结束或失败均恢复，不污染下一次关闭该参数的规划。修改 YAML 后重启生效，也可在新版本节点上用 `ros2 param set /curobo_motion_planning_node keep_tool_z_vertical false` 切换后续规划。

竖直偏好的绕路优化：允许约 15° 的小倾角，范围内不再追求绝对 0°，因此当前约 11° 的抓取姿态不与中间偏好冲突。开启时还加入起终点之间约 10 cm 半径的位置软走廊，抑制为了姿态而大幅侧移；它不是硬性直线约束，也不保证最大偏移小于 10 cm。障碍、关节限位及原碰撞验收仍生效，输入终点不被改写；关闭时这两项附加代价均为零。不额外增加一轮 OPT 求解。
