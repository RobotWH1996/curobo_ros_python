# curobo_ros_python

ROS 2 封装，用 cuRobo 做双臂运动规划。默认机器人 **Lycus**（`lycus.yml`），也支持 K100（`robot:=k100.yml`）。

同一节点提供两套算法，debug / deploy 只改可视化和是否下发控制器，不切换算法：

| | TrajOpt | MPC |
|--|---------|-----|
| 接口 | Service `plan_*` / `compute_ik` | Action `/follow_mpc` |
| 做什么 | 一次规划整条轨迹 | 周期滚动优化，跟笛卡尔目标 |
| 结果 | `/curobo/planned_trajectory` | `/curobo/mpc_plan` |
| 执行 | `execute_trajectory:=true` 时 FollowJointTrajectory | 同一开关：debug 每拍下发，deploy 只发轨迹 |
| GPU | 节点启动时加载并 warmup | 第一次 `follow_mpc` 才创建；空闲不 tick |

节点和 cuRobo 跑在**同一个 Python 3.10**（cuRobo 的 uv 环境 `.venv` 里要同时有 `torch`；`rclpy` 来自 ROS 2 Humble）。

和 `curobo`、`Lycus_ros` 同级：`/home/victor/wh/curobo_ros_python`。cuRobo 本体在 `../curobo/`（`import curobo`），机器人 YAML 在 `curobo/content/configs/robot/`。

## 目录

```
curobo_ros_python/
  msg/
    curobo_ros_interfaces/        # srv / msg / action
    perception_msgs/              # /obj_poses（Pose6DArray）
  src/curobo_ros_python/
    config/debug.yaml             # 仿真 / RViz：可视化 + 下发控制器
    config/deploy.yaml            # 实机：只发 topic，不画、不跟控制器
    launch/
    scripts/
      pub_left_waypoints.py       # 循环发左臂 MPC 路点
      obj_pose_to_plan.py         # 一帧目标位姿 → plan_pose_goal
      plot_mpc_joint_tracking.py  # 录命令 vs /joint_states
    curobo_ros_python/
      main.py                     # 节点入口
      motion_planning_node.py     # ROS 接口
      planning/
        algorithms/trajopt.py     # CuroboPlanner
        algorithms/mpc.py         # DualArmMpc
        mpc_follow.py             # FollowMpc 周期
        trajectory.py / groups.py / geom.py / pointcloud.py
      visualization/
        rviz.py                   # 拼 Marker / DisplayTrajectory
        publish.py                # 按开关发到 RViz
```

## 依赖

- ROS 2 Humble（`/opt/ros/humble`，apt）
- cuRobo 仓库根目录的 `.venv`（与本仓库同级的 `../curobo/.venv`，Python 3.10，能 `import curobo` 和 `torch`）
- NVIDIA GPU + CUDA
- 机器人 bringup：`/joint_states`、臂控制器、TF

zsh 下 source `setup.zsh`。节点 Python 查找顺序：launch 参数 `curobo_python` → 环境变量 `CUROBO_PYTHON` → 当前虚拟环境或 conda → 从本仓库和当前目录向上找到的 cuRobo `.venv`。不写死本机路径。

## 编译

在仓库根目录编译即可，colcon 会递归找到 `msg/` 和 `src/`：

```bash
cd /home/victor/wh/curobo_ros_python
source /opt/ros/humble/setup.zsh
colcon build --symlink-install
source install/setup.zsh
```

`colcon` 用 ROS 的 Python。规划节点会自己找到旁边 cuRobo 仓库的 `.venv`。也可以先激活它：

```bash
source ../curobo/.venv/bin/activate
```

链到已有工作空间：

```bash
ln -s /home/victor/wh/curobo_ros_python/msg/curobo_ros_interfaces ~/ros2_ws/src/
ln -s /home/victor/wh/curobo_ros_python/msg/perception_msgs ~/ros2_ws/src/
ln -s /home/victor/wh/curobo_ros_python/src/curobo_ros_python ~/ros2_ws/src/
cd ~/ros2_ws && colcon build --packages-select curobo_ros_interfaces perception_msgs curobo_ros_python
```

## 启动

先起机器人。参数在 `config/debug.yaml` / `config/deploy.yaml`。优先级：**命令行显式传入 > yaml > 节点默认值**。调试 launch 的覆盖项默认都是空，不传就不会盖 yaml。

仿真 / 对着 RViz 调（可视化开，`execute_trajectory: true`，插值 `interpolation_dt: 0.1`）：

```bash
ros2 launch curobo_ros_python curobo_motion_planning.launch.py
```

实机（订 `/joint_states` 和点云，结果发 topic，不画 RViz、不跟控制器；插值 `interpolation_dt: 0.02`）：

```bash
ros2 launch curobo_ros_python curobo_motion_planning_deploy.launch.py
```

常用覆盖：

```bash
# K100
ros2 launch curobo_ros_python curobo_motion_planning.launch.py robot:=k100.yml

# 只规划不执行（TrajOpt 和 MPC 都不下发控制器）
ros2 launch curobo_ros_python curobo_motion_planning.launch.py execute_trajectory:=false

# 跳过 TrajOpt warmup（启动快，第一次规划慢）
ros2 launch curobo_ros_python curobo_motion_planning.launch.py warmup_iterations:=0

# 自动找到的解释器不对时，再显式指定
ros2 launch curobo_ros_python curobo_motion_planning.launch.py \
  curobo_python:=/path/to/curobo/.venv/bin/python
```

`config:=deploy.yaml` 会按部署配置跑调试 launch。细节参数改 yaml；注释写在 yaml 里。

## 接口一览

节点名：`curobo_motion_planning_node`。`group_name`：`left_arm`、`right_arm`、`both_arms`。响应都有 `success`、`message`。

| 接口 | 类型 | 作用 |
|---|---|---|
| `/plan_joint_goal` | `PlanJointGoal` | 关节路点规划 |
| `/plan_pose_goal` | `PlanPoseGoal` | 末端位姿路点规划 |
| `/compute_ik` | `ComputeIK` | 只求 IK |
| `/follow_mpc` | `FollowMpc` | 打开 MPC；取消即关 |

`follow_mpc` 进行中会拒绝 `plan_*` / `compute_ik`；已有 Goal 时拒绝第二个 `follow_mpc`。TrajOpt 占着 GPU 时，新的 `follow_mpc` 也会被拒绝。

订阅：`/joint_states`、点云（debug / deploy 默认 `/all_points`）、`/curobo/target`（`MpcTarget`）。

发布：

| 话题 | 何时发 | 内容 |
|---|---|---|
| `/curobo/planned_trajectory` | `plan_*` 成功 | `JointTrajectory` |
| `/curobo/mpc_plan` | `follow_mpc` 每个周期 | `MpcPlan`（左右轨迹 + 误差） |
| `/display_planned_path` | debug 可视化开 | `DisplayTrajectory`，整机关节动画 |
| `/{robot}_motion_trajectory` | 同上 | `MarkerArray`：TrajOpt 绿线；MPC 左青 / 右品红预测末端 + 目标球 |
| `/curobo/goal_waypoints` | `visualize_goal_waypoints` 开 | 笛卡尔 / 关节目标的球和 RGB 轴；规划失败也保留 |
| `/curobo/collision_world` | 点云可视化开 | 橙色体素 |

Lycus 末端线话题默认是 `/lycus_motion_trajectory`。`/curobo/goal_waypoints` 和 `/curobo/collision_world` 是 latched（`TRANSIENT_LOCAL`），后开的 RViz 也能看到上一帧。

## TrajOpt

一次规划整条轨迹。`execute_trajectory:=true` 时再 FollowJointTrajectory。debug 默认开，deploy 默认关。

- **单臂**：只优化该臂 7 轴，另一臂钉在当前关节 / 当前 FK。
- **`both_arms` 关节**：每个路点 14 个数，前 7 左、后 7 右。
- **`both_arms` 位姿**：`waypoints`（默认左）和等长的 `other_waypoints`（默认右）一起规划。只填 `waypoints` 时另一臂钉在当前 FK。

1 个路点就是点到点；多个路点按顺序飞过。同一下标是同一时刻的左右目标。四元数是 ROS `xyzw`。`PoseStamped.header.frame_id` 为空或等于 `base_frame` 时直接用，否则查 TF。

关节单位 rad，顺序与 FollowJointTrajectory 控制器一致。`plan_joint_goal` 每个 `waypoints[i].positions` 长度必须等于规划组自由度（7 或 14）。

**Lycus** 单臂：`AR5-5_08L/R-W4C1C9-ZY2_joint_1` … `_joint_7`。`both_arms` = 左 7 + 右 7。

**K100** 单臂：`left_` / `right_` + `shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw`。

执行：

- `left_arm` → `/left_arm_controller/follow_joint_trajectory`
- `right_arm` → `/right_arm_controller/follow_joint_trajectory`
- `both_arms` → `/both_arms_controller/follow_joint_trajectory`

左右臂控制器关节不重叠，可同时激活。`both_arms_controller` 占 14 轴，和单臂控制器互斥，执行前会 `switch_controller`。只用单臂时可 `switch_controllers:=false`。

`trajopt_velocity_scale` 把下发轨迹的时间拉长（debug / deploy 默认 `1.0`）。`0.5` 大约半速。`interpolation_dt` 是规划轨迹的插值步长：debug `0.1` s，deploy `0.02` s。

需要先有 `/joint_states`。规划可能要几秒到十几秒，客户端超时请设大。下面按 Lycus。位姿用仿真里 `tf2_echo base_link left_arm_ee_link` 记下的可达姿态；远路点只改 xyz，不要把四元数写成 `w: 1`。

```bash
# 单臂关节
ros2 service call /plan_joint_goal curobo_ros_interfaces/srv/PlanJointGoal \
  "{group_name: left_arm, waypoints: [{positions: [0, 0, 0, 0, 0, 0, 0]}]}"

# 单臂多路点
ros2 service call /plan_joint_goal curobo_ros_interfaces/srv/PlanJointGoal \
  "{group_name: left_arm, waypoints: [
     {positions: [0, 0, 0, 0, 0, 0, 0]},
     {positions: [0.2, 0, 0, 0.5, 0, 0, 0]}]}"

# 双臂关节（14 维）
ros2 service call /plan_joint_goal curobo_ros_interfaces/srv/PlanJointGoal \
  "{group_name: both_arms, waypoints: [
     {positions: [0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0]}]}"

# 单臂位姿（当前左臂末端；frame_id 可改成能变到 base_link 的坐标系）
ros2 service call /plan_pose_goal curobo_ros_interfaces/srv/PlanPoseGoal \
  "{group_name: left_arm, ee_link: left_arm_ee_link,
    waypoints: [
      {header: {frame_id: base_link},
       pose: {position: {x: 0.270, y: 0.755, z: 1.153},
              orientation: {x: 0.884, y: 0.031, z: 0.401, w: -0.239}}}
    ]}"

# 双臂位姿（waypoints 左、other_waypoints 右，等长；右臂姿态请换成实测可达值）
ros2 service call /plan_pose_goal curobo_ros_interfaces/srv/PlanPoseGoal \
  "{group_name: both_arms,
    waypoints: [
      {header: {frame_id: base_link},
       pose: {position: {x: 0.270, y: 0.755, z: 1.153},
              orientation: {x: 0.884, y: 0.031, z: 0.401, w: -0.239}}}
    ],
    other_waypoints: [
      {header: {frame_id: base_link},
       pose: {position: {x: 0.270, y: -0.755, z: 1.153},
              orientation: {x: 0.884, y: 0.031, z: 0.401, w: -0.239}}}
    ]}"

# IK
ros2 service call /compute_ik curobo_ros_interfaces/srv/ComputeIK \
  "{group_name: left_arm,
    pose: {header: {frame_id: base_link},
           pose: {position: {x: 0.270, y: 0.755, z: 1.153},
                  orientation: {x: 0.884, y: 0.031, z: 0.401, w: -0.239}}}}"
```

`visualize_goal_waypoints:=true`（debug 默认开）时，请求一到就往 `/curobo/goal_waypoints` 发目标和中间路点，规划失败也留着。左臂黄球、右臂品红球、中间路点白球，旁边是红 / 绿 / 蓝坐标轴。

## MPC

笛卡尔跟随。先发 `/curobo/target`，再开 `/follow_mpc`。每个周期：读关节 → 点云更新碰撞 → 写左右末端目标 → 滚动优化 → 发 `MpcPlan`。`execute_trajectory:=true` 时把这一拍的短轨迹异步下发给 `FollowJointTrajectory`（`start_delay=0`，下一拍顶替；取消 Goal 即停）。debug 默认会下发，deploy 默认只发 topic。

- 第一次 Goal 才创建 `ModelPredictiveControl`；第一次 `step` 会冷启动（较慢），之后热启动。
- MPC 只规划左右臂 14 个关节，腰/胸/头等关节在建模时就按当前值锁死。否则求解器会安排躯干配合，而躯干没有控制器跟随，手臂就会替它乱补。
- `mpc_control_period` 是 ROS tick 周期。debug / deploy 都是 `0.10` s。
- `mpc_optimization_dt` 是一条 `action_sequence` 盖住的时间，debug / deploy 都是 `0.14` s。内部 `interpolation_steps=4`，相邻命令点间隔是 `optimization_dt / 4`（`command_dt=0.035` s）。这两个参数不用写成同一个数。
- 求解器的速度 / 加速度沿用上一拍最后一个命令点，避免每拍都从静止重新起步。
- `mpc_velocity_scale` 拉长下发时间，跟手保持 `1.0`。小于 1 时每拍只走轨迹前半段。
- 默认 `mpc_run_ik:=false`，纯笛卡尔跟踪。远路点直接下发即可，不必拆成近路点。开 IK 会额外加一个关节空间吸引点，和笛卡尔代价拉扯，末端常停在半路。
- 某一侧 `has_*:=false`：该臂钉在当前末端，另一臂仍跟。`MpcTarget` 里缺省四元数是单位四元数，不能靠 `w: 1` 表示「没给姿态」。
- `timeout<=0` 一直跟到取消；`position_tolerance` / `orientation_tolerance` `<=0` 不因到位自动成功。取消 Goal 即停。
- 点云碰撞与 TrajOpt 相同，每个周期把当前点云打成一块 `cam` mesh 整表替换。

先跟当前左臂姿态，再把 xyz 换成远点（四元数保持可达姿态）：

```bash
ros2 topic pub -r 10 /curobo/target curobo_ros_interfaces/msg/MpcTarget "{
  has_left: true,
  has_right: false,
  left: {header: {frame_id: base_link}, pose: {
    position: {x: 0.270, y: 0.755, z: 1.153},
    orientation: {x: 0.884, y: 0.031, z: 0.401, w: -0.239}}}
}"
ros2 action send_goal /follow_mpc curobo_ros_interfaces/action/FollowMpc \
  "{timeout: 0.0, position_tolerance: 0.0, orientation_tolerance: 0.0}" --feedback
```

循环发三个已记录的左臂路点（1 s、1 s、5 s，10 Hz）：

```bash
python src/curobo_ros_python/scripts/pub_left_waypoints.py
```

`MpcPlan` 含 `success`、`message`、`solve_time`，以及 `left` / `right`（`target_valid`、误差、`used_target`、短时域 `trajectory`）。

对照命令和 `/joint_states`（仿真里速度常为 0，脚本会改用位置差分；仿真的 actual 往往就是插值后的命令）：

```bash
python src/curobo_ros_python/scripts/plot_mpc_joint_tracking.py --arm left --duration 15
```

图默认写到 `/tmp/mpc_joint_tracking.png`。

## 感知目标 → 规划

`scripts/obj_pose_to_plan.py` 等一帧 `/obj_poses`（`perception_msgs/Pose6DArray`），留下 `valid==1` 且 `label` 匹配的目标，变到 `base_link` 后取 **+Y 最大**（机器人左侧）的那个，加上偏移再调 `/plan_pose_goal`。默认 `label=5`（BOX），偏移 `(-0.4, 0, 0.10)` m。

```bash
python src/curobo_ros_python/scripts/obj_pose_to_plan.py --dry-run
python src/curobo_ros_python/scripts/obj_pose_to_plan.py --label 5 --group left_arm
```

## RViz

debug.yaml 默认打开。RViz Fixed Frame 用 `base_link` 或 `world`。

1. **Add → MarkerArray**，话题 `/lycus_motion_trajectory`（或 `/{robot}_motion_trajectory`）
   - TrajOpt：绿色末端路径
   - MPC：左青 / 右品红预测末端，大球是本周期实际目标
2. **Add → MarkerArray**，话题 `/curobo/goal_waypoints`：请求里的目标（失败也留）
3. **Add → MarkerArray**，话题 `/curobo/collision_world`：点云体素
4. MoveIt **MotionPlanning** 已听 `/display_planned_path` 时，会播整机关节预览

| 话题 | 消息 | 看见什么 |
|---|---|---|
| `/display_planned_path` | `DisplayTrajectory` | 整台机器人按关节角演轨迹 |
| `/lycus_motion_trajectory` | `MarkerArray` | 末端扫过的线 |
| `/curobo/goal_waypoints` | `MarkerArray` | 目标球 + RGB 轴 |
| `/curobo/collision_world` | `MarkerArray` | 规划用的点云障碍 |

`visualize_trajectory:=false` 不发前两个；`visualize_goal_waypoints:=false` 不发目标球；`visualize_mesh_obstacles:=false` 仍做点云碰撞，只是不发体素。deploy 默认都关。`follow_mpc` 开始时会清掉 TrajOpt 预览，结束后清掉 MPC 那组 marker。

## 点云碰撞

取最近一帧点云，查 TF `pointcloud_parent_frame ← pointcloud_child_frame`（debug / deploy 默认 `base_link ← camera_01_color_optical_frame`），变到规划系后 `Mesh.from_pointcloud`。`pointcloud_child_frame` 为空时用消息的 `header.frame_id`。头会动，所以每次现查 TF。

debug / deploy 的点云话题是 `/all_points`。节点在没读到 yaml 时的默认值仍是 `/camera/depth_registered/points` 和 `camera_depth_optical_frame`，正常 launch 不会走到那里。

- `use_pointcloud:=false` 或 `pointcloud_topic` 为空：按空场景规划
- `filter_pointcloud:=true` 才做距离滤波和体素下采样（默认关）
- `require_pointcloud:=true`：没有点云 / TF 则失败；`false` 则告警后 TrajOpt 按空场景、MPC 沿用上一环境

## 和 MoveIt 版的差异

- 规划器是 cuRobo，不是 OMPL/PILZ。
- 双臂在同一套模型里规划；下发给控制器的只有规划组那 7/14 维。
- 单臂请求会钉住另一臂。
- `plan_pose_goal` 在 `both_arms` + `other_waypoints` 时两臂笛卡尔目标同时优化。
- 点云障碍从 TF 取当前头部位姿，不是写死外参。

## 常见问题

- **服务很久才返回**：正常。TrajOpt warmup 和第一次规划更慢；MPC 第一次 `follow_mpc` 的第一拍会冷启动。
- **`尚未收到 /joint_states`**：先起机器人状态发布。
- **Action 不可用 / 轨迹没执行**：控制器没起来，或 `execute_trajectory:=false`。deploy 默认就是 false，只发 `/curobo/mpc_plan` 和 `/curobo/planned_trajectory`。
- **`import curobo` / `No module named torch`**：launch 没找到 cuRobo 的 `.venv`。激活该环境，或设置 `curobo_python` / `CUROBO_PYTHON`。
- **zsh 里 source setup.bash 找不到 setup.sh**：改用 `setup.zsh`。
- **IK / 规划失败**：看 `message`。会区分末端不可达、自碰、场景碰撞、关节限位。Lycus 末端不要用单位四元数当目标。
- **点云没进规划**：要有 TF `camera_01_color_optical_frame → base_link`，以及 `/all_points` 有数据。
- **RViz 没有线 / 没有体素 / 没有目标球**：确认 debug launch。TrajOpt 绿线要先成功规划一次；目标球在请求到达时就发，失败也留着；MPC 预览要先 `follow_mpc`。
- **未知机器人配置**：`robot` 目前只认 `lycus.yml` / `k100.yml`。

## Deploy 的机器人 TF

Deploy 默认独立启动 `robot_state_publisher`，读取所选 cuRobo robot YAML 的同一份 URDF，订阅 deploy 配置中的真实 `joint_states_topic`，发布 `/tf` 和 `/tf_static`，无需 MoveIt。不会启动假关节状态或控制器。已有其他机器人 TF 发布者时，追加 `publish_robot_tf:=false`；`robot_urdf:=/absolute/path/robot.urdf` 可显式指定 TF 模型，应与规划器保持一致。

若光学系未定义在 URDF，且没有相机驱动等节点发布该外参，可启用静态 TF：

```bash
ros2 launch curobo_ros_python curobo_motion_planning_deploy.launch.py \
  publish_camera_tf:=true camera_parent_frame:=camera_01_link \
  camera_frame:=camera_01_color_optical_frame \
  camera_xyz:="0 0 0" camera_rpy:="-1.5707963267948966 0 -1.5707963267948966"
```

上面使用标准相机本体系（X 前、Y 左、Z 上）到光学系（X 右、Y 下、Z 前）的旋转，平移暂设零；实际安装有偏移或轴向不同时须填写标定结果。相机静态 TF 默认不发布，避免与相机驱动重复。点云可读取该 TF 树；规划器另通过关节反馈更新躯干/头部固定几何。
