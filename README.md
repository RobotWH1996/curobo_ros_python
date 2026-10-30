# curobo_ros_python

ROS 2 Humble 双臂规划节点，支持 Lycus（`lycus.yml`）和 K100（`k100.yml`）。OPT 通过 Service 规划整条轨迹，MPC 通过 Action 跟踪末端目标。两者在同一节点中互斥使用 GPU。

## 环境安装与构建

推荐目录结构：

```text
workspace/
  curobo/                 # 已配置 CUDA/Torch 的 Python 3.10 .venv
  curobo_ros_python/      # 本仓库
  Lycus_ros/             # 机器人驱动、控制器和描述（按部署需要）
```

前提是 NVIDIA GPU、可工作的 CUDA/Torch、ROS 2 Humble，以及 `../curobo/.venv/bin/python` 能导入 cuRobo。安装脚本复用该环境，不负责安装驱动、Torch 或 cuRobo 本体。

在本仓库根目录执行：

```bash
# 首次安装：ROS 系统依赖、FCL Python 依赖、构建所有接口及节点包
bash scripts/setup_environment.sh --system-deps

# 已安装系统依赖：只补 Python 依赖并构建
bash scripts/setup_environment.sh

# 只检查，既不安装也不构建
bash scripts/setup_environment.sh --check
```

`CUROBO_PYTHON=/absolute/path/bin/python` 可指定 Python；`ROS_SETUP=/opt/ros/humble/setup.bash` 可指定 ROS 环境。Python 依赖安装优先使用目标环境的 pip，没有 pip 时尝试 `uv pip --python`。二者都没有时脚本会给出错误，不自动更换 Python 或安装 Torch。

OPT 必需依赖为 `python-fcl==0.7.0.11`、`trimesh>=4.0`，见 `requirements-fcl.txt`。脚本会检查 CUDA、ROS 导入、Lycus URDF/配置、FCL 独立忽略表及相机连杆名称。

Jetson 若已配置 `libcusolver_batched_stub.so`，脚本与 launch 会使用现有兼容库；脚本不生成兼容库。遇到 Torch/cuSOLVER 导入错误，先修复 cuRobo 环境。

手动构建：

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
- Lycus 全身 FCL 使用包内 `planning/lycus_fcl_collision.yaml/self_collision_ignore` 独立完整表。此表检查左右 link5/link7；GPU 暂时忽略这两对以避免粗球误报。代码没有五七特例，也没有独立腕部模块。
- 其他机器人尚未提供独立 FCL 表，沿用各自机器人配置中的配对。
- FCL 在 CPU 上检查 mesh 表面相交，覆盖 IK 候选、输入关节目标和实际输出终点；不覆盖整条轨迹，不是连续碰撞或封闭物体包含检测。中间路径及拼接检查仍使用球模型。
- URDF 没有 collision 几何的连杆会在启动日志列出，不进行 mesh 检查。

日志示例：

```text
OPT: IK 候选已启用全机器人 FCL 自碰撞和环境碰撞筛选
FCL IK筛选: 检查=2, 通过=2, 剔除=0, FCL耗时=...ms; 环境=...
FCL 输出终点: 通过, 自碰撞通过; 环境=..., 耗时=...ms
FCL 关节目标[1/3]: 通过, ...
```

候选 FCL 耗时已包含在 IK 阶段计时中，不能重复相加；`check` 是底层规划器原有验收计时，不代表 FCL。`wall` 是 ROS 请求处理到规划完成的实际耗时，含状态同步、目标准备和输出整理，不含后续轨迹执行等待。

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

辅助脚本位于 `src/curobo_ros_python/scripts/`，包括目标位姿规划、MPC 路点发布和关节跟踪记录；使用各脚本 `--help` 查看参数。

## 回归验证

加载 ROS 环境并使用规划 Python，在仓库根目录运行：

```bash
export PYTHONPATH="$PWD/src/curobo_ros_python:$PYTHONPATH"
../curobo/.venv/bin/python -m unittest discover \
  -s src/curobo_ros_python/test -p 'test_mesh_candidate_filter.py'
../curobo/.venv/bin/python -m unittest discover \
  -s src/curobo_ros_python/test -p 'test_opt_flow_cleanup.py'
```

Jetson 直接运行 Python 时，按现有环境设置 `LD_PRELOAD`；正常 launch 会通过 `launch_utils` 加载已有兼容库。测试不发运动命令。

## 代码工作区约定

后续默认直接在远程 `ai:/home/ai/wh/curobo_ros_python` 修改和验证，本机为同步副本。提交时必须同时包含主流程与新增 FCL/固定状态模块、策略 YAML 和测试，避免 Git 更新后仅留下无调用的孤立文件。构建产物、日志、虚拟环境和离线基准数据不属于核心源码上传范围。
