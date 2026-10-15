"""调试 launch：先读 config yaml，命令行显式传入的参数才覆盖。"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from curobo_ros_python.launch_utils import conda_lib_env, resolve_python

# 默认空：不传则用 yaml，yaml 没有则用节点 _PARAM_DEFAULTS。
_OVERRIDE_ARGS = (
    ("robot", None, "覆盖 yaml 的 robot，如 k100.yml"),
    ("scene_model", None, None),
    ("warmup_iterations", int, None),
    ("execute_trajectory", bool, None),
    ("switch_controllers", bool, None),
    ("visualize_trajectory", bool, "false 则不发布 DisplayTrajectory / TrajOpt 绿线 / MPC 预览"),
    ("compute_ee_paths", bool, "false 则不算末端绿线 FK"),
    ("visualize_mesh_obstacles", bool, "false 则不发布 /curobo/collision_world"),
    ("visualize_goal_waypoints", bool, "false 则不发布 /curobo/goal_waypoints"),
    ("trajopt_velocity_scale", float, None),
    ("mpc_velocity_scale", float, None),
    ("interpolation_dt", float, None),
    ("use_cuda_graph", bool, None),
    ("joint_states_topic", None, None),
    ("base_frame", None, None),
    ("use_pointcloud", bool, "false 则关闭点云处理，按空场景规划"),
    ("pointcloud_topic", None, "点云话题；传空字符串以外的值才覆盖 yaml"),
    ("pointcloud_pitch", float, None),
    ("pointcloud_max_points", int, None),
    ("pointcloud_min_range", float, None),
    ("pointcloud_max_range", float, None),
    ("filter_pointcloud", bool, "true 时对点云做距离滤波和体素下采样"),
    ("require_pointcloud", bool, None),
    ("pointcloud_parent_frame", None, "点云 TF 父坐标系，规划前查 TF"),
    ("pointcloud_child_frame", None, "点云所在坐标系；空则用消息 header.frame_id"),
)


def _config_path(name: str) -> str:
    share = get_package_share_directory("curobo_ros_python")
    if os.path.isabs(name):
        return name
    return os.path.join(share, "config", name)


def _parse_override(raw: str, typ):
    if typ is bool:
        return raw.lower() in ("true", "1", "yes")
    if typ is int:
        return int(raw)
    if typ is float:
        return float(raw)
    return raw


def _cli_overrides(context) -> dict:
    extra = {}
    for name, typ, _desc in _OVERRIDE_ARGS:
        raw = LaunchConfiguration(name).perform(context).strip()
        if not raw:
            continue
        extra[name] = _parse_override(raw, typ)
    return extra


def _launch_setup(context, *args, **kwargs):
    python_exe = resolve_python(LaunchConfiguration("curobo_python").perform(context))
    config_file = _config_path(LaunchConfiguration("config").perform(context))
    params = [config_file]
    extra = _cli_overrides(context)
    if extra:
        params.append(extra)
    return [
        Node(
            package="curobo_ros_python",
            executable="curobo_motion_planning_node",
            name="curobo_motion_planning_node",
            output="screen",
            prefix=python_exe,
            additional_env=conda_lib_env(python_exe),
            parameters=params,
        )
    ]


def generate_launch_description():
    args = [
        DeclareLaunchArgument(
            "config",
            default_value="debug.yaml",
            description="share/curobo_ros_python/config 下的 yaml，或绝对路径",
        ),
        DeclareLaunchArgument(
            "curobo_python",
            default_value="",
            description="带 torch 的 Python 3.10；空则用 cuRobo .venv / CUROBO_PYTHON",
        ),
    ]
    for name, _typ, desc in _OVERRIDE_ARGS:
        kw = {"default_value": ""}
        if desc:
            kw["description"] = desc + "；空则用 yaml / 节点默认值"
        else:
            kw["description"] = "空则用 yaml / 节点默认值"
        args.append(DeclareLaunchArgument(name, **kw))
    args.append(OpaqueFunction(function=_launch_setup))
    return LaunchDescription(args)
