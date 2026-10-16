"""部署 launch：读 config/deploy.yaml，只订状态/点云并发布规划轨迹。"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from curobo_ros_python.launch_utils import conda_lib_env, resolve_python


def _config_path(name: str) -> str:
    share = get_package_share_directory("curobo_ros_python")
    if os.path.isabs(name):
        return name
    return os.path.join(share, "config", name)


def _launch_setup(context, *args, **kwargs):
    python_exe = resolve_python(LaunchConfiguration("curobo_python").perform(context))
    config_file = _config_path(LaunchConfiguration("config").perform(context))
    params = [config_file]
    robot = LaunchConfiguration("robot").perform(context).strip()
    if robot:
        params.append({"robot": robot})
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
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config",
                default_value="deploy.yaml",
                description="share/curobo_ros_python/config 下的 yaml，或绝对路径",
            ),
            DeclareLaunchArgument(
                "robot",
                default_value="",
                description="覆盖 yaml 里的 robot，如 k100.yml；空则用 config",
            ),
            DeclareLaunchArgument(
                "curobo_python",
                default_value="",
                description="带 torch 的 Python 3.10；空则用已激活环境或旁边的 cuRobo .venv",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
