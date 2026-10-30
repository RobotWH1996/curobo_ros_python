"""部署 launch：读 config/deploy.yaml，只订状态/点云并发布规划轨迹。"""

import os
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from curobo_ros_python.launch_utils import conda_lib_env, resolve_python, robot_urdf_path


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
    nodes = [
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
    publish_tf = LaunchConfiguration("publish_robot_tf").perform(context).lower()
    publish_camera = LaunchConfiguration("publish_camera_tf").perform(context).lower()
    if publish_tf not in {"true", "false"} or publish_camera not in {"true", "false"}:
        raise RuntimeError("publish_robot_tf / publish_camera_tf 必须为 true 或 false")
    if publish_tf == "true" or publish_camera == "true":
        config = yaml.safe_load(Path(config_file).read_text())
        settings = config['curobo_motion_planning_node']['ros__parameters']
        urdf = Path(robot_urdf_path(python_exe, robot or settings.get('robot', 'lycus.yml'),
                                  LaunchConfiguration('robot_urdf').perform(context))).read_text()
        if publish_tf == "true":
            nodes.append(Node(
                package="robot_state_publisher", executable="robot_state_publisher",
                name="robot_state_publisher", output="screen",
                parameters=[{'robot_description': ParameterValue(urdf, value_type=str),
                             'use_sim_time': settings.get('use_sim_time', False)}],
                remappings=[('joint_states', settings.get('joint_states_topic', '/joint_states'))],
            ))
        if publish_camera == "true":
            parent = LaunchConfiguration('camera_parent_frame').perform(context).lstrip('/')
            child = LaunchConfiguration('camera_frame').perform(context).lstrip('/')
            links = {link.attrib['name'] for link in ET.fromstring(urdf).findall('link')}
            if parent not in links or not child or child in links:
                raise RuntimeError('相机静态 TF 父系须存在于 URDF，子系须不在 URDF 中，避免重复发布')
            xyz = LaunchConfiguration('camera_xyz').perform(context).split()
            rpy = LaunchConfiguration('camera_rpy').perform(context).split()
            if any(len(v) != 3 for v in (xyz, rpy)) or not all(math.isfinite(float(v)) for v in xyz+rpy):
                raise RuntimeError('camera_xyz / camera_rpy 各须提供三个有限数，单位 m / rad')
            nodes.append(Node(
                package='tf2_ros', executable='static_transform_publisher',
                name='camera_optical_static_tf', output='screen',
                arguments=['--x', xyz[0], '--y', xyz[1], '--z', xyz[2],
                           '--roll', rpy[0], '--pitch', rpy[1], '--yaw', rpy[2],
                           '--frame-id', parent, '--child-frame-id', child],
            ))
    return nodes



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
            DeclareLaunchArgument('publish_robot_tf', default_value='true',
                                  description='发布真实关节状态对应的机器人 TF；已有发布者时关闭'),
            DeclareLaunchArgument('robot_urdf', default_value='',
                                  description='TF 使用的 URDF；空则从规划器 robot YAML 解析'),
            DeclareLaunchArgument('publish_camera_tf', default_value='false',
                                  description='发布 camera_parent_frame 到光学系的静态外参'),
            DeclareLaunchArgument('camera_parent_frame', default_value='camera_01_link'),
            DeclareLaunchArgument('camera_frame', default_value='camera_01_color_optical_frame'),
            DeclareLaunchArgument('camera_xyz', default_value='0 0 0',
                                  description='光学系在父系中的平移，m；启用前核对标定值'),
            DeclareLaunchArgument('camera_rpy', default_value='-1.5707963267948966 0 -1.5707963267948966',
                                  description='光学系在父系中的 roll pitch yaw，rad'),
            OpaqueFunction(function=_launch_setup),
        ]
    )
