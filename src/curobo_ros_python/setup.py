from setuptools import find_packages, setup
import os
from glob import glob

package_name = "curobo_ros_python"

_config_files = [
    p for p in glob("config/*.yml") + glob("config/*.yaml") if os.path.isfile(p)
]
_data_files = [
    ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
    ("share/" + package_name, ["package.xml"]),
    (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
]
if _config_files:
    _data_files.append((os.path.join("share", package_name, "config"), _config_files))

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=_data_files,
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="victor",
    maintainer_email="victor@todo.todo",
    description="cuRobo ROS 2 motion planning services (robot selected by the robot YAML parameter)",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "curobo_motion_planning_node = curobo_ros_python.main:main",
        ],
    },
)
