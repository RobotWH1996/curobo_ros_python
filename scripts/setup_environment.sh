#!/usr/bin/env bash
# Reuse an existing cuRobo environment; do not install Torch or GPU drivers.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
check_only=false
system_deps=false
skip_build=false
for arg in "$@"; do
  case "$arg" in
    --check) check_only=true ;;
    --system-deps) system_deps=true ;;
    --skip-build) skip_build=true ;;
    --help|-h) printf '%s\n' 'Usage: bash scripts/setup_environment.sh [--check] [--system-deps] [--skip-build]' 'CUROBO_PYTHON selects existing Python 3.10; ROS_SETUP selects ROS setup.bash.'; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$arg" >&2; exit 2 ;;
  esac
done
ros_setup="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
planning_python="${CUROBO_PYTHON:-$repo_root/../curobo/.venv/bin/python}"
[[ -f "$ros_setup" ]] || { printf 'Missing ROS setup: %s\n' "$ros_setup" >&2; exit 1; }
[[ -x "$planning_python" ]] || { printf 'Missing cuRobo Python: %s\n' "$planning_python" >&2; exit 1; }
if $system_deps && ! $check_only; then
  sudo apt-get update
  sudo apt-get install -y python3-colcon-common-extensions python3-pip python3-yaml \
    libfcl-dev liboctomap-dev libccd-dev ros-humble-robot-state-publisher \
    ros-humble-tf2-ros ros-humble-moveit-msgs ros-humble-control-msgs \
    ros-humble-controller-manager-msgs ros-humble-ros2launch
fi
# ROS-generated shell scripts may reference unset variables.
set +u
source "$ros_setup"
set -u
python_prefix="$("$planning_python" -c 'import sys; print(sys.prefix)')"
if [[ -f "$python_prefix/lib/libcusolver_batched_stub.so" ]]; then
  export LD_PRELOAD="$python_prefix/lib/libcusolver_batched_stub.so${LD_PRELOAD:+ $LD_PRELOAD}"
fi
export LD_LIBRARY_PATH="$python_prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$planning_python" -c 'import sys, torch, curobo; assert sys.version_info[:2] == (3, 10), sys.version; assert torch.cuda.is_available(), "CUDA is unavailable"; print("cuRobo/Torch/CUDA OK")'
if ! $check_only; then
  if "$planning_python" -m pip --version >/dev/null 2>&1; then
    "$planning_python" -m pip install -r "$repo_root/requirements-fcl.txt"
  elif command -v uv >/dev/null 2>&1; then
    uv pip install --python "$planning_python" -r "$repo_root/requirements-fcl.txt"
  else
    printf '%s\n' 'Target Python has no pip and uv is unavailable. Install pip into that environment or install uv, then retry.' >&2
    exit 1
  fi
fi
export PYTHONPATH="$repo_root/src/curobo_ros_python${PYTHONPATH:+:$PYTHONPATH}"
"$planning_python" - "$repo_root" <<'PY'
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
import fcl
import trimesh
import yaml
import rclpy
from curobo.config_io import load_yaml
from curobo.content import get_assets_path, get_robot_configs_path
root = Path(sys.argv[1])
cfg = load_yaml(str(Path(get_robot_configs_path()) / 'lycus.yml'))['robot_cfg']['kinematics']
urdf = Path(get_assets_path()) / cfg['urdf_path']
links = {node.get('name') for node in ET.parse(urdf).getroot().findall('link')}
missing = set(cfg['collision_link_names']) - links
assert not missing, f'URDF collision links missing: {missing}'
assert 'camera_01_link' in links, 'Update cuRobo to camera rename commit 1ebff3b or newer'
policy = root / 'src/curobo_ros_python/curobo_ros_python/planning/lycus_fcl_collision.yaml'
assert isinstance(yaml.safe_load(policy.read_text())['self_collision_ignore'], dict)
# Exercise the installed FCL binding, not only the Python import.
a = fcl.CollisionObject(fcl.Box(1., 1., 1.))
b = fcl.CollisionObject(fcl.Box(1., 1., 1.))
assert fcl.collide(a, b, fcl.CollisionRequest(), fcl.CollisionResult()) > 0
print('ROS/FCL/URDF/FCL policy OK; no motion commands sent')
PY
if ! $check_only && ! $skip_build; then
  cd "$repo_root"
  colcon build --symlink-install
fi
printf '%s\n' 'Environment check complete. Source install/setup.bash (zsh: setup.zsh) before launch.'
