"""解析带 torch/curobo 的 Python 3.10。供 launch 使用，不能 import torch。"""

from __future__ import annotations

import os
import subprocess


def resolve_python(explicit: str) -> str:
    """ros2 本身是系统 Python；节点必须用带 torch 的 3.10。

    查找顺序：launch 参数、``CUROBO_PYTHON``、已激活的虚拟环境或 conda、
    再从本文件和当前目录向上找 cuRobo 仓库里的 ``.venv``。
    """
    seen: list[str] = []
    for path in _python_candidates(explicit):
        if not path or path in seen:
            continue
        seen.append(path)
        if _usable_curobo_python(path):
            return path
    raise RuntimeError(
        "找不到能 import torch/curobo 的 Python 3.10。"
        "请激活 cuRobo 仓库的 .venv，"
        "或设置 launch 参数 curobo_python / 环境变量 CUROBO_PYTHON。"
        f" 已尝试: {seen}"
    )


def conda_lib_env(python_exe: str) -> dict:
    """把虚拟环境的 lib 放在系统 CUDA 前面，并补上 PyTorch 缺的两个符号。

    Jetson 的 cuSOLVER 11.6 能正常建句柄，但没有
    ``cusolverDnXsyevBatched``。PyPI 的 11.7 有这两个符号，
    在 CUDA 12.6 驱动上 ``cusolverDnCreate`` 会失败。
    ``libcusolver_batched_stub.so`` 只提供这两个符号，其余仍用系统库。
    """
    prefix = os.path.dirname(os.path.dirname(os.path.abspath(python_exe)))
    extra_env = {}
    lib_dir = os.path.join(prefix, "lib")
    if os.path.isdir(lib_dir):
        extra_env["LD_LIBRARY_PATH"] = lib_dir + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
    stub = os.path.join(lib_dir, "libcusolver_batched_stub.so")
    if os.path.isfile(stub):
        preload = os.environ.get("LD_PRELOAD", "").strip()
        extra_env["LD_PRELOAD"] = stub if not preload else stub + " " + preload
    return extra_env


def _python_candidates(explicit: str) -> list[str]:
    candidates: list[str] = []
    explicit = (explicit or "").strip()
    if explicit:
        candidates.append(os.path.expanduser(explicit))
    env_py = os.environ.get("CUROBO_PYTHON", "").strip()
    if env_py:
        candidates.append(os.path.expanduser(env_py))
    virtual_env = os.environ.get("VIRTUAL_ENV", "").strip()
    if virtual_env:
        candidates.append(os.path.join(os.path.expanduser(virtual_env), "bin", "python"))
    candidates.extend(_nearby_curobo_pythons())
    conda = os.environ.get("CONDA_PREFIX", "").strip()
    if conda:
        candidates.append(os.path.join(os.path.expanduser(conda), "bin", "python"))
    return candidates


def _nearby_curobo_pythons() -> list[str]:
    """向上找 cuRobo 仓库的 ``.venv/bin/python``，不写死本机目录。"""
    found: list[str] = []
    roots = (os.path.dirname(os.path.realpath(__file__)), os.getcwd())
    seen_dirs: set[str] = set()
    for root in roots:
        current = os.path.abspath(root)
        while current not in seen_dirs:
            seen_dirs.add(current)
            repo = current if _is_curobo_repo(current) else os.path.join(current, "curobo")
            if _is_curobo_repo(repo):
                found.append(os.path.join(repo, ".venv", "bin", "python"))
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent
    return found


def _is_curobo_repo(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "pyproject.toml")) and os.path.isfile(
        os.path.join(path, "curobo", "__init__.py")
    )


def _usable_curobo_python(path: str) -> bool:
    if not (os.path.isfile(path) and os.access(path, os.X_OK)):
        return False
    script = (
        "import sys\n"
        "if sys.version_info[:2] != (3, 10):\n"
        "    raise SystemExit(1)\n"
        "import torch, curobo\n"
    )
    try:
        subprocess.check_call(
            [path, "-c", script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return True


def robot_urdf_path(python_exe: str, robot: str, explicit: str = "") -> str:
    """Resolve the planner model using its Python environment, without loading CUDA."""
    if explicit.strip():
        path = os.path.abspath(os.path.expanduser(explicit.strip()))
    else:
        script = (
            "import os,sys\n"
            "from curobo.config_io import load_yaml\n"
            "from curobo.content import get_robot_configs_path, get_assets_path\n"
            "cfg=load_yaml(os.path.join(str(get_robot_configs_path()), sys.argv[1]))\n"
            "print(os.path.join(str(get_assets_path()), cfg['robot_cfg']['kinematics']['urdf_path']))\n"
        )
        result = subprocess.run([python_exe, "-c", script, robot], check=True,
                                capture_output=True, text=True, timeout=30,
                                env=dict(os.environ, **conda_lib_env(python_exe)))
        path = result.stdout.strip()
    if not os.path.isfile(path):
        raise RuntimeError(f"找不到 robot_state_publisher 使用的 URDF：{path}")
    return path
