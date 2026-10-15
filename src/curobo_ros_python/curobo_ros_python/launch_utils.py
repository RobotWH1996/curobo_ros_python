"""解析带 torch/curobo 的 Python 3.10。供 launch 使用，不能 import torch。"""

from __future__ import annotations

import os
import subprocess


def resolve_python(explicit: str) -> str:
    """ros2 本身是系统 Python；节点必须用带 torch 的 3.10。"""
    candidates = []
    explicit = (explicit or "").strip()
    if explicit:
        candidates.append(os.path.expanduser(explicit))
    env_py = os.environ.get("CUROBO_PYTHON", "").strip()
    if env_py:
        candidates.append(os.path.expanduser(env_py))
    candidates.append(
        os.path.expanduser("~/wh/curobo/curobo/.venv/bin/python")
    )
    candidates.append(os.path.expanduser("~/miniconda3/envs/curobo310_test/bin/python"))
    conda = os.environ.get("CONDA_PREFIX", "")
    if conda:
        candidates.append(os.path.join(conda, "bin", "python"))

    seen = []
    for path in candidates:
        if not path or path in seen:
            continue
        seen.append(path)
        if not (os.path.isfile(path) and os.access(path, os.X_OK)):
            continue
        try:
            ver = subprocess.check_output(
                [path, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            continue
        if ver == "3.10":
            return path
    raise RuntimeError(
        "找不到 Python 3.10（需要能 import torch/curobo）。"
        "请使用 cuRobo 的 uv 环境 ~/wh/curobo/curobo/.venv，"
        "或设置 launch 参数 curobo_python / 环境变量 CUROBO_PYTHON。"
        f" 已尝试: {seen}"
    )


def conda_lib_env(python_exe: str) -> dict:
    conda_prefix = os.path.dirname(os.path.dirname(os.path.abspath(python_exe)))
    extra_env = {}
    lib_dir = os.path.join(conda_prefix, "lib")
    if os.path.isdir(lib_dir):
        extra_env["LD_LIBRARY_PATH"] = lib_dir + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
    return extra_env
