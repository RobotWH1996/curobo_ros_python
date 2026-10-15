"""四元数与刚体变换。四元数一律 wxyz。"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def rpy_to_quat_wxyz(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    """ROS 固定轴 RPY：R = Rz(yaw) Ry(pitch) Rx(roll)，单位弧度。"""
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    return (
        float(cr * cp * cy + sr * sp * sy),
        float(sr * cp * cy - cr * sp * sy),
        float(cr * sp * cy + sr * cp * sy),
        float(cr * cp * sy - sr * sp * cy),
    )


def quat_mul_wxyz(
    a: Sequence[float], b: Sequence[float]
) -> Tuple[float, float, float, float]:
    aw, ax, ay, az = (float(a[0]), float(a[1]), float(a[2]), float(a[3]))
    bw, bx, by, bz = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def quat_angle_wxyz(a: Sequence[float], b: Sequence[float]) -> float:
    """两姿态夹角（弧度）。q 与 -q 视为同一姿态。"""
    dot = abs(
        float(a[0]) * float(b[0])
        + float(a[1]) * float(b[1])
        + float(a[2]) * float(b[2])
        + float(a[3]) * float(b[3])
    )
    return float(2.0 * np.arccos(min(1.0, max(0.0, dot))))


def quat_wxyz_to_R(q: Sequence[float]) -> np.ndarray:
    w, x, y, z = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def rotate_vec_wxyz(
    q: Sequence[float], v: Sequence[float]
) -> Tuple[float, float, float]:
    out = quat_wxyz_to_R(q) @ np.asarray(v, dtype=np.float32)
    return float(out[0]), float(out[1]), float(out[2])


def transform_pose_wxyz(
    t_ab: Sequence[float],
    q_ab: Sequence[float],
    t_bc: Sequence[float],
    q_bc: Sequence[float],
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
    """T_ac = T_ab * T_bc。四元数均为 wxyz。"""
    px, py, pz = rotate_vec_wxyz(q_ab, t_bc)
    return (
        (px + float(t_ab[0]), py + float(t_ab[1]), pz + float(t_ab[2])),
        quat_mul_wxyz(q_ab, q_bc),
    )


def transform_points(
    pts: np.ndarray,
    translation: Sequence[float],
    quat_wxyz: Sequence[float],
) -> np.ndarray:
    if pts.size == 0:
        return pts.reshape(0, 3)
    r = quat_wxyz_to_R(quat_wxyz)
    t = np.asarray(translation, dtype=np.float32).reshape(1, 3)
    return pts.astype(np.float32, copy=False) @ r.T + t
