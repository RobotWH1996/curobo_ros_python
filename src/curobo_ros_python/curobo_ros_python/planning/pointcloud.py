"""把 ROS PointCloud2 转成规划用的 xyz（不依赖 cuRobo）。"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from sensor_msgs.msg import PointCloud2, PointField

from curobo_ros_python.planning.geom import transform_points


def _xyz_from_cloud(msg: PointCloud2) -> np.ndarray:
    n = int(msg.width) * int(msg.height)
    if n <= 0 or not msg.data:
        return np.zeros((0, 3), dtype=np.float32)
    offsets = {f.name: f.offset for f in msg.fields}
    dtypes = {f.name: f.datatype for f in msg.fields}
    for name in ("x", "y", "z"):
        if name not in offsets:
            raise ValueError(f"PointCloud2 缺少字段 {name}，现有: {list(offsets)}")
        if dtypes[name] != PointField.FLOAT32:
            raise ValueError(f"PointCloud2 字段 {name} 不是 float32")
    dtype = np.dtype(
        {
            "names": ["x", "y", "z"],
            "formats": ["<f4", "<f4", "<f4"],
            "offsets": [offsets["x"], offsets["y"], offsets["z"]],
            "itemsize": int(msg.point_step),
        }
    )
    arr = np.frombuffer(msg.data, dtype=dtype, count=n)
    return np.stack((arr["x"], arr["y"], arr["z"]), axis=1)


def _filter_and_downsample(
    pts: np.ndarray,
    min_range: float,
    max_range: float,
    pitch: float,
    max_points: int,
) -> np.ndarray:
    if pts.size == 0:
        return pts.reshape(0, 3)
    pts = np.asarray(pts, dtype=np.float32)
    finite = np.isfinite(pts).all(axis=1)
    dist = np.linalg.norm(pts, axis=1)
    mask = finite & (dist > float(min_range)) & (dist < float(max_range))
    pts = pts[mask]
    if pts.shape[0] == 0:
        return pts
    pitch = max(float(pitch), 1e-4)
    keys = np.floor(pts / pitch).astype(np.int32)
    keys -= keys.min(axis=0)
    hashed = (
        keys[:, 0].astype(np.int64) * 73856093
        ^ keys[:, 1].astype(np.int64) * 19349663
        ^ keys[:, 2].astype(np.int64) * 83492791
    )
    _, idx = np.unique(hashed, return_index=True)
    pts = pts[np.sort(idx)]
    max_points = max(int(max_points), 1)
    if pts.shape[0] > max_points:
        sel = np.linspace(0, pts.shape[0] - 1, max_points).astype(np.int64)
        pts = pts[sel]
    return pts


def cloud_to_base_points(
    msg: PointCloud2,
    translation: Optional[Sequence[float]],
    quat_wxyz: Optional[Sequence[float]],
    min_range: float,
    max_range: float,
    pitch: float,
    max_points: int,
    filter_cloud: bool = False,
) -> Tuple[np.ndarray, int]:
    raw = _xyz_from_cloud(msg)
    n_raw = int(raw.shape[0])
    pts = _filter_and_downsample(raw, min_range, max_range, pitch, max_points) if filter_cloud else raw
    if translation is not None and quat_wxyz is not None:
        pts = transform_points(pts, translation, quat_wxyz)
    return pts, n_raw
