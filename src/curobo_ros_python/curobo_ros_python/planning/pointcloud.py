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


def crop_bounds(
    points: np.ndarray,
    bounds_min: Optional[Sequence[float]],
    bounds_max: Optional[Sequence[float]],
) -> np.ndarray:
    """只保留 base_link 轴对齐盒子里的点。盒子外的点不参与后面的栅格化。"""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] == 0 or bounds_min is None or bounds_max is None:
        return pts
    lo = np.asarray(bounds_min, dtype=np.float32).reshape(3)
    hi = np.asarray(bounds_max, dtype=np.float32).reshape(3)
    keep = np.all((pts >= lo) & (pts <= hi), axis=1)
    return pts[keep]


def cloud_to_base_points(
    msg: PointCloud2,
    translation: Optional[Sequence[float]],
    quat_wxyz: Optional[Sequence[float]],
    min_range: float,
    max_range: float,
    pitch: float,
    max_points: int,
    filter_cloud: bool = False,
    bounds_min: Optional[Sequence[float]] = None,
    bounds_max: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, int]:
    raw = _xyz_from_cloud(msg)
    n_raw = int(raw.shape[0])
    pts = _filter_and_downsample(raw, min_range, max_range, pitch, max_points) if filter_cloud else raw
    if translation is not None and quat_wxyz is not None:
        pts = transform_points(pts, translation, quat_wxyz)
    return crop_bounds(pts, bounds_min, bounds_max), n_raw


def exclude_boxes_from_param(value) -> Optional[np.ndarray]:
    """解析固定剔除盒。每 6 个数一只：xmin, ymin, zmin, xmax, ymax, zmax，单位米，base_link。"""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.replace(";", " ").replace(",", " ")
        parts = [p for p in text.split() if p]
        if not parts:
            return None
        nums = [float(p) for p in parts]
    else:
        nums = [float(p) for p in value]
        if not nums:
            return None
    if len(nums) < 6:
        return None
    if len(nums) % 6 != 0:
        raise ValueError(f"pointcloud_exclude_boxes 需要 6 的倍数，收到 {len(nums)} 个数")
    boxes = np.asarray(nums, dtype=np.float32).reshape(-1, 6)
    lo = np.minimum(boxes[:, :3], boxes[:, 3:])
    hi = np.maximum(boxes[:, :3], boxes[:, 3:])
    return np.concatenate((lo, hi), axis=1)


def snap_voxel_dims(dims: Sequence[float], pitch: float, max_cells: int = 4_000_000) -> Tuple[float, float, float]:
    """把盒子边长收成 pitch 的整数倍，和 cuRobo 格子索引对齐。"""
    pitch = float(pitch)
    if pitch <= 0.0:
        raise ValueError(f"pointcloud_pitch 必须 > 0，收到 {pitch}")
    snapped = []
    cells = 1
    for raw in dims:
        n = int(round(float(raw) / pitch))
        if n < 2:
            raise ValueError(
                f"体素盒子边长 {float(raw):.3f} m 相对 pitch={pitch:.3f} m 不足 2 格"
            )
        snapped.append(n * pitch)
        cells *= n
    if cells > int(max_cells):
        raise ValueError(
            f"体素格子共 {cells} 个，超过 {int(max_cells)}。"
            f"加大 pointcloud_pitch 或缩小 pointcloud_voxel_dims"
        )
    return snapped[0], snapped[1], snapped[2]


def points_to_esdf(
    points: np.ndarray,
    center: Sequence[float],
    dims: Sequence[float],
    pitch: float,
    empty_distance: float = 5.0,
) -> Tuple[np.ndarray, int]:
    """点云写入固定格子，返回 ESDF（米）。内部为负，外部为正。

    数组形状 (nx, ny, nz)，C 顺序，z 变化最快，与 cuRobo 的体素下标一致。
    没有任何点落在盒子里时，整张表都是 empty_distance。
    """
    from scipy.ndimage import distance_transform_edt

    pitch = float(pitch)
    dims_s = snap_voxel_dims(dims, pitch)
    center_a = np.asarray(center, dtype=np.float64).reshape(3)
    shape = tuple(int(round(d / pitch)) for d in dims_s)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    occupied = np.zeros(shape, dtype=bool)
    n_inside = 0
    if pts.shape[0] > 0:
        offset = np.asarray(shape, dtype=np.float64) * pitch * 0.5
        ijk = np.floor((pts - center_a + offset) / pitch).astype(np.int64)
        inside = (
            (ijk[:, 0] >= 0)
            & (ijk[:, 0] < shape[0])
            & (ijk[:, 1] >= 0)
            & (ijk[:, 1] < shape[1])
            & (ijk[:, 2] >= 0)
            & (ijk[:, 2] < shape[2])
        )
        n_inside = int(np.count_nonzero(inside))
        if n_inside:
            hit = ijk[inside]
            occupied[hit[:, 0], hit[:, 1], hit[:, 2]] = True
    if n_inside == 0:
        esdf = np.full(shape, float(empty_distance), dtype=np.float32)
        return esdf, 0
    dist_to_occ = distance_transform_edt(~occupied)
    dist_to_free = distance_transform_edt(occupied)
    esdf = ((dist_to_occ - dist_to_free) * pitch).astype(np.float32)
    return esdf, n_inside


def drop_boxes(points: np.ndarray, boxes: Optional[np.ndarray]) -> np.ndarray:
    """丢掉落在固定盒子里的点。盒子是录包里的夹爪，不随当前手臂动。"""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if boxes is None or pts.shape[0] == 0:
        return pts
    keep = np.ones(pts.shape[0], dtype=bool)
    for box in np.asarray(boxes, dtype=np.float32).reshape(-1, 6):
        inside = np.all((pts >= box[:3]) & (pts <= box[3:]), axis=1)
        keep &= ~inside
    return pts[keep]
