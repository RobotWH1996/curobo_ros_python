"""Read a perception reference pose together with its original ROS frame."""
import math
import time

import rclpy
from perception_msgs.msg import Pose6DArray
from rclpy.qos import qos_profile_sensor_data

from curobo_ros_python.planning.geom import quat_mul_wxyz, transform_pose_wxyz


def unit_quaternion(values):
    """Read and normalize a ROS xyzw mapping, returning wxyz."""
    q = tuple(float(values.get(k, 0.0)) for k in ("w", "x", "y", "z"))
    norm = math.sqrt(sum(v * v for v in q))
    if not math.isfinite(norm) or norm < 1e-8:
        raise ValueError("目标/工件相对位姿的四元数必须有限且非零")
    return tuple(v / norm for v in q)


def read_pick_target(node, task, settings):
    """Select a fresh object and retain its source frame, stamp and coordinates."""
    topic = settings.get("topic", "/obj_poses")
    timeout = float(settings.get("wait_timeout", 10.0))
    label = int(settings.get("label", 11))
    track = settings.get("track_id")
    track = None if track is None else int(track)
    minimum = float(settings.get("min_score", 0.0))
    if not topic or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("感知话题不能为空，wait_timeout 必须为有限正秒数")
    if not math.isfinite(minimum) or not 0.0 <= minimum <= 1.0:
        raise ValueError("min_score 必须在 [0, 1] 内")
    for name in ("pick_pose", "prepick_in_target", "lift_in_target"):
        if not isinstance(task.get(name), dict):
            raise ValueError(f"{name} 必须是相对于 tool_pose 的位姿")
        unit_quaternion(task[name].get("orientation", {}))
    latest = []
    subscription = node.create_subscription(
        Pose6DArray, topic, lambda msg: latest.__setitem__(slice(None), [msg]),
        qos_profile_sensor_data,
    )
    deadline = time.monotonic() + timeout
    reason = "尚未收到新消息"
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=min(0.1, max(0.0, deadline - time.monotonic())))
            if not latest:
                continue
            message = latest[-1]
            candidates = []
            for pose in message.poses:
                if pose.valid != 1 or pose.label != label or (track is not None and pose.track_id != track):
                    continue
                if not all(math.isfinite(v) for v in (pose.x, pose.y, pose.z, pose.qw, pose.qx, pose.qy, pose.qz, pose.score)) or pose.score < minimum:
                    continue
                try:
                    unit_quaternion(dict(zip(("w", "x", "y", "z"), (pose.qw, pose.qx, pose.qy, pose.qz))))
                except ValueError:
                    continue
                candidates.append(pose)
            if not candidates:
                reason = f"没有有效目标 label={label} track_id={track} min_score={minimum}"
                continue
            selected = max(candidates, key=lambda p: (p.score, -p.track_id))
            source = message.header.frame_id
            if not source.strip().lstrip("/"):
                reason = "感知消息 header.frame_id 为空，无法确定坐标系"
                continue
            position = (float(selected.x), float(selected.y), float(selected.z))
            q = unit_quaternion(dict(zip(("w", "x", "y", "z"), (selected.qw, selected.qx, selected.qy, selected.qz))))
            orientation = dict(zip("xyzw", (q[1], q[2], q[3], q[0])))
            task["tool_pose"] = {
                "header": {"frame_id": source,
                           "stamp": {"sec": message.header.stamp.sec,
                                     "nanosec": message.header.stamp.nanosec}},
                "pose": {"position": dict(zip("xyz", position)), "orientation": orientation},
            }
            report = dict(topic=topic, label=label, track_id=int(selected.track_id),
                          score=float(selected.score), frame_id=source,
                          tool_position=list(position), tool_orientation=orientation)
            in_base = bool(task.get("in_base_frame", False))
            for name in ("pick_pose", "prepick_in_target", "lift_in_target"):
                relative = task[name]
                offset = tuple(float(relative["position"][k]) for k in "xyz")
                rotation = unit_quaternion(relative["orientation"])
                if in_base:
                    point = tuple(position[i] + offset[i] for i in range(3))
                    quat = quat_mul_wxyz(rotation, q)
                else:
                    point, quat = transform_pose_wxyz(position, q, offset, rotation)
                report[name] = {"frame_id": source, "position": list(point),
                                "orientation": dict(zip("xyzw", quat[1:] + quat[:1]))}
            return report
        raise RuntimeError(f"读取 {topic} 超时（{timeout:g}s）：{reason}；未提交 Action")
    finally:
        node.destroy_subscription(subscription)
