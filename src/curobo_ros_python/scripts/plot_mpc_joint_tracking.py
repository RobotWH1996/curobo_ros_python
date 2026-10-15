#!/usr/bin/env python3
"""录 MPC 下发轨迹 vs /joint_states，画各关节位置/速度/加速度，评估命令流是否平滑。

三个取舍，看图前先知道：

1. 仿真的 /joint_states 速度字段常年全零。整段都是零时按「未上报」处理，改用位置数值微分，
   并在标题里注明。
2. 命令曲线按 send_replace 的真实语义截断（第 i 拍只画到第 i+1 拍到达为止），两拍之间的
   空窗**不填假点**，用 NaN 断开，另外在底部时间条上标出来。填零会凭空造出锯齿。
3. MoveIt fake controller 把下发轨迹插值后原样当 /joint_states 发回，所以仿真里
   actual≈commanded 是构造出来的。这张图在仿真里只能判断「命令流平不平滑」，
   真机跟踪质量要上硬件再量。
"""

from __future__ import annotations

import argparse
import os
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState

from curobo_ros_interfaces.msg import MpcPlan

LYCUS_LEFT = [f"AR5-5_08L-W4C1C9-ZY2_joint_{i}" for i in range(1, 8)]
LYCUS_RIGHT = [f"AR5-5_08R-W4C1C9-ZY2_joint_{i}" for i in range(1, 8)]
SHORT = [f"j{i}" for i in range(1, 8)]

C_CMD = "#d97706"
C_ACT = "#2563eb"
C_GAP = "#dc2626"


def stamp_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def duration_sec(dur) -> float:
    return float(dur.sec) + float(dur.nanosec) * 1e-9


def pick_indices(names: Sequence[str], want: Sequence[str]) -> Optional[List[int]]:
    lookup = {n: i for i, n in enumerate(names)}
    if any(n not in lookup for n in want):
        return None
    return [lookup[n] for n in want]


class Recorder(Node):
    def __init__(self, joints: Sequence[str], plan_topic: str, js_topic: str):
        super().__init__("plot_mpc_joint_tracking")
        self.joints = list(joints)
        self.act_t: List[float] = []
        self.act_q: List[np.ndarray] = []
        self.act_v: List[np.ndarray] = []
        # 每拍: (轨迹起始时刻, 相对时间, 位置)
        self.plans: List[Tuple[float, np.ndarray, np.ndarray]] = []
        qos = QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(JointState, js_topic, self._on_js, qos)
        self.create_subscription(MpcPlan, plan_topic, self._on_plan, 50)

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_js(self, msg: JointState):
        idx = pick_indices(msg.name, self.joints)
        if idx is None:
            return
        t = stamp_sec(msg.header.stamp) or self._now()
        self.act_t.append(t)
        self.act_q.append(np.array([msg.position[i] for i in idx], dtype=np.float64))
        if msg.velocity and len(msg.velocity) == len(msg.name):
            self.act_v.append(np.array([msg.velocity[i] for i in idx], dtype=np.float64))
        else:
            self.act_v.append(np.full(len(idx), np.nan))

    def _on_plan(self, msg: MpcPlan):
        arm = msg.left if msg.left.target_valid else msg.right
        traj = arm.trajectory
        if not traj.points or not traj.joint_names:
            return
        idx = pick_indices(traj.joint_names, self.joints)
        if idx is None:
            return
        t0 = stamp_sec(traj.header.stamp) or stamp_sec(msg.header.stamp) or self._now()
        rel = np.array([duration_sec(p.time_from_start) for p in traj.points])
        q = np.array(
            [[p.positions[j] for j in idx] for p in traj.points], dtype=np.float64
        )
        self.plans.append((t0, rel, q))


def effective_command(
    plans: List[Tuple[float, np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[float, float]]]:
    """按 send_replace 语义截断成实际生效的命令流。

    空窗用 NaN 断开而不是补零点，避免画出并不存在的锯齿；空窗区间单独返回。
    """
    if not plans:
        return np.zeros(0), np.zeros((0, 0)), []
    dof = plans[0][2].shape[1]
    ts: List[float] = []
    qs: List[np.ndarray] = []
    gaps: List[Tuple[float, float]] = []
    nan_row = np.full(dof, np.nan)
    for i, (t0, rel, q) in enumerate(plans):
        t_end = plans[i + 1][0] if i + 1 < len(plans) else t0 + rel[-1]
        last_t = None
        for k in range(len(rel)):
            t = t0 + rel[k]
            if t > t_end + 1e-9:
                break
            ts.append(t)
            qs.append(q[k])
            last_t = t
        if last_t is None:
            continue
        if last_t < t_end - 1e-4:
            gaps.append((last_t, t_end))
            ts.append(np.nan)
            qs.append(nan_row)
    return np.asarray(ts), np.vstack(qs), gaps


def smooth_rows(y: np.ndarray, window: int) -> np.ndarray:
    """滑动平均。必须按边缘值补边，用 mode='same' 的零填充会在首尾造出巨大的假导数。"""
    if window <= 1 or y.shape[0] < window:
        return y
    half = window // 2
    padded = np.pad(y, ((half, window - 1 - half), (0, 0)), mode="edge")
    kernel = np.ones(window) / float(window)
    out = np.empty_like(y)
    for c in range(y.shape[1]):
        out[:, c] = np.convolve(padded[:, c], kernel, mode="valid")
    return out


def robust_ylim(*arrays: np.ndarray) -> Optional[Tuple[float, float]]:
    """按 0.5/99.5 分位定纵轴，避免个别毛刺把有用信号压成一条线。"""
    vals = [a[np.isfinite(a)].ravel() for a in arrays if a.size]
    vals = [v for v in vals if v.size]
    if not vals:
        return None
    merged = np.concatenate(vals)
    lo, hi = np.percentile(merged, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if hi - lo < 1e-12:
        lo, hi = lo - 1e-6, hi + 1e-6
    margin = 0.15 * (hi - lo)
    return float(lo - margin), float(hi + margin)


def diff_segments(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """对被 NaN 切成若干段的序列逐段求导，段太短则留 NaN。"""
    out = np.full_like(y, np.nan)
    finite = np.isfinite(t)
    n = len(t)
    i = 0
    while i < n:
        if not finite[i]:
            i += 1
            continue
        j = i
        while j < n and finite[j]:
            j += 1
        if j - i >= 2:
            out[i:j] = np.gradient(y[i:j], t[i:j], axis=0)
        i = j
    return out


def summarize(
    plans: List[Tuple[float, np.ndarray, np.ndarray]],
    gaps: List[Tuple[float, float]],
) -> Tuple[List[str], List[str]]:
    """返回 (中文日志行, 英文图注行)。图注只能用 ASCII，DejaVu 没有中文字形。"""
    if len(plans) < 2:
        return [], []
    period = np.diff(np.array([p[0] for p in plans]))
    span = np.array([p[1][-1] for p in plans])
    duty = float(np.sum(np.minimum(span[:-1], period)) / np.sum(period))
    zh = [
        f"拍周期 {period.mean()*1e3:.1f} ms (min {period.min()*1e3:.1f}, "
        f"max {period.max()*1e3:.1f})",
        f"命令跨度 {span.mean()*1e3:.1f} ms，占空比 {duty*100:.0f}%",
    ]
    en = [
        f"tick {period.mean()*1e3:.1f} ms (min {period.min()*1e3:.1f}, "
        f"max {period.max()*1e3:.1f})",
        f"command span {span.mean()*1e3:.1f} ms, duty {duty*100:.0f}%",
    ]
    if gaps:
        g = np.array([b - a for a, b in gaps])
        zh.append(
            f"空窗 {len(gaps)} 段，平均 {g.mean()*1e3:.1f} ms，最大 {g.max()*1e3:.1f} ms"
        )
        en.append(
            f"{len(gaps)} gaps, mean {g.mean()*1e3:.1f} ms, max {g.max()*1e3:.1f} ms"
        )
    else:
        zh.append("无空窗：命令首尾相接")
        en.append("no gaps: commands are contiguous")
    return zh, en


def plot(
    t_act, q_act, v_act, a_act, t_cmd, q_cmd, v_cmd, a_cmd, gaps, out_path, title, caption
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = q_act.shape[1]
    base = min(
        float(np.nanmin(t_act)) if t_act.size else np.inf,
        float(np.nanmin(t_cmd)) if t_cmd.size else np.inf,
    )
    ta = t_act - base
    tc = t_cmd - base
    t_max = float(np.nanmax(ta)) if ta.size else 1.0

    fig = plt.figure(figsize=(16, 2.0 * n + 2.0))
    gs = fig.add_gridspec(n + 1, 3, height_ratios=[1.0] * n + [0.7], hspace=0.25)
    fig.suptitle(title, fontsize=12)

    series = [
        ("q (rad)", q_act, q_cmd),
        ("dq (rad/s)", v_act, v_cmd),
        ("ddq (rad/s^2)", a_act, a_cmd),
    ]
    for r in range(n):
        for c, (unit, act, cmd) in enumerate(series):
            ax = fig.add_subplot(gs[r, c])
            for g0, g1 in gaps:
                ax.axvspan(g0 - base, g1 - base, color=C_GAP, alpha=0.10, lw=0)
            if ta.size:
                ax.plot(ta, act[:, r], color=C_ACT, lw=1.6, alpha=0.55, label="actual")
            if tc.size:
                ax.plot(tc, cmd[:, r], color=C_CMD, lw=0.9, label="commanded")
            if c > 0:
                lim = robust_ylim(
                    act[:, r] if ta.size else np.zeros(0),
                    cmd[:, r] if tc.size else np.zeros(0),
                )
                if lim is not None:
                    ax.set_ylim(*lim)
            ax.set_ylabel(f"{SHORT[r]} {unit}", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)
            if r == 0 and c == 0:
                ax.legend(loc="upper right", fontsize=7)
            if r < n - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("t (s)", fontsize=8)

    ax = fig.add_subplot(gs[n, :])
    for g0, g1 in gaps:
        ax.axvspan(g0 - base, g1 - base, color=C_GAP, alpha=0.25, lw=0)
    if tc.size:
        finite = np.isfinite(tc)
        ax.plot(tc[finite], np.zeros(finite.sum()), "|", color=C_CMD, ms=10, mew=1.0)
    ax.set_xlim(0, t_max)
    ax.set_yticks([])
    ax.set_xlabel("t (s)", fontsize=8)
    ax.set_title(
        "command coverage: ticks (orange) and uncovered gaps (red)", fontsize=9
    )
    ax.tick_params(labelsize=7)

    fig.text(0.01, 0.004, caption, fontsize=8, va="bottom")
    fig.subplots_adjust(left=0.06, right=0.99, top=0.96, bottom=0.08)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def parse_args():
    p = argparse.ArgumentParser(description="画 MPC 下发 vs 实际关节轨迹（位置/速度/加速度）")
    p.add_argument("--duration", type=float, default=15.0, help="录制秒数，Ctrl-C 提前结束")
    p.add_argument("--arm", choices=("left", "right"), default="left")
    p.add_argument("--plan-topic", default="/curobo/mpc_plan")
    p.add_argument("--joint-states", default="/joint_states")
    p.add_argument(
        "--smooth",
        type=int,
        default=5,
        help="实测位置求导前的滑动平均窗口（采样点数），1 表示不平滑",
    )
    p.add_argument("--out", default="/tmp/mpc_joint_tracking.png")
    return p.parse_args()


def main():
    args = parse_args()
    joints = LYCUS_LEFT if args.arm == "left" else LYCUS_RIGHT
    rclpy.init()
    node = Recorder(joints, args.plan_topic, args.joint_states)
    log = node.get_logger()
    log.info(f"录制 {args.arm} 臂 {args.duration:.1f}s：{args.plan_topic} + {args.joint_states}")
    started = time.monotonic()
    try:
        while rclpy.ok() and (time.monotonic() - started) < args.duration:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass

    n_js, n_plan = len(node.act_t), len(node.plans)
    log.info(f"收到 joint_states={n_js} 帧，mpc_plan={n_plan} 拍")
    if n_js < 3:
        log.error("joint_states 太少，无法画图")
        node.destroy_node()
        rclpy.shutdown()
        return

    t_act = np.asarray(node.act_t)
    q_act = np.vstack(node.act_q)
    v_rep = np.vstack(node.act_v)

    # 仿真常年上报全零速度：整段都是零就当没上报，改用位置微分。
    reported = np.isfinite(v_rep).any() and float(np.nanmax(np.abs(v_rep))) > 1e-9
    q_for_diff = smooth_rows(q_act, args.smooth)
    if reported:
        v_act = v_rep
        vel_note = "actual dq: reported by /joint_states"
    else:
        v_act = np.gradient(q_for_diff, t_act, axis=0)
        vel_note = "actual dq: differentiated (/joint_states velocity was all-zero)"
    a_act = np.gradient(v_act, t_act, axis=0)

    t_cmd, q_cmd, gaps = effective_command(node.plans)
    if t_cmd.size:
        v_cmd = diff_segments(t_cmd, q_cmd)
        a_cmd = diff_segments(t_cmd, v_cmd)
    else:
        v_cmd = a_cmd = q_cmd

    stats_zh, stats_en = summarize(node.plans, gaps)
    for s in stats_zh:
        log.info(s)

    title = f"MPC {args.arm} arm - commanded vs actual joint trajectory ({args.duration:.0f}s)"
    caption = (
        f"{n_plan} plans, {n_js} joint_states | " + " | ".join(stats_en) + "\n"
        f"{vel_note}; smoothing window {args.smooth} samples. "
        "Commanded truncated at send_replace; gaps left blank (red), not zero-filled. "
        "Sim note: fake controller replays the command, so actual~commanded by construction."
    )
    path = plot(
        t_act, q_act, v_act, a_act, t_cmd, q_cmd, v_cmd, a_cmd, gaps, args.out, title, caption
    )
    log.info(f"已保存 {path}")
    node.destroy_node()
    rclpy.shutdown()
    print(path)


if __name__ == "__main__":
    main()
