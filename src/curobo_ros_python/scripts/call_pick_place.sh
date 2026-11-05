#!/usr/bin/env bash
# Reuse the Action client; preserve the caller's ROS_DOMAIN_ID and node mode.
set -eo pipefail

usage() {
    cat <<'EOF'
用法：./src/curobo_ros_python/scripts/call_pick_place.sh pick|place|pick_place [客户端参数]

示例：
  ./src/curobo_ros_python/scripts/call_pick_place.sh pick --dry-run
  ./src/curobo_ros_python/scripts/call_pick_place.sh pick --label 11 --track-id 2 --dry-run
  ./src/curobo_ros_python/scripts/call_pick_place.sh pick --output /tmp/pick_result.json
  ./src/curobo_ros_python/scripts/call_pick_place.sh place --output /tmp/place_result.json
  ./src/curobo_ros_python/scripts/call_pick_place.sh pick_place --output /tmp/pick_place_result.json
  ./src/curobo_ros_python/scripts/call_pick_place.sh pick_place --goal-file /path/to/my_goal.yaml

默认配置：src/curobo_ros_python/config/pick_place.yaml
一份配置包含 pick/place；调用单个接口时自动提取对应任务。
pick / pick_place 只从感知话题覆盖抓取 tool_pose，预抓取、抬升、工件尺寸和夹爪仍用该文件。
也支持独立请求格式 {task: ..., timeout: ...}，JSON/YAML 均可。
默认 pick/pick_place 从 /obj_poses 读取本次目标的位置和姿态，默认 label=11（ROD），
可用 --label、--track-id、--topic 选择目标；未指定 ID 时选最高置信度。
tool_pose.header.frame_id 和时间戳直接取感知消息，位置和姿态保留在该坐标系。
规划节点负责通过 TF 转到规划坐标系；相对位姿继续相对 tool_pose。
place 的放置目标继续读 YAML，不从抓取感知目标推断放置位置。
--no-perception 改用 YAML 中 tool_pose 的位姿及其 header.frame_id。

--dry-run 会读取感知、校验并打印，不提交任务；正常调用是否运动由规划节点的
debug/deploy（execute_trajectory）配置决定，不由本脚本修改。
脚本使用节点的当前 /joint_states，不设置机器人初始关节角。
ROS_DOMAIN_ID 沿用当前终端；可用 CUROBO_PYTHON 指定 Python。
EOF
}

case "${1:-}" in
    pick|place|pick_place) action="$1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ "$repo_root" != "/" && ! -f "$repo_root/install/setup.bash" ]]; do
    repo_root="$(dirname -- "$repo_root")"
done
ros_setup="/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
workspace_setup="$repo_root/install/setup.bash"
if [[ ! -f "$ros_setup" || ! -f "$workspace_setup" ]]; then
    echo "找不到 ROS/工程环境，请先构建工作空间并检查 ROS_DISTRO。" >&2
    exit 1
fi
source "$ros_setup"
source "$workspace_setup"

python_bin="${CUROBO_PYTHON:-$repo_root/../curobo/.venv/bin/python}"
if [[ -z "${CUROBO_PYTHON:-}" && ! -x "$python_bin" ]]; then
    python_bin="$(command -v python3)"
fi
printf 'Action: /%s | ROS_DOMAIN_ID: %s\n' "$action" "${ROS_DOMAIN_ID:-0}" >&2
exec "$python_bin" "$repo_root/src/curobo_ros_python/scripts/pick_place_client.py" \
    --action "$action" \
    --goal-file "$repo_root/src/curobo_ros_python/config/pick_place.yaml" "$@"
