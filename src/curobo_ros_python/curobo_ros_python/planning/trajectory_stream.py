"""Append validated prefixes on one controller timeline without waiting for motion."""
import copy
import threading
from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory
from rclpy.duration import Duration
from curobo_ros_python.planning.trajectory import copy_joint_trajectory, trajectory_is_finite, _point_time


class TrajectoryStream:
    def __init__(self, executor):
        self.executor = executor
        self.stamp = (executor._node.get_clock().now() + Duration(seconds=.1)).to_msg()
        self._stamp_ns = self.stamp.sec * 1000000000 + self.stamp.nanosec
        self.lock = threading.RLock()
        self.error = ''
        self.fault = False
        self.closed = False
        self.cancelled = False
        self._sending = False
        self._revision = 0
        self._sent_end = None
        self._tail = None
        self._queued = None
        self._acks = set()
        self._ack_deadline_ns = 0
        self._results = set()
        self._handles = []

    def done(self):
        with self.lock:
            return self.closed and not (self._sending or self._queued or self._acks or self._results or self.fault)

    def append(self, commands):
        with self.lock:
            if self.closed or self.cancelled or self.error:
                raise RuntimeError(self.error or '轨迹流已结束')
            if not commands or len({group for group, _ in commands}) != len(commands):
                raise ValueError('轨迹流控制器列表无效')
            packed = {}
            for group, trajectory in commands:
                if not trajectory_is_finite(trajectory):
                    raise ValueError('轨迹流收到非有限或空轨迹')
                client = self.executor._clients.get(group)
                if client is None or not client.server_is_ready():
                    raise RuntimeError(f'轨迹流控制器不可用: {group}')
                packed[group] = copy.deepcopy(trajectory)
            ends = [_point_time(tr.points[-1]) for tr in packed.values()]
            if max(ends)-min(ends) > 1e-8:
                raise ValueError('手臂与夹爪轨迹必须共用时间轴')
            if self._tail is not None:
                if list(packed) != list(self._tail):
                    raise ValueError('轨迹流不能更换控制器')
                for group, old in self._tail.items():
                    new = packed[group]
                    if (new.joint_names != old.joint_names or len(new.points) < len(old.points)
                            or new.points[:len(old.points)] != old.points):
                        raise ValueError('续接轨迹修改了已经提交的前缀')
            self._tail = packed
            self._queued = packed
            self._pump()
            if self.error:
                raise RuntimeError(self.error)

    def _pump(self):
        if self._sending or self._acks or self.cancelled or self.error or self._queued is None:
            return
        elapsed = (self.executor._node.get_clock().now().nanoseconds-self._stamp_ns)*1e-9
        if self._sent_end is not None and elapsed+.05 >= self._sent_end:
            self._fail('后续规划或控制器确认已错过轨迹续接窗口')
            return
        commands, self._queued = self._queued, None
        self._revision += 1
        revision = self._revision
        self._sent_end = max(_point_time(tr.points[-1]) for tr in commands.values())
        now_ns = self.executor._node.get_clock().now().nanoseconds
        self._ack_deadline_ns = self._stamp_ns if revision == 1 else now_ns + 500000000
        self._sending = True
        try:
            for group, trajectory in commands.items():
                if self.cancelled:
                    break
                goal = FollowJointTrajectory.Goal()
                goal.trajectory = copy_joint_trajectory(trajectory)
                goal.trajectory.header.stamp = copy.deepcopy(self.stamp)
                future = self.executor._clients[group].send_goal_async(goal)
                self._acks.add(future)
                future.add_done_callback(lambda f, g=group, r=revision: self._ack(f, g, r))
        except Exception as exc:
            self._fail(f'轨迹流提交异常: {exc}', unknown=True)
        finally:
            self._sending = False
        self._pump()

    def _ack(self, future, group, revision):
        with self.lock:
            try:
                handle = future.result()
                if handle is None or not handle.accepted:
                    self._fail(f'控制器拒绝轨迹块: {group}')
                else:
                    self._handles.append(handle)
                    result = handle.get_result_async()
                    self._results.add(result)
                    result.add_done_callback(lambda f, g=group, r=revision: self._result(f, g, r))
                    if self.cancelled:
                        self.executor._cancel_handles([handle])
            except Exception as exc:
                self._fail(f'轨迹块确认异常: {exc}', unknown=True)
            finally:
                self._acks.discard(future)
            self._pump()

    def _result(self, future, group, revision):
        with self.lock:
            try:
                wrapped = future.result()
                status = None if wrapped is None else wrapped.status
                replacing = revision < self._revision and status == GoalStatus.STATUS_CANCELED
                stopping = self.cancelled and status in (
                    GoalStatus.STATUS_CANCELED, GoalStatus.STATUS_ABORTED, GoalStatus.STATUS_SUCCEEDED)
                if not (replacing or stopping):
                    error = self.executor._goal_error(wrapped)
                    if error:
                        self._fail(f'{group}: {error}', unknown=status not in (
                            GoalStatus.STATUS_CANCELED, GoalStatus.STATUS_ABORTED, GoalStatus.STATUS_SUCCEEDED))
            except Exception as exc:
                self._fail(f'轨迹块结果异常: {exc}', unknown=True)
            finally:
                self._results.discard(future)

    def _fail(self, message, unknown=False):
        self.error = self.error or message
        self.fault |= unknown
        if unknown:
            self.executor._execution_fault = True
        self.executor._node.get_logger().error(self.error)
        self.cancel()

    def poll(self):
        with self.lock:
            if self._acks and not self.cancelled:
                if self.executor._node.get_clock().now().nanoseconds >= self._ack_deadline_ns:
                    self._fail('轨迹块目标确认超时；迟到确认将被取消')

    def finish(self):
        with self.lock:
            self.closed = True
            self._pump()

    def cancel(self):
        with self.lock:
            self.cancelled = True
            self.closed = True
            self._queued = None
            self.executor._cancel_handles(list(self._handles))
