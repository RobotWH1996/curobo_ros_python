"""规划失败日志要标出碰撞落在哪个路点，以及相撞的两侧。"""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from curobo.types import JointState
from curobo_ros_python.planning.algorithms.trajopt import (
    CuroboPlanner,
    format_collision_pair,
    segment_collision_labels,
    trajectory_sample_indices,
    trajectory_sample_label,
    waypoint_collision_label,
)


class CollisionFailureLogTest(unittest.TestCase):
    def test_labels_distinguish_start_via_and_goal(self):
        self.assertEqual(
            [waypoint_collision_label(i, 4) for i in range(4)],
            ["初始点", "路点1", "路点2", "目标点"],
        )
        self.assertEqual(waypoint_collision_label(1, 2), "目标点")

    def test_segment_labels_follow_the_failed_leg(self):
        self.assertEqual(segment_collision_labels(0, 1), ("初始点", "目标点"))
        self.assertEqual(segment_collision_labels(0, 3), ("初始点", "路点1"))
        self.assertEqual(segment_collision_labels(1, 3), ("路点1", "路点2"))
        self.assertEqual(segment_collision_labels(2, 3), ("路点2", "目标点"))

    def test_trajectory_sample_names_the_open_interval(self):
        self.assertEqual(trajectory_sample_label(0, 10, "初始点", "路点1"), "初始点")
        self.assertEqual(trajectory_sample_label(9, 10, "初始点", "路点1"), "路点1")
        self.assertEqual(
            trajectory_sample_label(4, 10, "路点1", "目标点"),
            "路点1到目标点的轨迹第5/10点",
        )

    def test_pair_text_names_both_sides(self):
        self.assertEqual(format_collision_pair("self", "left_link5", "left_link7"), "自碰撞 left_link5 / left_link7")
        self.assertEqual(format_collision_pair("environment", "left_gripper_left_jaw", "environment"), "环境碰撞 left_gripper_left_jaw / environment")
        self.assertEqual(trajectory_sample_indices(3), [0, 1, 2])
        self.assertEqual(trajectory_sample_indices(0), [])
        self.assertEqual(len(trajectory_sample_indices(40)), 8)


class _Hit:
    def __init__(self, kind, first, second):
        self.kind = kind
        self.first = first
        self.second = second


class CollisionDetailTest(unittest.TestCase):
    def _planner(self):
        planner = object.__new__(CuroboPlanner)
        planner._mesh_checker = SimpleNamespace(skip_fcl_checks=False)
        planner._motion = None
        planner.planner = SimpleNamespace(last_ik_result=None)

        def hit(self, state):
            value = float(state.position.detach().reshape(-1)[0].item())
            if value < -0.2:
                return _Hit("environment", "left_gripper_left_jaw", "environment")
            if value > 0.8:
                return _Hit("self", "left_link5", "left_link7")
            return None

        planner._collision_hit = hit.__get__(planner, CuroboPlanner)
        return planner

    def test_start_collision_names_the_pair(self):
        current = JointState.from_position(torch.tensor([[-1.0, 0.0]]), joint_names=["a", "b"])
        message = self._planner()._with_collision_detail("规划失败: 碰撞", None, current)
        self.assertIn("初始点碰撞: 环境碰撞 left_gripper_left_jaw / environment", message)

    def test_interior_sample_names_the_failed_leg(self):
        current = JointState.from_position(torch.tensor([[0.0, 0.0]]), joint_names=["a", "b"])
        result = SimpleNamespace(
            js_solution=JointState.from_position(
                torch.tensor([[0.2, 0.0], [0.3, 0.0], [-0.4, 0.0], [0.1, 0.0]]),
                joint_names=["a", "b"],
            ),
            interpolated_trajectory=None,
            seed_rank=None,
        )
        message = self._planner()._with_collision_detail(
            "规划失败: segment 1 failed", result, current, failed_index=1, n_goals=3,
        )
        self.assertIn("路点1到路点2的轨迹第3/4点碰撞: 环境碰撞 left_gripper_left_jaw / environment", message)
        self.assertNotIn("初始点碰撞", message)


class ReviewRegressionTest(unittest.TestCase):
 def planner(self):
  p=object.__new__(CuroboPlanner);p._mesh_checker=SimpleNamespace(skip_fcl_checks=False);p._motion=None
  p.planner=SimpleNamespace(last_ik_result=None);p._fcl_candidate_ms=12.
  p._collision_hit=Mock(return_value=None)
  return p
 def test_ik_failure_does_not_claim_requested_goal_collides(self):
  p=self.planner();candidate=JointState.from_position(torch.tensor([[.5]]),joint_names=['q'])
  p.planner.last_ik_result=object();p._ik_collision_state=Mock(return_value=candidate)
  p._collision_hit=Mock(return_value=SimpleNamespace(kind='environment',first='wrist',second='environment'))
  message=p._with_collision_detail('IK不可达',None,None)
  self.assertIn('目标点的IK候选碰撞',message)
  self.assertNotIn('；目标点碰撞',message)
 def test_failure_probe_does_not_pollute_ik_filter_timing(self):
  p=self.planner();current=JointState.from_position(torch.tensor([[.5]]),joint_names=['q'])
  def hit(state):p._fcl_candidate_ms+=7.;return None
  p._collision_hit=hit;p._with_collision_detail('failed',None,current)
  self.assertEqual(p._fcl_candidate_ms,12.)
 def test_failed_trajectory_end_is_labeled_candidate(self):
  p=self.planner();result=SimpleNamespace(js_solution=JointState.from_position(torch.tensor([[0.],[1.]]),joint_names=['q']),seed_rank=None)
  p._collision_hit=lambda state:SimpleNamespace(kind='environment',first='wrist',second='environment') if state.position.item()>.5 else None
  message=p._with_collision_detail('未到达目标',result,None)
  self.assertIn('目标点的失败轨迹末端候选碰撞',message)
  self.assertNotIn('；目标点碰撞',message)

 def test_failed_waypoint_uses_previous_segment_endpoint(self):
  p=self.planner();p._with_collision_detail=Mock(return_value='failed');p._fail=Mock()
  initial=JointState.from_position(torch.tensor([[0.]]),joint_names=['q'])
  previous=JointState.from_position(torch.tensor([[0.],[.4]]),joint_names=['q'])
  result=SimpleNamespace(success=torch.tensor([False]),status='IK failed',failed_index=1,
                         failed_result=None,segment_trajectories=[previous])
  p._from_waypoint_result(result,initial,[],3)
  start=p._with_collision_detail.call_args.args[2]
  self.assertAlmostEqual(float(start.position.item()),.4,places=6)


if __name__ == "__main__":
    unittest.main()
