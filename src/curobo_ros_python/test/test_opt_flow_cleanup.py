"""CPU regressions for request validation, FK reuse and trajectory expansion."""
from types import SimpleNamespace as NS
import ast
from pathlib import Path
from unittest.mock import Mock, patch
import unittest
import torch

from curobo.types import JointState
from curobo_ros_python.planning.algorithms.trajopt import CuroboPlanner
from curobo_ros_python.planning.groups import apply_robot, both_arm_ee_links


class OptFlowCleanupTest(unittest.TestCase):
    def setUp(self):
        apply_robot('lycus.yml')
        self.left, self.right = both_arm_ee_links()
        self.p = CuroboPlanner.__new__(CuroboPlanner)
        self.p.planner = NS(joint_names=['left', 'right'])
        self.p._finetune_attempts = 0
        self.p._sync_locked_arm = Mock(side_effect=AssertionError('must reject before solver setup'))

    def test_both_arms_missing_target_rejected_before_planning(self):
        for goals in [[{self.left: object()}],
                      [{self.left: object(), self.right: object()}, {self.right: object()}]]:
            result = self.p.plan_pose('both_arms', None, goals)
            self.assertFalse(result.success)
        self.p._sync_locked_arm.assert_not_called()

    def test_ros_input_rejects_before_pose_conversion(self):
        from curobo_ros_python.planning.algorithms import trajopt
        source = Path(trajopt.__file__).parents[2] / 'motion_planning_node.py'
        method = next(n for n in ast.walk(ast.parse(source.read_text()))
                      if isinstance(n, ast.FunctionDef) and n.name == '_build_pose_waypoints')
        namespace = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
        result = namespace['_build_pose_waypoints'](
            NS(_raw_goal_items=lambda request: []), NS(group_name='both_arms'), [], '')
        self.assertIsNone(result[0])
        self.assertIn('both_arms', result[2])

    def test_complete_dual_goal_reaches_planner(self):
        motion = NS(tool_frames=[self.left, self.right], plan_pose=Mock(return_value='result'), last_timing={})
        self.p._sync_locked_arm = Mock(return_value=motion)
        self.p._active = lambda state: state
        self.p.fk_tool_poses = Mock(side_effect=AssertionError('unnecessary FK'))
        self.p._goal_for_ee_poses = Mock(return_value=('goal', []))
        self.p._from_single_result = Mock(return_value='passed')
        result = self.p.plan_pose('both_arms', None, [{self.left: object(), self.right: object()}])
        self.assertEqual(result, 'passed')
        motion.plan_pose.assert_called_once()

    def test_single_arm_multiple_goals_share_current_fk(self):
        motion = NS(tool_frames=[self.left, self.right],
                    plan_cartesian_waypoints=Mock(return_value='result'), last_timing={})
        self.p._sync_locked_arm = Mock(return_value=motion)
        self.p._active = lambda state: state
        self.p.fk_tool_poses = Mock(return_value={self.right: 'held pose'})
        self.p._goal_for_ee_poses = Mock(return_value=('goal', []))
        self.p._from_waypoint_result = Mock(return_value='passed')
        result = self.p.plan_pose('left_arm', None, [{self.left: object()}, {self.left: object()}])
        self.assertEqual(result, 'passed')
        self.p.fk_tool_poses.assert_called_once()
        for call in self.p._goal_for_ee_poses.call_args_list:
            self.assertEqual(call.kwargs['current_fk'], {self.right: 'held pose'})

    def test_complete_dual_goals_do_not_compute_unused_fk(self):
        self.p._motion = NS(tool_frames=[self.left, self.right])
        self.p.fk_tool_poses = Mock(side_effect=AssertionError('unnecessary FK'))
        module = 'curobo_ros_python.planning.algorithms.trajopt'
        with patch(module + '.squeeze_pose', side_effect=lambda value: value), \
             patch(module + '.GoalToolPose') as goal_type:
            self.p._goal_for_ee_poses(None, {self.left: object(), self.right: object()})
            goal_type.from_poses.assert_called_once()
        self.p.fk_tool_poses.assert_not_called()

    def test_expansion_preserves_all_derivatives_and_joint_order(self):
        self.p.planner.joint_names = ['right', 'left']
        partial = JointState.from_position(torch.tensor([[0.], [1.]]), joint_names=['left'])
        for field, value in [('velocity', 1.), ('acceleration', 2.), ('jerk', 3.)]:
            setattr(partial, field, torch.full_like(partial.position, value))
        hold = JointState.from_position(torch.tensor([[.4, 0.]]), joint_names=['right', 'left'])
        result = self.p._expand_to_reference(partial, hold)
        torch.testing.assert_close(result.position[:, 0], torch.tensor([.4, .4]))
        for field in ['velocity', 'acceleration', 'jerk']:
            torch.testing.assert_close(getattr(result, field)[:, 1:], getattr(partial, field))
            torch.testing.assert_close(getattr(result, field)[:, :1], torch.zeros(2, 1))

    def test_unchanged_cloud_reuses_fcl_across_groups_and_new_cloud_rebuilds(self):
        p = self.p
        p._planners = {group: NS(scene_collision_checker=NS(), update_world=Mock())
                       for group in ['left_arm', 'right_arm']}
        p._cloud_loaded = {}; p._cloud_sig = None; p._cloud_vertices = None
        p._cloud_fcl_sig = None; p._cloud_fcl_scene = None
        p._mesh_scenes = {}; p._mesh_checker = NS(prepare_scene=Mock(side_effect=lambda scene: {}))
        p._info = Mock()
        points = [[.5, 0, 1], [.52, 0, 1]]
        p.update_world_from_points(points, group_name='left_arm')
        p.update_world_from_points(points, group_name='right_arm')
        self.assertEqual(p._mesh_checker.prepare_scene.call_count, 1)
        self.assertIs(p._mesh_scenes['left_arm'], p._mesh_scenes['right_arm'])
        p.update_world_from_points([[.6, 0, 1]], group_name='left_arm')
        self.assertEqual(p._mesh_checker.prepare_scene.call_count, 2)


if __name__ == '__main__':
    unittest.main()
