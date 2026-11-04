"""Posture caches and tensor-only rotation branches retain existing semantics."""
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock
import torch
from curobo_ros_python.planning.algorithms.mpc import (
    DualArmMpc, _rotvec_wxyz, _stack_link_jacobians, _POSTURE_TABLES,
)


class PostureCacheTest(unittest.TestCase):
    def test_rotation_identity_sign_and_small_angle(self):
        q = torch.tensor([1., 0., 0., 0.])
        for target in (q, -q):
            torch.testing.assert_close(_rotvec_wxyz(q, target), torch.zeros(3))
        target = torch.tensor([1., 1e-9, 0., 0.])
        torch.testing.assert_close(_rotvec_wxyz(q, target), torch.tensor([2e-9, 0., 0.]))
        target = torch.tensor([2**-.5, 0., 0., 2**-.5])
        torch.testing.assert_close(_rotvec_wxyz(q, target), torch.tensor([0., 0., torch.pi/2]))

    def test_stack_link_jacobians_follows_requested_order(self):
        frames = ['left', 'right']
        tool = torch.zeros(2, 6, 3)
        tool[0, :, 0] = 1
        tool[1, :, 1] = 2
        right_only = _stack_link_jacobians(tool, frames, ['right'])
        torch.testing.assert_close(right_only, tool[1])
        both = _stack_link_jacobians(tool, frames, ['right', 'left'])
        torch.testing.assert_close(both[:6], tool[1])
        torch.testing.assert_close(both[6:], tool[0])
        self.assertIsNone(_stack_link_jacobians(tool, frames, ['missing']))
        self.assertIsNone(_stack_link_jacobians(tool, frames, []))
        bad = tool.clone()
        bad[0, 0, 0] = float('nan')
        self.assertIsNone(_stack_link_jacobians(bad, frames, ['left']))

    def test_targets_cached_and_posture_switch(self):
        m = DualArmMpc.__new__(DualArmMpc)
        name = next(n for n, v in _POSTURE_TABLES['l'].items() if v != 0)
        m.mpc = NS(joint_names=[name, 'unknown'], default_joint_state=NS(position=torch.tensor([.1, .2])))
        m._posture_targets = {}
        m.posture_name = 'l'
        q = torch.zeros(2)
        a = m._l_posture(q)
        self.assertIs(a, m._l_posture(q))
        torch.testing.assert_close(a, torch.tensor([_POSTURE_TABLES['l'][name], .2]))
        m.posture_name = 'zero'
        torch.testing.assert_close(m._l_posture(q), torch.tensor([0., .2]))
        m.posture_name = 'l'
        self.assertIs(a, m._l_posture(q))

    def test_costs_limits_and_weights_keep_buffers(self):
        m = DualArmMpc.__new__(DualArmMpc)
        cfg = NS(joint_limits=NS(position=torch.tensor([[-1., -2.], [1., 2.]])),
                 cspace_target_dof_weight=torch.ones(2), cspace_target_weight=torch.ones(1),
                 cspace_non_terminal_weight_factor=torch.ones(1))
        cost = NS(config=cfg)
        rollout = NS(get_cost_component_by_name=Mock(return_value=[cost]))
        get = Mock(return_value=[rollout, rollout])
        m.mpc = NS(core=NS(get_all_rollout_instances=get))
        m._posture_cost_cache = None
        m._posture_limits = {}
        ptr = cfg.cspace_target_weight.data_ptr()
        m._write_posture_weight(5.)
        m._write_nonterminal_factor(.5)
        self.assertTrue(m._write_dof_weight(torch.tensor([.2, .7])))
        limits = m._joint_position_limits(torch.device('cpu'), torch.float32)
        self.assertIs(limits, m._joint_position_limits(torch.device('cpu'), torch.float32))
        self.assertEqual(get.call_count, 1)
        self.assertEqual(ptr, cfg.cspace_target_weight.data_ptr())
        torch.testing.assert_close(cfg.cspace_target_weight, torch.tensor([5.]))
        torch.testing.assert_close(cfg.cspace_target_dof_weight, torch.tensor([.2, .7]))


if __name__ == '__main__':
    unittest.main()
