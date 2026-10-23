"""Feedback-only jaws: mimic motion, shared copies, and stable graph buffers."""
from types import SimpleNamespace as NS
import unittest

import torch

from curobo_ros_python.planning.gripper import GripperStateUpdater


class GripperStateTest(unittest.TestCase):
    def setUp(self):
        self.updater = GripperStateUpdater.from_robot('lycus.yml')

    def model(self, transforms=None):
        fingers = self.updater._fingers
        if transforms is None:
            transforms = torch.stack([f.origin.clone() for f in fingers])
        kin = NS(fixed_transforms=transforms,
                 link_name_to_idx_map={f.child: i for i, f in enumerate(fingers)},
                 joint_map_type=torch.full((len(fingers),), -1))
        lock = NS(joint_names=list(self.updater.joint_names),
                  position=torch.zeros(2), velocity=torch.ones(2),
                  acceleration=torch.ones(2), jerk=torch.ones(2))
        return NS(config=NS(kinematics_config=kin), joint_names=['arm'], lock_jointstate=lock)

    def test_mimic_and_shared_copies_keep_addresses(self):
        a = self.model()
        tensor = a.config.kinematics_config.fixed_transforms
        b = self.model(tensor)
        c = self.model()
        ptr = tensor.data_ptr()
        q = dict(left_gripper=-.02, right_gripper=-.01)
        self.assertTrue(self.updater.update(q, [a, b, c]))
        self.assertEqual(ptr, tensor.data_ptr())
        for i, finger in enumerate(self.updater._fingers):
            expected = finger.origin[:, 3] + finger.direction * (q[finger.source] * finger.multiplier + finger.offset)
            torch.testing.assert_close(tensor[i, :, 3], expected)
        for model in (a, b, c):
            torch.testing.assert_close(model.lock_jointstate.position, torch.tensor([-.02, -.01]))
            self.assertEqual(model.lock_jointstate.velocity.count_nonzero(), 0)
            torch.testing.assert_close(model.config.kinematics_config.fixed_transforms, tensor)
        self.assertFalse(self.updater.update(q, [a, b, c]))
        fresh = self.model()
        self.assertFalse(self.updater.update(q, [fresh]))
        torch.testing.assert_close(fresh.config.kinematics_config.fixed_transforms, tensor)

    def test_invalid_feedback_does_not_modify_geometry(self):
        model = self.model()
        before = model.config.kinematics_config.fixed_transforms.clone()
        for q in ({}, dict(left_gripper=float('nan'), right_gripper=0),
                  dict(left_gripper=-.1, right_gripper=0)):
            with self.assertRaises(ValueError):
                self.updater.update(q, [model])
            torch.testing.assert_close(model.config.kinematics_config.fixed_transforms, before)

    def test_reject_optimized_jaws(self):
        model = self.model()
        model.joint_names.append('left_gripper')
        with self.assertRaises(RuntimeError):
            self.updater.update(dict(left_gripper=0, right_gripper=0), [model])

    def test_config_locks_grippers_for_all_groups(self):
        from curobo_ros_python.planning.groups import apply_robot, GROUP_JOINTS
        from curobo_ros_python.planning.algorithms.common import robot_cfg_with_locked_joints
        apply_robot('lycus.yml')
        for group, idle, size in [('both_arms', [], 14),
                                 ('left_arm', GROUP_JOINTS['right_arm'], 7),
                                 ('right_arm', GROUP_JOINTS['left_arm'], 7)]:
            data, locks = robot_cfg_with_locked_joints('lycus.yml', {}, idle)
            names = data['robot_cfg']['kinematics']['cspace']['joint_names']
            active = [n for n in names if n not in locks]
            self.assertEqual(len(active), size)
            self.assertEqual(set(active), set(GROUP_JOINTS[group]))
            self.assertTrue(set(self.updater.joint_names).issubset(locks))


if __name__ == '__main__':
    unittest.main()
