"""Measured fixed joints preserve CUDA buffers, geometry and lock metadata."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
import torch
from curobo_ros_python.planning.fixed_state import FixedJointStateUpdater


class FixedStateTest(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / 'robot.urdf'
        path.write_text('''<robot name="fixture"><link name="base"/><link name="chest"/><link name="head"/>
        <joint name="turn" type="revolute"><parent link="base"/><child link="chest"/>
        <origin xyz="1 0 0"/><axis xyz="0 0 1"/></joint>
        <joint name="slide" type="prismatic"><parent link="chest"/><child link="head"/>
        <origin xyz="0 0 1" rpy="0 0 1.57079632679"/><axis xyz="1 0 0"/></joint></robot>''')
        self.updater = FixedJointStateUpdater(str(path), {'turn': 0., 'slide': 0.})

    def model(self, transforms=None):
        if transforms is None:
            transforms = torch.zeros(2, 3, 4)
        lock = NS(joint_names=['turn', 'slide'], position=torch.zeros(1, 2),
                  velocity=torch.ones(1, 2), acceleration=torch.ones(1, 2), jerk=torch.ones(1, 2))
        kin = NS(fixed_transforms=transforms, link_name_to_idx_map={'chest': 0, 'head': 1},
                 joint_map_type=torch.tensor([-1, -1]))
        return NS(config=NS(kinematics_config=kin), joint_names=['arm'], lock_jointstate=lock)

    def test_revolute_prismatic_shared_copies_and_metadata(self):
        a = self.model(); tensor = a.config.kinematics_config.fixed_transforms
        b = self.model(tensor); c = self.model()
        pointer = tensor.data_ptr()
        values = {'turn': torch.pi / 2, 'slide': .2}
        self.assertTrue(self.updater.update(values, [a, b, c]))
        expected_rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        torch.testing.assert_close(tensor[0, :, :3], expected_rotation, atol=1e-6, rtol=0)
        torch.testing.assert_close(tensor[0, :, 3], torch.tensor([1., 0., 0.]))
        torch.testing.assert_close(tensor[1, :, 3], torch.tensor([0., .2, 1.]), atol=1e-6, rtol=0)
        self.assertEqual(pointer, tensor.data_ptr())
        for model in [a, b, c]:
            torch.testing.assert_close(model.config.kinematics_config.fixed_transforms, tensor)
            torch.testing.assert_close(model.lock_jointstate.position, torch.tensor([[torch.pi / 2, .2]]))
            self.assertEqual(model.lock_jointstate.velocity.count_nonzero(), 0)
            self.assertEqual(model.lock_jointstate.acceleration.count_nonzero(), 0)
        self.assertFalse(self.updater.update(values, [a, b, c]))
        fresh = self.model()
        self.assertFalse(self.updater.update(values, [fresh]))
        torch.testing.assert_close(fresh.config.kinematics_config.fixed_transforms, tensor)

    def test_missing_feedback_keeps_previous_fixed_state(self):
        model = self.model()
        self.updater.update({'turn': .3}, [model])
        self.updater.update({'slide': .1}, [model])
        self.assertEqual(self.updater.positions, {'turn': .3, 'slide': .1})

    def test_nonfinite_feedback_does_not_change_geometry(self):
        model = self.model()
        before = model.config.kinematics_config.fixed_transforms.clone()
        with self.assertRaises(ValueError):
            self.updater.update({'turn': float('nan')}, [model])
        torch.testing.assert_close(model.config.kinematics_config.fixed_transforms, before)

    def test_active_joint_and_unlocked_row_are_rejected(self):
        model = self.model()
        model.joint_names = ['turn']
        with self.assertRaises(RuntimeError):
            self.updater.update({'turn': .1}, [model])
        model.joint_names = ['arm']
        model.config.kinematics_config.joint_map_type[0] = 0
        with self.assertRaises(RuntimeError):
            self.updater.update({'turn': .1}, [model])


if __name__ == '__main__':
    unittest.main()
