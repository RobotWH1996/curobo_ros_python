"""Contact scopes restore geometry; payload contacts remain object-specific."""
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
import fcl
import numpy as np
import torch
from curobo_ros_python.planning.pick_collision import contact_scope
from curobo_ros_python.planning.mesh_collision import MeshCollisionChecker
from curobo_ros_python.planning.payload import Payload,pose_matrix


class ContactTest(unittest.TestCase):
    def test_scope_restores_all_sphere_values_and_fcl_policy_after_exception(self):
        links={'left_gripper_left_jaw':0,'left_gripper_right_jaw':1,'arm':2}
        kin=NS(link_spheres=torch.tensor([[[0.,0.,0.,.1],[0.,0.,0.,.2],[0.,0.,0.,.3]]]),
               link_name_to_idx_map=links,
               get_sphere_index_from_link_name=lambda name:torch.tensor([links[name]]))
        original=kin.link_spheres.clone()
        previous={'other'};checker=NS(environment_ignore_links=previous)
        p=NS(_mesh_checker=checker)
        with patch('curobo_ros_python.planning.pick_collision.models',return_value=[kin]):
            with self.assertRaises(TimeoutError):
                with contact_scope(p,'left_arm',True):
                    self.assertEqual(float(kin.link_spheres[0,0,3]),-100.)
                    self.assertAlmostEqual(float(kin.link_spheres[0,2,3]),.3,places=6)
                    self.assertIn('left_gripper_left_jaw',checker.environment_ignore_links)
                    raise TimeoutError('cancel')
        torch.testing.assert_close(kin.link_spheres,original)
        self.assertIs(checker.environment_ignore_links,previous)

    def fixture(self):
        c=MeshCollisionChecker.__new__(MeshCollisionChecker)
        c.fcl=fcl;c.names=['left_gripper_left_jaw','arm']
        c.objects=[fcl.CollisionObject(fcl.Box(.05,.05,.05),fcl.Transform(np.eye(3),np.array([0.,0.,.5]))),
                   fcl.CollisionObject(fcl.Box(.05,.05,.05),fcl.Transform(np.eye(3),np.array([2.,0.,0.])))]
        c.payload=Payload('left_arm',np.array([.1,.1,.1],dtype=np.float32),pose_matrix([0,0,.5],[1,0,0,0]),np.eye(4))
        c._transforms=lambda positions,links:np.eye(4)[None]
        c.environment={}
        return c

    def test_payload_touches_own_fingers_but_not_arm(self):
        c=self.fixture();self.assertIsNone(c._check_payload({},[]))
        c.objects[1].setTranslation(np.array([0.,0.,.5]))
        self.assertEqual(c._check_payload({},[]).kind,'payload_self')

    def test_payload_environment_contact_is_rejected(self):
        c=self.fixture()
        obj=fcl.CollisionObject(fcl.Box(.05,.05,.05),fcl.Transform(np.eye(3),np.array([0.,0.,.5])))
        c.environment={'table':(obj,np.array([[-1,-1,0],[1,1,1]],dtype=np.float32))}
        self.assertEqual(c._check_payload({},[]).kind,'payload_environment')

    def test_pointcloud_flat_faces_reach_native_fcl(self):
        from curobo._src.geom.types import Mesh,SceneCfg
        checker=MeshCollisionChecker.__new__(MeshCollisionChecker);checker.fcl=fcl
        mesh=Mesh.from_pointcloud(np.array([[0.,0.,0.],[.03,0.,0.]],dtype=np.float32),pitch=.02,name='cam')
        scene=checker.prepare_scene(SceneCfg(mesh=[mesh]))
        self.assertIn('cam',scene)
        result=fcl.CollisionResult()
        hit=fcl.collide(scene['cam'][0],fcl.CollisionObject(fcl.Box(.05,.05,.05)),fcl.CollisionRequest(),result)
        self.assertGreater(hit,0)


if __name__=='__main__': unittest.main()
