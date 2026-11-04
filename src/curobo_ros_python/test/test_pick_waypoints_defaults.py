"""Request/default geometry selection and route ordering before contact stages."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import numpy as np
from curobo_ros_python.planning.payload import resolve_object_geometry,pose_matrix


class DefaultGeometryTest(unittest.TestCase):
    def test_omitted_geometry_uses_config_xyz_xyzw(self):
        size,pose=resolve_object_geometry([0,0,0],np.eye(4),True,[.1,.2,.3],[0,.15,0,0,0,0,1])
        np.testing.assert_allclose(size,[.1,.2,.3])
        np.testing.assert_allclose(pose[:3,3],[0,.15,0])

    def test_explicit_geometry_and_explicit_identity_override_config(self):
        size,pose=resolve_object_geometry([.01,.02,.03],np.eye(4),False,[0,0,0],[0,.15,0,0,0,0,1])
        np.testing.assert_allclose(size,[.01,.02,.03]);np.testing.assert_array_equal(pose,np.eye(4))

    def test_unconfigured_partial_invalid_and_conflicting_pose_rejected(self):
        for requested,default in (([0,0,0],[0,0,0]),([.1,0,.1],[.1,.2,.3]),([-1,.1,.1],[.1,.2,.3])):
            with self.assertRaises(ValueError): resolve_object_geometry(requested,np.eye(4),True,default,[0,0,0,0,0,0,1])
        with self.assertRaises(ValueError):
            resolve_object_geometry([.1,.1,.1],pose_matrix([0,.1,0],[1,0,0,0]),True,[.1,.1,.1],[0,0,0,0,0,0,1])


class RouteTest(unittest.TestCase):
    def method(self):
        path=Path(__file__).parents[1]/'curobo_ros_python/planning/pick_place_server.py'
        node=next(n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.FunctionDef) and n.name=='_route_for_stage')
        namespace={'stamped_to_pose':lambda pose,device:pose}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
        return namespace['_route_for_stage']

    def test_routes_ordered_before_generated_prepose(self):
        backend=NS(pick_via=['pick1','pick2'],place_via=['place1'],p=NS(device_cfg=None))
        method=self.method()
        self.assertEqual(method(backend,'prepick'),['pick1','pick2'])
        self.assertEqual(method(backend,'preplace'),['place1'])

    def test_contact_and_retreat_stages_do_not_replay_routes(self):
        backend=NS(pick_via=['pick1'],place_via=['place1'],p=NS(device_cfg=None))
        for stage in ('pick','lift','place','retreat'):
            self.assertEqual(self.method()(backend,stage),[])


if __name__=='__main__': unittest.main()
