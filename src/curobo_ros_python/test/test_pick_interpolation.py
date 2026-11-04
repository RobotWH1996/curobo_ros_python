"""Focused checks for interpolation policy, live feedback and protected objects."""
import time,unittest,xml.etree.ElementTree as ET
from types import SimpleNamespace as NS
from unittest.mock import Mock
import numpy as np,torch
from sensor_msgs.msg import PointCloud2,PointField
from curobo.types import JointState
from curobo_ros_python.planning.cartesian_interpolation import sample_poses,validate_path
from curobo_ros_python.planning.pick_place_server import TaskBackend
from curobo_ros_python.planning.pointcloud import cloud_to_base_points
from curobo_ros_python.planning.self_filter import SelfFilterMixin

class InterpolationTest(unittest.TestCase):
 def test_shortest_quaternion_path_and_line_rejection(self):
  start=np.array([0,0,0,1,0,0,0],np.float32);end=np.array([0,0,.1,-1,0,0,0],np.float32)
  path=sample_poses(start,end,np.linspace(0,1,5,dtype=np.float32));np.testing.assert_allclose(path[:,3],1.)
  validate_path(path,start,end,.005,.05)
  path[2,0]=.02
  with self.assertRaises(RuntimeError):validate_path(path,start,end,.005,.05)
 def test_labeled_objects_survive_body_mask_and_crop(self):
  dtype=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('label','<u4')]);data=np.zeros(4,dtype=dtype)
  data['x']=[.2,.3,.4,1.5];data['z']=.8;data['label']=[0,5,11,5]
  fields=[PointField(name=n,offset=i*4,datatype=PointField.FLOAT32 if i<3 else PointField.UINT32,count=1) for i,n in enumerate(['x','y','z','label'])]
  msg=PointCloud2(height=1,width=4,fields=fields,point_step=16,row_step=64,data=data.tobytes())
  points,count=cloud_to_base_points(msg,None,None,0.,3.,.02,100,bounds_min=[0,-1,0],bounds_max=[1,1,2],preserve_labeled_points=True,point_filter=lambda p,protected:p[protected])
  self.assertEqual(count,4);np.testing.assert_allclose(points[:,0],[.3,.4])
 def test_protection_does_not_mutate_geometry_cache(self):
  node=NS(self_filter_mode='apply',_self_filters={'left_arm':NS(required=['q'],mask=Mock(return_value=np.array([True,True,False])))},_joint_map={'q':0.},_self_filter_cache=None,self_filter_padding=.05,self_filter_max_processing_ms=300.,get_logger=lambda:Mock(),_publish_self_filter=Mock())
  pts=np.array([[.1,0,0],[.2,0,0],[.3,0,0]],np.float32)
  first=SelfFilterMixin._filter_robot_points(node,None,pts,'left_arm',protected=np.array([False,True,True]))
  np.testing.assert_array_equal(first,pts[1:]);self.assertEqual(int(node._self_filter_cache[2].sum()),2)
  second=SelfFilterMixin._filter_robot_points(node,None,pts,'left_arm',protected=np.zeros(3,dtype=bool))
  np.testing.assert_array_equal(second,pts[2:]);self.assertEqual(node._self_filter.mask.call_count,1)
 def test_live_feedback_still_rejects_stale_stream(self):
  b=object.__new__(TaskBackend);b.p=NS(_mesh_checker=NS(ordered=[ET.fromstring('<joint name="q" type="revolute"/>')]))
  b.virtual_joints={'q':0.};b.node=NS(_joint_received={'q':time.monotonic()},_p=lambda k:1.)
  b._validate_joint_feedback();b.node._joint_received['q']-=2.
  with self.assertRaisesRegex(RuntimeError,'反馈过期'):b._validate_joint_feedback()
 def test_single_actions_only_access_their_own_targets(self):
  pick=object.__new__(TaskBackend);pick.base_prepick=object();self.assertIs(pick._target('prepick'),pick.base_prepick)
  place=object.__new__(TaskBackend);place.base_preplace=object();self.assertIs(place._target('preplace'),place.base_preplace)
 def test_arm_segment_keeps_closed_gripper(self):
  b=object.__new__(TaskBackend);b.joint_names=['q','left_gripper'];b.virtual_joints={'q':0.,'left_gripper':-.025};b.virtual_state=object();b.p=NS(interpolation_dt=.02);b.node=NS(trajopt_velocity_scale=.5,base_frame='base_link');b.trajectories=[];b.trajectory_stages=[]
  js=JointState.from_position(torch.tensor([[0.],[.1]],dtype=torch.float32),joint_names=['q']);js.velocity=torch.zeros(2,1);js.acceleration=torch.zeros(2,1)
  traj=b._record_motion(js,'lift');np.testing.assert_allclose([p.positions[1] for p in traj.points],-.025)
  self.assertIsNone(b.virtual_state);self.assertEqual(b.trajectory_stages,['lift'])
