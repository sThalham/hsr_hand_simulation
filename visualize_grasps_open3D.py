import numpy as np
import open3d as o3d
import _pickle as cPickle
import transforms3d as tf3d
import copy

gripper = o3d.io.read_triangle_mesh("/home/stefan/hsr_hand_simulation/hsrb_description/meshes/hand_open.ply")
canister = o3d.io.read_triangle_mesh("/home/stefan/data/datasets/canister/models/obj_000001.ply")
with open(r"/home/stefan/hsr_hand_simulation/grasp_poses/canister.pkl", "rb") as input_file:
    anno = cPickle.load(input_file)

grippers = [canister]
for idx in range(len(anno)):
    obj_pose = np.eye(4)
    gri_pose = np.eye(4)

    obj_pose[:3, 3] = anno[str(idx+1)]['obj_pos']
    obj_pose[:3, :3] = tf3d.quaternions.quat2mat(anno[str(idx+1)]['obj_ori'])

    gri_pose[:3, 3] = anno[str(idx+1)]['grasp_pos']
    gri_pose[:3, :3] = tf3d.quaternions.quat2mat(anno[str(idx+1)]['grasp_ori'])

    pose = np.linalg.inv(obj_pose) @ gri_pose
    #print('pose 1: ', pose)
    #pose = anno[str(idx+1)]['grasp_rot']
    #pose[:3, 3] = anno[str(idx+1)]['grasp_transl']
    #pose = np.linalg.inv(obj_pose) @ pose
    #print('success: ', anno[str(idx+1)]['success_rate'])
    
    if anno[str(idx+1)]['success_rate'] < 1.0:
        continue

    grip_now = copy.deepcopy(gripper)
    grip_now.paint_uniform_color([np.random.random(), np.random.random(), np.random.random()])
    grip_now.transform(pose)
    grippers.append(grip_now)
    
    obj_now = copy.deepcopy(canister)
    #obj_now.transform(obj_pose)
    o3d.visualization.draw_geometries([obj_now, grip_now])

o3d.visualization.draw_geometries(grippers)
