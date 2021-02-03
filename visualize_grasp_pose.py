import pybullet as p
import time
import pybullet_data
import os
import math
import numpy as np
import copy
import pickle
#import transforms3d as tf3d
#import cv2
import random

TIME_STEP = 1. / 240.
#TIME_STEP = 1. / 480.

base_dir = '/home/kw/0_code/hsr_hand_simulation/hsr_hand_simulation/objs/'
#data_split = 'train'
#scene = 'ABF10'

class RobotGripper:
    def __init__(self, translation, orientation, is_open=True):
        self.hand_id = p.loadURDF("./hsrb_description/robots/hand.urdf", translation,
                                  orientation)  # , flags=p.URDF_USE_SELF_COLLISION
        self.base_constraint = p.createConstraint(
            parentBodyUniqueId=self.hand_id,
            parentLinkIndex=-1,
            childBodyUniqueId=-1,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=translation,
            childFrameOrientation=orientation)
        # for accessing translation and orientation values
        self.translation = translation
        self.orientation = orientation
        self.n_joints = p.getNumJoints(self.hand_id)
        self.joint_id = []
        self.joint_names = []
        self.target_joint_names = [b'hand_l_proximal_joint', b'hand_r_proximal_joint', b'hand_l_distal_joint',
                                   b'hand_r_distal_joint', b'hand_palm_joint']
        self.hand_palm_joint_id = -1
        self.target_joint = []
        self.multi = []
        self.offset = []
        self.forces = []
        self.contact = []
        for i in range(self.n_joints):
            joints = p.getJointInfo(self.hand_id, i)
            self.joint_id.append(joints[0])
            self.joint_names.append(joints[1])
            if joints[1] in self.target_joint_names:
                self.target_joint.append(joints[0])
                if joints[1] in [b'hand_l_proximal_joint', b'hand_r_proximal_joint']:
                    self.multi.append(1)
                    self.offset.append(0)
                    self.forces.append(1)
                    self.contact.append(False)
                else:
                    self.multi.append(-1)
                    self.offset.append(-0.087)
                    self.forces.append(1000)
                    self.contact.append(True)

        for i in range(self.n_joints):
            if self.joint_names[i] == b'hand_palm_joint':
                self.hand_palm_joint_id = self.joint_id[i]
                break
        if self.hand_palm_joint_id >= 0:
            print('hand_palm_joint', self.joint_id[self.hand_palm_joint_id])
        else:
            print('ERROR: Not id for hand_palm_joint')

        for j_id, j in enumerate(self.target_joint):
            if self.contact[j_id]:
                # dynamics = p.getDynamicsInfo(self.hand_id,j)
                p.changeDynamics(self.hand_id, linkIndex=j,
                                 rollingFriction=0.7)  # , lateralFriction=0.1)#,contactStiffness=0.001,contactDamping=0.99)
            else:
                pass
                # p.changeDynamics(self.hand_id, linkIndex=j, contactStiffness=0.1, contactDamping=0.1)

        if is_open:
            t_pos = 1.2
        else:
            t_pos = 0
        self.prev_joints = []
        for j_id, j in enumerate(self.target_joint):
            joint_val = t_pos * self.multi[j_id] + self.offset[j_id]
            p.resetJointState(self.hand_id, jointIndex=j, targetValue=joint_val)
            self.prev_joints.append(joint_val)
        self.current_pos = t_pos
        self.count_stay = 0
        self.goal_stay = 0
        self.sign = 0

    def reset_internal_val(self):
        self.count_stay = 0
        self.goal_stay = 0
        self.sign = 0

    def update_grasp(self, target_pos, speed, time_step):
        """
        [input]
        is_open: true for opening, false for closing
        speed: motor speed (radian/s)
        time_step: time step for the simulation (s)

        [return]
        true: finished (reach the goal or not moving anymore)
        false: moving
        """
        if self.sign == 0:
            self.sign = np.sign(target_pos - self.current_pos)
        step = self.sign * speed * time_step

        # Check current position of the gripper
        current_joints = []
        diffs = []
        for j_id, j in enumerate(self.target_joint):
            joint_state = p.getJointState(self.hand_id, jointIndex=j)
            current_joints.append(joint_state[0])
            if not self.contact[j_id]:
                diffs.append(np.abs(joint_state[0] - self.prev_joints[j_id]))
        diff = np.sum(diffs)
        if diff < 0.0001:
            self.count_stay += 1
        elif self.count_stay > 0 and diff < 0.001:
            self.count_stay += 1
        else:
            self.count_stay = 0

        self.prev_joints = current_joints
        self.current_pos = self.current_pos + step

        if self.sign > 0:
            self.current_pos = min(self.current_pos, target_pos)
        else:
            self.current_pos = max(self.current_pos, target_pos)

        if np.abs(self.current_pos == target_pos) < 0.001:
            self.goal_stay += 1
        else:
            self.goal_stay = 0

        for j_id, j in enumerate(self.target_joint):
            joint_pose = self.multi[j_id] * self.current_pos + self.offset[j_id]
            p.setJointMotorControl2(bodyIndex=self.hand_id,
                                    jointIndex=j,
                                    controlMode=p.POSITION_CONTROL,  # mode = p.TORQUE_CONTROL
                                    targetPosition=joint_pose,
                                    force=self.forces[j_id],
                                    maxVelocity=5)

        if self.count_stay > 0.5 / time_step or self.goal_stay > 2 / time_step:
            return True
        else:
            return False

    def reset(self, translation, orientation, add_constraint):
        orientation_quaternion = p.getQuaternionFromEuler(orientation)
        p.resetBasePositionAndOrientation(self.hand_id, translation, orientation_quaternion)
        if add_constraint:
            self.base_constraint = p.createConstraint(
                parentBodyUniqueId=self.hand_id,
                parentLinkIndex=-1,
                childBodyUniqueId=-1,
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, 0],
                childFramePosition=translation,
                childFrameOrientation=orientation_quaternion)

def env():
    if not p.isConnected(p.GUI):
        p.connect(p.GUI)
        print ("\nConnect GUI...\n")

    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optional
    p.setGravity(0, 0, -10)

    # button(s)
    buttonID = p.addUserDebugParameter("next",1,0,1)

    # use a list for returning more parameters
    paramIDs = buttonID

    return paramIDs

def visualize(obj_name, grasp_poses, buttonID):
    # Plane
    p.loadURDF("plane.urdf")

    temp = p.readUserDebugParameter(buttonID)

    model_fn = os.path.join(base_dir, obj_name)
    
    obj_pos = np.copy(grasp_poses["obj_pos"])
    obj_ori = np.copy(grasp_poses["obj_ori"])

    mesh_scale = [0.001, 0.001, 0.001]
    # mesh_scale = [1.0, 1.0, 1.0]
    
    # Gripper
    init_pos = np.copy(grasp_poses["grasp_pos"])
    init_ori = p.getQuaternionFromEuler([0, math.pi, 0])  

    hand = RobotGripper(init_pos, init_ori)

    # Add object to the environment
    visual_shape_id = p.createVisualShape(shapeType=p.GEOM_MESH,
                                        fileName=model_fn,
                                        rgbaColor=[1, 1, 1, 1],
                                        specularColor=[0.4, .4, 0],
                                        meshScale=mesh_scale)
    collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                                fileName=model_fn,
                                                meshScale=mesh_scale)
    target_obj = p.createMultiBody(baseMass=0.01,
                                   baseInertialFramePosition=[0, 0, 0],
                                   baseCollisionShapeIndex=collision_shape_id,
                                   baseVisualShapeIndex=visual_shape_id,
                                   basePosition=obj_pos,
                                   baseOrientation=obj_ori)
    obj_constraint = p.createConstraint(parentBodyUniqueId=target_obj,
                                        parentLinkIndex=-1,
                                        childBodyUniqueId=-1,
                                        childLinkIndex=-1,
                                        jointType=p.JOINT_FIXED,
                                        jointAxis=[0, 0, 0],
                                        parentFramePosition=[0, 0, 0],
                                        childFramePosition=obj_pos,
                                        childFrameOrientation=obj_ori)
    p.changeConstraint(obj_constraint, maxForce=10)

    #p.addUserDebugText("Hello!", [0, 0, 0.6], textColorRGB=[1,0,0], textSize=1.5)
    p.addUserDebugText("Success rate: %.1f%%" % (grasp_poses["success_rate"]), [0, 0, 0.6], textColorRGB=[1,0,0], textSize=1.5)

    while (temp == p.readUserDebugParameter(buttonID)):
        # stay in the loop until the button has again been pressed
        time.sleep(1)

    #grasp_poses["grasp_pos"] = temp_pos
    p.resetSimulation()

    #p.disconnect()
    # return ...

def load_pickle_data(f_name):
    with open(f_name, 'rb') as f:
        try:
            pickle_data = pickle.load(f, encoding='latin1')
        except:
            pickle_data = pickle.load(f)
    return pickle_data

if __name__ == '__main__':

    # start GUI and get ids of buttons, etc.
    paramIDs = env()
    
    grasp_dir = base_dir.replace("objs", "grasp_poses")

    # frame_ids = sorted(os.listdir(os.path.join(base_dir, data_split, scene, 'rgb')))
    frame_ids = sorted(os.listdir(base_dir))
    frame_grasps = sorted(os.listdir(grasp_dir))

    grasp_poses = {}

    for f in frame_ids:
        print('File: %s\n' % f)

        f_str = f.split('.')[0]
        grasp_f = os.path.join(grasp_dir, f_str + '.pkl')
        exist_grasp_f = False

        for a in frame_grasps:
            if f_str == a.split('.')[0]:
                exist_grasp_f = True

        if exist_grasp_f == False:
            print('There is no previous created grasp pose file for the object %s' % f)
            continue
        else:
            grasp_poses = load_pickle_data(grasp_f)

        for pose_key, pose_values in grasp_poses.items():
            # call visualization function
            visualize(f, pose_values, paramIDs)

    p.disconnect()
