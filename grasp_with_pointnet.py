import pybullet as p
import time
import pybullet_data
import os
import math
import numpy as np
import copy
import pickle
import transforms3d as tf3d
import cv2

# TIME_STEP = 1./240.
TIME_STEP = 1. / 480.

base_dir = '/home/tpatten/Data/Hands/HO3D/'
data_split = 'train'
scene = 'ABF10'


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
        is_open:true for opening, false for closing
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


def grasp_example(obj_id, obj_rot, obj_trans, grasp):
    # p.connect(p.DIRECT)
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optional
    p.setGravity(0, 0, -10)

    # Gripper
    init_pos = np.copy(grasp[0:3, 3])
    q = tf3d.quaternions.mat2quat(grasp[:3, :3])
    init_ori = [q[1], q[2], q[3], q[0]]
    # tf3d quaternion: w, x, y, z
    # PyBullet quaterion: x, y, z, w
    hand = RobotGripper(init_pos, init_ori)

    # Load the object
    model_fn = os.path.join(base_dir, 'models', obj_id, 'vhacd.obj')
    obj_pos = np.copy(obj_trans)
    q = tf3d.quaternions.mat2quat(np.linalg.inv(cv2.Rodrigues(obj_rot)[0].T))
    obj_ori = [q[1], q[2], q[3], q[0]]
    mesh_scale = [0.001, 0.001, 0.001]

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
    # p.changeConstraint(obj_constraint, maxForce=1)
    p.changeConstraint(obj_constraint, maxForce=0)

    #time.sleep(2)
    time.sleep(1000)
    minimum_grasp_height = 0.225
    close_angle = -0.05
    speed = 0.5
    time_step = TIME_STEP
    move_step = 0.0005
    gripper_offset = 0.0125
    approach_timeout = 10
    grasp_timeout = 5
    lift_timeout = 15
    success_threshold = 0.05

    '''
    # Move down to grasp
    contacts = p.getContactPoints(target_obj, hand.hand_id)
    if len(contacts) > 0:
        print('ERROR: Gripper already in contact with object, exiting')
        p.disconnect()
        return False
    tool_pos = init_pos
    curr_hand_pos, _ = p.getBasePositionAndOrientation(hand.hand_id)
    start_obj_pos, _ = p.getBasePositionAndOrientation(target_obj)
    closest_points = p.getClosestPoints(hand.hand_id, target_obj, init_pos[2])
    max_distance = -1
    check_joint = True
    if hand.hand_palm_joint_id < 0:
        check_joint = False
    for c in closest_points:
        if c[8] > max_distance:
            if check_joint:
                if c[3] == hand.hand_palm_joint_id:
                    max_distance = c[8]
            else:
                max_distance = c[8]
    reach_limit = init_pos[2] - max_distance + gripper_offset
    if reach_limit < minimum_grasp_height:
        print('WARNING: reach_limit is too close to the floor')
        reach_limit = minimum_grasp_height

    t = 0
    print('STATE: Approaching')
    while tool_pos[2] > reach_limit:
        p.stepSimulation()
        tool_pos[2] = tool_pos[2] - move_step
        p.changeConstraint(hand.base_constraint, tool_pos, init_ori, maxForce=100000)
        time.sleep(time_step)
        t = t + time_step
        if t > approach_timeout:
            print('ERROR: Waited and did not reach the desired position')
            p.disconnect()
            return False

    # Close the gripper
    print('STATE: Closing gripper')
    grasp_success = False
    t = 0
    while not grasp_success:
        grasp_success = hand.update_grasp(close_angle, speed, time_step)
        p.stepSimulation()
        time.sleep(time_step)
        t = t + time_step
        if t > grasp_timeout:
            print('ERROR: Waited and did not grasp')
            p.disconnect()
            return False

    # Lift the object
    print('STATE: Lifting')
    lift_limit = init_pos[2] + 0.2
    lift_height = lift_limit - tool_pos[2]
    t = 0
    p.changeConstraint(obj_constraint, maxForce=0)
    while tool_pos[2] < lift_limit:
        p.stepSimulation()
        tool_pos[2] = tool_pos[2] + move_step
        p.changeConstraint(hand.base_constraint, tool_pos, init_ori, maxForce=100000)
        time.sleep(time_step)
        t = t + time_step
        if t > lift_timeout:
            print('ERROR: Waited and did not arrive at the lift height')
            p.disconnect()
            return False
    time.sleep(1)

    # Check if successful
    print('STATE: Checking success')
    # Success of object position is now higher (by the height the gripper is lifted)
    end_obj_pos, _ = p.getBasePositionAndOrientation(target_obj)
    contacts = p.getContactPoints(target_obj, hand.hand_id)
    success = False
    if len(contacts) > 0 and lift_height - (end_obj_pos[2] - start_obj_pos[2]) <= success_threshold:
        print('Successfully grasped object!')
        success = True
    else:
        print('Failed to grasp object!')
    '''

    p.disconnect()
    #return success
    return False


def load_pickle_data(f_name):
    with open(f_name, 'rb') as f:
        try:
            pickle_data = pickle.load(f, encoding='latin1')
        except:
            pickle_data = pickle.load(f)
    return pickle_data


if __name__ == '__main__':
    frame_ids = sorted(os.listdir(os.path.join(base_dir, data_split, scene, 'rgb')))

    for f in frame_ids:
        f_str = f.split('.')[0]
        f_name = os.path.join(base_dir, data_split, scene, 'meta2', f_str + '.pkl')
        anno = load_pickle_data(f_name)
        #f_name_2 = f_name.replace("meta", "meta2")
        #with open(f_name_2, 'wb') as f:
        #    pickle.dump(anno, f, protocol=2)

        f_name = f_name.replace("meta2/", "meta2/grasp_bl_")
        grasp = load_pickle_data(f_name)
        #f_name_2 = f_name.replace("meta", "meta2")
        #with open(f_name_2, 'wb') as f:
        #    pickle.dump(grasp, f, protocol=2)

        obj_rot = anno['objRot']
        obj_trans = anno['objTrans']
        obj_id = anno['objName']
        grasp = grasp.reshape(4, 4)

        grasp_example(obj_id, obj_rot, obj_trans, grasp)

        sys.exit(0)
