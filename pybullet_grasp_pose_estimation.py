import pybullet as p
import time
import pybullet_data
import os
import math
import numpy as np
import copy
import pickle
import transforms3d as tf3d
import random

TIME_STEP = 1. / 240.
#TIME_STEP = 1. / 480.

NUM_GRIPPERS = 3
NUM_POSES = 10
NUM_TOP = 6

base_dir = os.path.dirname(os.path.abspath(__file__))
obj_dir = os.path.join(base_dir, "canister")


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
                #p.changeDynamics(self.hand_id, linkIndex=j, mass = 1, 
                #                 rollingFriction=0.7, localInertiaDiagonal = [1,1,1])  # , lateralFriction=0.1)#,contactStiffness=0.001,contactDamping=0.99)
                p.changeDynamics(self.hand_id, linkIndex=j,
                                 rollingFriction=0.7)#, lateralFriction=0.1)#,contactStiffness=0.001,contactDamping=0.99)
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


def simStep(pause_buttonID):
    # ...press button -> value +1...

    while not (p.readUserDebugParameter(pause_buttonID) % 2):
        # stay in the loop until the button has again been pressed
        time.sleep(1)

    p.stepSimulation()


def env():
    if not p.isConnected(p.GUI):
        p.connect(p.GUI)
        print ("\nConnect GUI...\n")

    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optional
    p.setGravity(0, 0, -10)

    # button(s)
    pauseID = p.addUserDebugParameter("toggle pause",1,0,1)

    # optional: list for returning more parameters
    paramIDs = pauseID

    return paramIDs


def place_grippers(num_grippers, init_pos, init_ori):
    # Gripper
    #init_pos = [0, 0, 0.5]
    #init_ori = p.getQuaternionFromEuler([0, math.pi, 0])

    grippers = []

    for i in range(num_grippers):
        grippers.append(RobotGripper([init_pos[0]+i, init_pos[1], init_pos[2]], init_ori))

    return grippers


def matTransf(grasp_poses):
    # Object
    obj_pos = np.copy(grasp_poses["obj_pos"])
    obj_ori = np.copy(grasp_poses["obj_ori"])

    # Gripper
    init_pos = np.copy(grasp_poses["grasp_pos"])
    init_ori = np.copy(grasp_poses["grasp_ori"])

    obj_mat = np.matrix(obj_pos)
    gripp_mat = np.matrix(init_pos)

    # homogeneous coordinates
    hom_obj_pos = np.concatenate((obj_mat.T, np.matrix([1])))
    hom_gripp_pos = np.concatenate((gripp_mat.T, np.matrix([1])))

    # transformation matrices
    transl_mat_obj = np.matrix(np.identity(4))
    transl_mat_obj[:3,3] = obj_mat.T * -1
    transl_mat_gripp = np.matrix(np.identity(4))
    transl_mat_gripp[:3,3] = gripp_mat.T * -1

    # pure translations
    transl_obj_pos = np.matmul(transl_mat_obj,hom_obj_pos)
    transl_gripper_pos = np.matmul(transl_mat_gripp,hom_gripp_pos)

    # rotations matrices
    rot_mat_obj = np.matrix(p.getMatrixFromQuaternion(obj_ori)).reshape(3,3)
    hom_rot_mat_obj = np.concatenate((np.concatenate((rot_mat_obj, np.matrix([0, 0, 0]).T), 1), np.matrix([0, 0, 0, 1])))
    rot_mat_gripp = np.matrix(p.getMatrixFromQuaternion(init_ori)).reshape(3,3)
    hom_rot_mat_gripp = np.concatenate((np.concatenate((rot_mat_gripp, np.matrix([0, 0, 0]).T), 1), np.matrix([0, 0, 0, 1])))

    # pure rotations of the gripper
    rotated_gripper = np.matmul(hom_rot_mat_gripp,np.linalg.inv(hom_rot_mat_gripp))
    obj_rotated_gripper = np.matmul(rotated_gripper,np.linalg.inv(hom_rot_mat_obj))
    final_rotated_gripper = np.matmul(obj_rotated_gripper,hom_rot_mat_gripp)

    # difference of the two initial positions
    diff_vec = hom_gripp_pos - hom_obj_pos
    diff_vec *= -1
    diff_vec[3] = 1
    
    # transform the difference vector
    obj_gripp_transl_gripper_pos = np.matmul(final_rotated_gripper,diff_vec)

    if obj_gripp_transl_gripper_pos[2] < 0:
        obj_gripp_transl_gripper_pos[2] *= -1

    init_pos = obj_gripp_transl_gripper_pos[:3]
    q = tf3d.quaternions.mat2quat(final_rotated_gripper[:3,:3])
    init_ori = [q[1], q[2], q[3], q[0]]

    grasp_poses["grasp_rot"] = final_rotated_gripper
    grasp_poses["grasp_transl"] = obj_gripp_transl_gripper_pos[:3]


def grasp_example(obj_name, grasp_poses, paramIDs, num_grippers):
    
    # Plane
    p.loadURDF("plane.urdf")

    model_fn = os.path.join(obj_dir, obj_name)
    
    obj_pos = np.copy(grasp_poses["obj_pos"])
    obj_ori = np.copy(grasp_poses["obj_ori"])

    # mesh_scale = [0.001, 0.001, 0.001]
    mesh_scale = [1.0, 1.0, 1.0]
    
    hand = place_grippers(num_grippers, grasp_poses["grasp_pos"], grasp_poses["grasp_ori"])

    target_objs = []
    obj_constraints = []

    # Add object to the environment
    visual_shape_id = p.createVisualShape(shapeType=p.GEOM_MESH,
                                        fileName=model_fn,
                                        rgbaColor=[1, 1, 1, 1],
                                        specularColor=[0.4, .4, 0],
                                        meshScale=mesh_scale)
    collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                                fileName=model_fn,
                                                meshScale=mesh_scale)

    rand_x = np.random.random()
    rand_y = np.random.random()
    rand_z = np.random.random()
    for i in range(num_grippers):
        target_obj = p.createMultiBody(baseMass=0.01,
                                    baseInertialFramePosition=[0, 0, 0],
                                    baseCollisionShapeIndex=collision_shape_id,
                                    baseVisualShapeIndex=visual_shape_id,
                                    basePosition=(obj_pos[0]+i, obj_pos[1], obj_pos[2]),
                                    baseOrientation=obj_ori)

        # get Axis Aligned Bounding Box
        min_vals = list(p.getAABB(target_obj)[0])
        max_vals = list(p.getAABB(target_obj)[1])

        dx = max_vals[0] - min_vals[0]
        dy = max_vals[1] - min_vals[1]
        dz = max_vals[2] - min_vals[2]


        mean_x = (max_vals[0] + min_vals[0]) / 2
        mean_y = (max_vals[1] + min_vals[1]) / 2
        mean_z = (max_vals[2] + min_vals[2]) / 2

        # vanilla heuristics (Tim/Kevin)
        # Align each object centered below each gripper
        # x around i (0, 1, 2, ...)
        # y around 0
        # z starts at 0.2
        #new_x = ((obj_pos[0] + i) - mean_x) + i
        #new_y = obj_pos[1] - mean_y
        #new_z = obj_pos[2] - mean_z + ( dy / 2 ) + 0.2
        #p.resetBasePositionAndOrientation(target_obj, [new_x, new_y, new_z], obj_ori)

        # Stefan heuristics
        obj_dimension = np.array([dx, dy, dz])
        main_axis = np.argmax(obj_dimension)
        dim = obj_dimension - 0.018 # subtract gripper breadth
        #new_x = obj_pos[0] + i + (rand_x * dim[0]) - (dim[0] * 0.5) - mean_x + i
        #new_y = obj_pos[1] + (rand_y * dim[1]) - (dim[1] * 0.5) - mean_y
        if main_axis == 0:
            new_x = obj_pos[0] + i - mean_x + i + (rand_x * dim[0]) - (dim[0] * 0.5)
            new_y = obj_pos[1] - mean_y
        elif main_axis == 1:
            new_x = obj_pos[0] + i - mean_x + i
            new_y = obj_pos[1] - mean_y + (rand_y * dim[1]) - (dim[1] * 0.5)
        else:
            new_x = obj_pos[0] + i - mean_x + i
            new_y = obj_pos[1] - mean_y
        #new_z = obj_pos[2] + (rand_z * dim[2]) - (dim[2] * 0.5) - mean_z + (dy / 2) + 0.2
        new_z = obj_pos[2] + mean_z

        p.resetBasePositionAndOrientation(target_obj, [new_x, new_y, new_z], obj_ori)

        obj_constraint = p.createConstraint(parentBodyUniqueId=target_obj,
                                            parentLinkIndex=-1,
                                            childBodyUniqueId=-1,
                                            childLinkIndex=-1,
                                            jointType=p.JOINT_FIXED,
                                            jointAxis=[0, 0, 0],
                                            parentFramePosition=[0, 0, 0],
                                            #childFramePosition=(obj_pos[0]+i, obj_pos[1], obj_pos[2]),
                                            childFramePosition=(new_x, new_y, new_z),
                                            #childFramePosition=obj_pos,
                                            childFrameOrientation=obj_ori)
        p.changeConstraint(obj_constraint, maxForce=20)

        target_objs.append(target_obj)
        obj_constraints.append(obj_constraint)

    # Keep used object position
    grasp_poses["obj_pos"], _ = p.getBasePositionAndOrientation(target_objs[0])

    time.sleep(1)

    # parameters for lifting (or moving down the gripper)
    #minimum_grasp_height = 0.225
    minimum_grasp_height = 0.07 + obj_dimension[2] * 0.5

    print('min grasp height: ', minimum_grasp_height)
    close_angle = -0.05
    speed = 0.5
    time_step = TIME_STEP
    move_step = 0.0005
    gripper_offset = 0.0125
    
    approach_timeout = 10
    grasp_timeout = 5
    lift_timeout = 15
    shake_timeout = 2

    success_threshold = 0.05

    hand_pos_delta = 0.005
    hand_ori_delta = 0.005

    # Keep inital positions
    init_pos = list(hand[0].translation)
    tool_pos = init_pos

    for i in range(num_grippers):
        # Check if the gripper and object are in collision
        contacts = p.getContactPoints(target_objs[i], hand[i].hand_id)
        if len(contacts) > 0:
            print('ERROR: Gripper already in contact with object, exiting')
            # p.disconnect()
            return 0

    if not grasp_poses["is_initial"]:
        for i in range(num_grippers):
            hand[i].translation = np.copy(grasp_poses["grasp_pos"])
            hand[i].translation[0] += i
            p.changeConstraint(hand[i].base_constraint, hand[i].translation, hand[i].orientation, maxForce=100000)
    else:
        # Move down to grasp
        for i in range(num_grippers):
            closest_points = p.getClosestPoints(hand[i].hand_id, target_objs[i], init_pos[2])
            max_distance = -1
            check_joint = True
            if hand[i].hand_palm_joint_id < 0:
                check_joint = False
            for c in closest_points:
                if c[8] > max_distance:
                    if check_joint:
                        if c[3] == hand[i].hand_palm_joint_id:
                            max_distance = c[8]
                    else:
                        max_distance = c[8]
            reach_limit = init_pos[2] - max_distance + gripper_offset
            if reach_limit < minimum_grasp_height:
                print('WARNING: reach_limit is too close to the floor')
                reach_limit = minimum_grasp_height

        t = 0

        print('STATE: Approaching')
        while hand[0].translation[2] > reach_limit:
            # p.stepSimulation()
            simStep(paramIDs)
            
            for i in range(num_grippers):
                hand[i].translation[2] = hand[i].translation[2] - move_step
                p.changeConstraint(hand[i].base_constraint, hand[i].translation, hand[i].orientation, maxForce=100000)
            time.sleep(time_step)
            t = t + time_step
            if t > approach_timeout:
                print('ERROR: Waited and did not reach the desired position')
                # p.disconnect()
                return 0
        
        # Keep grasp position
        temp_pos = list(hand[0].translation)
        # Change saved gripper height in z-direction in relation to the height difference of the object - possibly pushed down while approaching
        curr_pos, _ = p.getBasePositionAndOrientation(target_objs[0])
        diff_pos = list(grasp_poses["obj_pos"])[2] - list(curr_pos)[2]
        temp_pos[2] += diff_pos + gripper_offset

    start_obj_pos, _ = p.getBasePositionAndOrientation(target_objs[0])

    # Close the gripper
    print('STATE: Closing gripper')
    grasp_success = False
    t = 0
    while not grasp_success:
        for i in range(num_grippers):
            grasp_success = hand[i].update_grasp(close_angle, speed, time_step)
        #p.stepSimulation()
        simStep(paramIDs)
        time.sleep(time_step)
        t = t + time_step
        if t > grasp_timeout:
            print('ERROR: Waited and did not grasp')
            # p.disconnect()
            return 0

    # Lift the object
    print('STATE: Lifting')

    lift_limit = init_pos[2] + 0.2
    lift_height = lift_limit - tool_pos[2]

    t = 0

    for i in range(num_grippers):
        p.changeConstraint(obj_constraints[i], maxForce=0)
        # Applying different forces to the object under each gripper
        #if num_grippers == 1:
        #    newForce = 0
        #else:
        #    newForce = i * 0.5/(num_grippers-1)
        #    print(" Index = %i, Force = %f" % (i, newForce))
        #p.changeConstraint(obj_constraints[i], newForce)

    while hand[0].translation[2] < lift_limit:
        #p.stepSimulation()
        simStep(paramIDs)

        for i in range(num_grippers):
            hand[i].translation[2] = hand[i].translation[2] + move_step
            p.changeConstraint(hand[i].base_constraint, hand[i].translation, hand[i].orientation, maxForce=100000)
        
        time.sleep(time_step)
        t = t + time_step
        if t > lift_timeout:
            print('ERROR: Waited and did not arrive at the lift height')
            # p.disconnect()
            return 0
    time.sleep(1)

    # Check if successful
    print('STATE: Checking success')
    # Success of object position is now higher (by the height the gripper is lifted)
    success = 0

    for i in range(num_grippers):
        end_obj_pos, _ = p.getBasePositionAndOrientation(target_objs[i])
        contacts = p.getContactPoints(target_objs[i], hand[i].hand_id)

        if len(contacts) > 0 and lift_height - (end_obj_pos[2] - start_obj_pos[2]) <= success_threshold:
            print('Successfully grasped and lifted object with the %s. gripper!' % str(i+1))
            if(grasp_poses["is_initial"]):
                grasp_poses["grasp_pos"] = temp_pos
            success += 1
        else:
            print('Failed to grasp and lift object! with the %s. gripper!' % str(i+1))
            # p.disconnect()
            # return success

        p.changeConstraint(obj_constraint, maxForce=0)

    if success:
        # Shaking the object
        time.sleep(1)
        print('STATE: Shaking the object')
        t = 0
        while t < shake_timeout:
            #p.stepSimulation()
            simStep(paramIDs)
            for i in range(num_grippers):
                curr_hand_pos, curr_hand_ori = p.getBasePositionAndOrientation(hand[i].hand_id)
                dx = random.uniform(-hand_pos_delta, hand_pos_delta)
                dy = random.uniform(-hand_pos_delta, hand_pos_delta)
                dz = random.uniform(-hand_pos_delta, hand_pos_delta)
                # print(dx, dy, dz)
                curr_hand_pos = [curr_hand_pos[0] + random.uniform(-hand_pos_delta, hand_pos_delta),
                                curr_hand_pos[1] + random.uniform(-hand_pos_delta, hand_pos_delta),
                                curr_hand_pos[2] + random.uniform(-hand_pos_delta, hand_pos_delta)]
                curr_hand_ori = [curr_hand_ori[0] + random.uniform(-hand_ori_delta, hand_ori_delta),
                                curr_hand_ori[1] + random.uniform(-hand_ori_delta, hand_ori_delta),
                                curr_hand_ori[2] + random.uniform(-hand_ori_delta, hand_ori_delta)]
                p.changeConstraint(hand[i].base_constraint, curr_hand_pos, curr_hand_ori, maxForce=100000)
            time.sleep(time_step)
            t = t + time_step

    # Check if successful
    time.sleep(2)
    
    print('STATE: Checking success')

    success = 0

    for i in range(num_grippers):
        # Success of object position is now higher (by the height the gripper is lifted)
        end_obj_pos, _ = p.getBasePositionAndOrientation(target_objs[i])
        contacts = p.getContactPoints(target_objs[i], hand[i].hand_id)

        if len(contacts) > 0:
            print('Successfully grasped and shook object with the %s. gripper!' % str(i+1))
            success += 1
        else:
            print('Failed to grasp and shake object with the %s. gripper!' % str(i+1))

    # transform grasp pose to a rotation matrix
    matTransf(grasp_poses)

    #grasp_poses["grasp_pos"] = temp_pos
    p.resetSimulation()

    #p.disconnect()
    return success


def load_pickle_data(f_name):
    with open(f_name, 'rb') as f:
        try:
            pickle_data = pickle.load(f, encoding='latin1')
        except:
            pickle_data = pickle.load(f)
    return pickle_data


if __name__ == '__main__':
    random.seed(None)

    # start GUI and get ids of buttons, etc.
    paramIDs = env()
    
    grasp_dir = obj_dir.replace(obj_dir, "grasp_poses")

    frame_ids = sorted(os.listdir(obj_dir))
    frame_grasps = sorted(os.listdir(grasp_dir))

    num_grippers = NUM_GRIPPERS
    num_samples = 1

    grasp_poses = {}

    for f in frame_ids: # loop over objects
        print('File: %s\n' % f)

        f_str = f.split('.')[0]
        print(f_str)
        grasp_f = os.path.join(grasp_dir, f_str + '.pkl')
        exist_grasp_f = False

        for a in frame_grasps:
            if f_str == a.split('.')[0]:
                exist_grasp_f = True
        print(exist_grasp_f)

        if exist_grasp_f == False:
            print('There is no previous created grasp pose file for the object %s' % f)
            print('Create one with default values ...\n')

            '''
            grasp_poses = {
                1: {'obj_pos': [0, 0, 0], 'obj_ori': p.getQuaternionFromEuler([0, 0, 0]), 'grasp_pos': [0, 0, 0.5],
                    'grasp_ori': p.getQuaternionFromEuler([0, math.pi, 0]), 'grasp_rot': np.matrix(np.identity(4)),
                    'grasp_transl': np.matrix([0, 0, 0, 1]), 'is_initial': True, 'success_rate': 0},
                2: {'obj_pos': [0, 0, 0], 'obj_ori': p.getQuaternionFromEuler([math.pi / 2, 0, 0]),
                    'grasp_pos': [0, 0, 0.5], 'grasp_ori': p.getQuaternionFromEuler([0, math.pi, 0]),
                    'grasp_rot': np.matrix(np.identity(4)), 'grasp_transl': np.matrix([0, 0, 0, 1]), 'is_initial': True,
                    'success_rate': 0},
                3: {'obj_pos': [0, 0, 0], 'obj_ori': p.getQuaternionFromEuler([math.pi / 2, math.pi / 2, 0]),
                    'grasp_pos': [0, 0, 0.5], 'grasp_ori': p.getQuaternionFromEuler([0, math.pi, 0]),
                    'grasp_rot': np.matrix(np.identity(4)), 'grasp_transl': np.matrix([0, 0, 0, 1]), 'is_initial': True,
                    'success_rate': 0},
                4: {'obj_pos': [0, 0, 0], 'obj_ori': p.getQuaternionFromEuler([0, math.pi / 2, 0]),
                    'grasp_pos': [0, 0, 0.5], 'grasp_ori': p.getQuaternionFromEuler([0, math.pi, 0]),
                    'grasp_rot': np.matrix(np.identity(4)), 'grasp_transl': np.matrix([0, 0, 0, 1]), 'is_initial': True,
                    'success_rate': 0},
                5: {'obj_pos': [0, 0, 0], 'obj_ori': p.getQuaternionFromEuler([math.pi / 2, 0, math.pi / 2]),
                    'grasp_pos': [0, 0, 0.5], 'grasp_ori': p.getQuaternionFromEuler([0, math.pi, 0]),
                    'grasp_rot': np.matrix(np.identity(4)), 'grasp_transl': np.matrix([0, 0, 0, 1]), 'is_initial': True,
                    'success_rate': 0},
                6: {'obj_pos': [0, 0, 0], 'obj_ori': p.getQuaternionFromEuler([0, 0, math.pi / 2]),
                    'grasp_pos': [0, 0, 0.5], 'grasp_ori': p.getQuaternionFromEuler([0, math.pi, 0]),
                    'grasp_rot': np.matrix(np.identity(4)), 'grasp_transl': np.matrix([0, 0, 0, 1]), 'is_initial': True,
                    'success_rate': 0}}
            '''
            grasp_poses = {}

            gdx = 1
            # rotate top
            #for og in range(NUM_TOP):
            #    obj_pos = [0.0, 0.0, 0.0]  # , np.random.random() * 0.11 - 0.055]
            #    grasp_pos = [0.009, 0, 0.5]
            #    obj_ori = p.getQuaternionFromEuler([math.pi * 0.5, 0.0, np.random.random() * math.pi * 2])
            #    grasp_ori = p.getQuaternionFromEuler([0.0, math.pi, 0.0])
            #    grasp_poses[str(gdx)] = {'obj_pos': obj_pos, 'obj_ori': obj_ori, 'grasp_pos': grasp_pos,
            #                             'grasp_ori': grasp_ori, 'is_initial': True, 'success_rate': 0}
            #    gdx += 1
            # rotate bottom
            #for og in range(NUM_TOP):
            #    obj_pos = [0.0, 0.0, 0.0]  # , np.random.random() * 0.11 - 0.055]
            #    grasp_pos = [0.009, 0, 0.5]
            #    obj_ori = p.getQuaternionFromEuler([math.pi * 1.5, 0.0, np.random.random() * math.pi * 2])
            #    grasp_ori = p.getQuaternionFromEuler([0.0, math.pi, 0.0])
            #    grasp_poses[str(gdx)] = {'obj_pos': obj_pos, 'obj_ori': obj_ori, 'grasp_pos': grasp_pos,
            #                             'grasp_ori': grasp_ori, 'is_initial': True, 'success_rate': 0}
            #    gdx += 1

            # random along z
            for og in range(NUM_POSES):
                obj_pos = [(np.random.random() * 0.1) - 0.05, 0.0, 0.0] #, np.random.random() * 0.11 - 0.055]
                grasp_pos = [0.009, 0, 0.5]
                if og % 2 == 0:
                    obj_ori = p.getQuaternionFromEuler([0.0, np.random.random() * math.pi * 2, math.pi * 0.5])
                else:
                    obj_ori = p.getQuaternionFromEuler([0.0, np.random.random() * math.pi * 2, math.pi * 1.5])
                grasp_ori = p.getQuaternionFromEuler([0.0, math.pi, 0.0])
                grasp_poses[str(gdx)] = {'obj_pos': obj_pos, 'obj_ori': obj_ori, 'grasp_pos': grasp_pos, 'grasp_ori': grasp_ori, 'is_initial': True, 'success_rate': 0}
                gdx += 1

        else:
            grasp_poses = load_pickle_data(grasp_f)

            for pose_key, pose_values in grasp_poses.items():
                pose_values['is_initial'] = False

        numpy_poses = np.empty((0, 16))
        for pose_key, pose_values in grasp_poses.items():

            success_count = 0

            for a in range(num_samples):
                success_count += grasp_example(f, pose_values, paramIDs, num_grippers)

            success_rate = float(success_count) / float(num_grippers * num_samples)

            print('%.1f%% success for object %s\n' % (success_rate * 100, f))
            pose_values['success_rate'] = success_rate
            
        # Save all values  
        with open(grasp_f, 'wb') as f:
            pickle.dump(grasp_poses, f, protocol=2)

    p.disconnect()
