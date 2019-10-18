import pybullet as p
import time
import pybullet_data
import math
import numpy as np
import copy


TIME_STEP = 1./240.


class RobotGripper:
    def __init__(self, translation, orientation, is_open=True):
        self.hand_id = p.loadURDF("./hsrb_description/robots/hand.urdf", translation, orientation)  # , flags=p.URDF_USE_SELF_COLLISION
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
                p.changeDynamics(self.hand_id, linkIndex=j, rollingFriction=0.7)  # , lateralFriction=0.1)#,contactStiffness=0.001,contactDamping=0.99)
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
            joint_pose = self.multi[j_id]*self.current_pos+self.offset[j_id]
            p.setJointMotorControl2(bodyIndex=self.hand_id,
                                    jointIndex=j,
                                    controlMode=p.POSITION_CONTROL,  # mode = p.TORQUE_CONTROL
                                    targetPosition=joint_pose,
                                    force=self.forces[j_id],
                                    maxVelocity=5)
        
        if self.count_stay > 0.5/time_step or self.goal_stay > 2/time_step:
            return True
        else:
            return False


def grasp_example():
    #p.connect(p.DIRECT)
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optional
    p.setGravity(0, 0, -10)

    # Gripper
    init_pos = [0, 0, 0.5]
    init_ori = p.getQuaternionFromEuler([0, math.pi, 0])
    hand = RobotGripper(init_pos, init_ori)

    # Plane
    p.loadURDF("plane.urdf")

    # Select the object
    grasp_watering_can = False
    if grasp_watering_can:
        model_fn = "./objs/can_linemod.obj"
        obj_pos = [-0.1, 0, 0.1]
        obj_ori = p.getQuaternionFromEuler([0, math.pi/2, math.pi/2])
    else:
        model_fn = "./objs/obj_000003.obj"
        obj_pos = [0, 0, 0.1]
        obj_ori = p.getQuaternionFromEuler([0, 0, 0])
        #obj_pos = [0, 0, 0.015]
        #obj_ori = p.getQuaternionFromEuler([0, math.pi/2, 0])
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
    #p.changeConstraint(obj_constraint, maxForce=1)
    p.changeConstraint(obj_constraint, maxForce=0)

    time.sleep(2)
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

    p.disconnect()
    return success


def render_example():
    # p.connect(p.GUI)
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)
    p.loadURDF("plane.urdf")
    mesh_scale = [0.001, 0.001, 0.001]
    model_fn = "./objs/obj_000003.obj"
    # model_fn = "./objs/can_linemod.obj"
    obj_pos = [0, 0, 0.1]
    obj_ori = p.getQuaternionFromEuler([0, 0, 0])
    # Create axis
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
    p.changeConstraint(obj_constraint, maxForce=1)

    position, orientation = p.getBasePositionAndOrientation(target_obj)
    print('Object pose: ', position, orientation)

    cam_target_pos = [0.2, 0, 0.25]
    roll = 0
    pitch = -90.0
    yaw = 0
    up_axis_id = 2
    cam_distance = 1
    pixel_width = 640
    pixel_height = 480
    near_plane = 0.01
    far_plane = 100
    fov = 60

    view_matrix = p.computeViewMatrixFromYawPitchRoll(cam_target_pos, cam_distance, yaw, pitch, roll, up_axis_id)
    aspect = pixel_width / pixel_height
    projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near_plane, far_plane)

    img_arr = p.getCameraImage(pixel_width, pixel_height, view_matrix, projection_matrix, shadow=1,
                               lightDirection=[1, 1, 1], renderer=p.ER_BULLET_HARDWARE_OPENGL)
    w = img_arr[0]  # width of the image, in pixels
    h = img_arr[1]  # height of the image, in pixels
    rgb = img_arr[2]  # color data RGB
    bullet_depth = img_arr[3]  # depth data
    depth = far_plane * near_plane / (far_plane - (far_plane - near_plane) * bullet_depth)
    # print(np.min(depth), np.max(depth))
    rgb = np.reshape(rgb, (h, w, 4))
    depth = np.reshape(depth, (h, w))

    import scipy.misc
    scipy.misc.toimage(rgb, cmin=0.0, cmax=255).save('/home/tpatten/rgb.png')
    scipy.misc.toimage(depth, cmin=np.min(depth), cmax=np.max(depth)).save('/home/tpatten/depth.png')

    p.disconnect()


def camera_example():
    import matplotlib.pyplot as plt
    plt.ion()

    img = [[1, 2, 3] * 50] * 100  # np.random.rand(200, 320)
    image = plt.imshow(img, interpolation='none', animated=True, label="Rendering", cmap='gray')
    ax = plt.gca()

    p.connect(p.DIRECT)
    #p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optional

    # pybullet.loadPlugin("eglRendererPlugin")
    p.loadURDF("plane.urdf")
    init_pos = [0, 0, 0.5]
    init_ori = p.getQuaternionFromEuler([0, 0, 0])
    p.loadURDF("r2d2.urdf", init_pos, init_ori)

    p.setGravity(0, 0, -10)
    camTargetPos = [0, 0, 1]
    pitch = -90.0
    roll = 0
    upAxisIndex = 2
    camDistance = 1
    pixelWidth = 640
    pixelHeight = 480
    nearPlane = 0.01
    farPlane = 100
    fov = 60

    main_start = time.time()
    while True:
        for yaw in range(0, 360, 120):
            p.stepSimulation()
            start = time.time()
            viewMatrix = p.computeViewMatrixFromYawPitchRoll(camTargetPos, camDistance, yaw, pitch,
                                                             roll, upAxisIndex)
            aspect = pixelWidth / pixelHeight
            projectionMatrix = p.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane)
            img_arr = p.getCameraImage(pixelWidth,
                                       pixelHeight,
                                       viewMatrix,
                                       projectionMatrix,
                                       shadow=1,
                                       lightDirection=[1, 1, 1],
                                       renderer=p.ER_BULLET_HARDWARE_OPENGL)
            stop = time.time()
            print("renderImage %f" % (stop - start))

            w = img_arr[0]  # width of the image, in pixels
            h = img_arr[1]  # height of the image, in pixels
            rgb = img_arr[2]  # color data RGB
            dep = img_arr[3]  # depth data
            # print(rgb)
            print('width = %d height = %d' % (w, h))

            # note that sending the data using imshow to matplotlib is really slow, so we use set_data

            # plt.imshow(rgb,interpolation='none')

            # reshape is needed
            #np_img_arr = np.reshape(rgb, (h, w, 4))
            #np_img_arr = np_img_arr * (1. / 255.)
            dep -= np.min(dep)
            dep /= np.max(dep)
            np_img_arr = np.reshape(dep, (h, w))
            np_img_arr = np_img_arr * 255.
            print(np.min(np_img_arr), np.max(np_img_arr))

            image.set_data(np_img_arr)
            ax.plot([0])
            # plt.draw()
            # plt.show()
            plt.pause(0.01)

    main_stop = time.time()

    print("Total time %f" % (main_stop - main_start))

    p.resetSimulation()


if __name__ == '__main__':
    grasp_example()
    #camera_example()
    #render_example()
