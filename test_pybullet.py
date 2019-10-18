import pybullet as p
import time
import pybullet_data
import math
import numpy as np


TIME_STEP = 1./240.


class RobotGripper:
    def __init__(self, translation, orientation, is_open=True):
        self.hand_id = p.loadURDF("./hsrb_description/robots/hand.urdf",translation, orientation)  #,flags=p.URDF_USE_SELF_COLLISION
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
        self.target_joint_names = [b'hand_l_proximal_joint',b'hand_r_proximal_joint',b'hand_l_distal_joint',b'hand_r_distal_joint']
        self.target_joint = []
        self.multi = []
        self.offset = []
        self.forces = []
        self.contact = []
        for i in range(self.n_joints):
            joints = p.getJointInfo(self.hand_id, i)
            self.joint_id.append(joints[0])
            self.joint_names.append(joints[1])
            print(joints[0], joints[1], joints[12])
            if joints[1] in self.target_joint_names:
                self.target_joint.append(joints[0])
                if joints[1] in [b'hand_l_proximal_joint',b'hand_r_proximal_joint']:
                    print(joints[0], joints[1], joints[2])
                    self.multi.append(1)
                    self.offset.append(0)
                    self.forces.append(1)
                    self.contact.append(False)  
                else:
                    self.multi.append(-1)
                    self.offset.append(-0.087)
                    self.forces.append(1000)
                    self.contact.append(True)

        print(p.getBodyInfo(self.hand_id))
        for j_id, j in enumerate(self.target_joint):
            if self.contact[j_id]:
                # dynamics = p.getDynamicsInfo(self.hand_id,j)
                p.changeDynamics(self.hand_id, linkIndex=j, rollingFriction=0.7)  #,lateralFriction=0.1)#,contactStiffness=0.001,contactDamping=0.99)
            else:
                pass
                # p.changeDynamics(self.hand_id, linkIndex=j, contactStiffness=0.1, contactDamping=0.1)
        
        if is_open:
            t_pos = 1.2
        else:
            t_pos = 0
        self.prev_joints =[]
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

    def update_grasp(self, target_pos, speed, time_step=1./240.):
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
        step = self.sign * speed / 240
        
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

        if np.abs(self.current_pos== target_pos) < 0.001:
            self.goal_stay += 1
        else:
            self.goal_stay = 0

        for j_id, j in enumerate(self.target_joint):
            joint_pose= self.multi[j_id]*self.current_pos+self.offset[j_id]
            p.setJointMotorControl2(bodyIndex=self.hand_id,
                                    jointIndex=j,
                                    controlMode=p.POSITION_CONTROL,  # mode = p.TORQUE_CONTROL
                                    targetPosition=joint_pose,
                                    force=self.forces[j_id],
                                    maxVelocity=5
                                    )
        
        if self.count_stay > 0.5/time_step or self.goal_stay > 2/time_step:
            return True
        else:
            return False


def main():
    physics_client = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optional
    p.setGravity(0, 0, -10)
    plane_id = p.loadURDF("plane.urdf")
    init_pos = [0, 0, 0.46]
    init_ori = p.getQuaternionFromEuler([0, math.pi, 0])
    hand = RobotGripper(init_pos, init_ori)
    mesh_scale = [0.001, 0.001, 0.001]

    # the visual shape and collision shape can be re-used by all createMultiBody instances (instancing)
    obj_pos = [0, 0, 0.2]
    obj_ori = p.getQuaternionFromEuler([0, math.pi/8, math.pi/2])

    model_fn = "./objs/obj_000003.obj"
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

    time.sleep(2)
    close_angle = -0.05
    to_grasp = close_angle
    speed = 0.5
    t_wait = 0
    t_wait2 = 0
    t = 0
    tool_pos = init_pos
    for i in range(100000):
        grasp = hand.update_grasp(to_grasp, speed)
        p.stepSimulation()
        time.sleep(TIME_STEP)  # possible to increase the speed of the simulation.
        t = t + TIME_STEP
        if t_wait2 > 0:
            t_wait2 += TIME_STEP
        if grasp or tool_pos > init_pos or t > 5:
            t_wait += TIME_STEP
            if t_wait > 4 and to_grasp == close_angle:
                p.changeConstraint(obj_constraint, maxForce=0)
                tool_pos[2] = min(tool_pos[2] + 0.0005, 0.7)
                p.changeConstraint(hand.base_constraint, tool_pos, init_ori, maxForce=100000)
            if tool_pos[2] == 0.7:
                t_wait2 += TIME_STEP
            if t_wait2 > 2 and to_grasp == close_angle:
                hand.reset_internal_val()
                to_grasp = 1
                speed = 10
        if t_wait2 > 10:
            break

    p.disconnect()


if __name__ == '__main__':
    main()

'''
Grasp pose is defined in the object coordinate space.
T_grasp @ obj_coordinate frame
I_object @ obj_coordinate_frame

1. Define T_grasp obj_coordinate frame
2. Transfrom the T_grasp to be aligned to [0,0,-1]

3. tool_Position = [0,0,0.5]
   tool_Orient = p.getQuaternionFromEuler([0,math.pi,0])
   T_grasp * T = T_target
   T_(obj2world)  = inv(T_grasp)*T_target
   Transform the object using T_(obj2world)s
4. Perform: simulation
5. Annotate, T_grasp -> success, duration of grasp (ms)
'''
