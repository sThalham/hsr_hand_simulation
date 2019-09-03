import pybullet as p
import time
import pybullet_data
import math
import numpy as np


class robot_gripper():
    def __init__(self,translation, orientation,is_open=True):
        self.handId = p.loadURDF("./hsrb_description/robots/hand.urdf",translation, orientation) #,flags=p.URDF_USE_SELF_COLLISION
        self.base_constraint = p.createConstraint(
                                parentBodyUniqueId = self.handId,
                                parentLinkIndex = -1,
                                childBodyUniqueId = -1,
                                childLinkIndex = -1,
                                jointType = p.JOINT_FIXED,
                                jointAxis = [0, 0, 0],
                                parentFramePosition = [0, 0, 0],
                                childFramePosition = translation,
                                childFrameOrientation = orientation)
        self.n_joints = p.getNumJoints(self.handId)
        self.joint_id=[]
        self.joint_names=[]
        self.target_joint_names = [b'hand_l_proximal_joint',b'hand_r_proximal_joint',b'hand_l_distal_joint',b'hand_r_distal_joint']
        self.target_joint=[]
        self.multi=[]
        self.offset = []
        self.forces=[]
        self.contact=[]
        for i in range(self.n_joints):
            joints = p.getJointInfo(self.handId,i)
            self.joint_id.append(joints[0])
            self.joint_names.append(joints[1])
            print(joints[0],joints[1],joints[12])
            if(joints[1] in self.target_joint_names):
                self.target_joint.append(joints[0])
                if(joints[1] in [b'hand_l_proximal_joint',b'hand_r_proximal_joint']):
                    print(joints[0],joints[1],joints[2])
                    self.multi.append(1)
                    self.offset.append(0)
                    self.forces.append(1)
                    self.contact.append(False)  
                else:#if(joints[1] in [b'hand_r_distal_joint']): #for distal joints (fingers)    
                    self.multi.append(-1)
                    self.offset.append(-0.087)
                    self.forces.append(1000)
                    self.contact.append(True)

        print(p.getBodyInfo(self.handId))
        for j_id,j in enumerate(self.target_joint):
            if(self.contact[j_id]):
                #dynamics = p.getDynamicsInfo(self.handId,j)
                
                p.changeDynamics(self.handId, linkIndex=j,rollingFriction=0.5,lateralFriction=0.1)#,contactStiffness=0.001,contactDamping=0.99)
            else:
                pass
                #p.changeDynamics(self.handId, linkIndex=j,contactStiffness=0.1,contactDamping=0.1)
        
        if is_open:
            t_pos = 1.2
        else:
            t_pos = 0
        self.prev_joints =[]
        for j_id,j in enumerate(self.target_joint):
            joint_val = t_pos*self.multi[j_id]+self.offset[j_id]
            p.resetJointState(self.handId, jointIndex=j,targetValue=joint_val)
            self.prev_joints.append(joint_val)
        self.current_pos= t_pos
        self.count_stay=0
        self.goal_stay=0
        self.sign=0

    def reset_internal_val(self):
        self.count_stay=0
        self.goal_stay=0
        self.sign=0
    def update_grasp(self,target_pos,speed,time_step=1./240.):
        '''
            [input]
            is_open:true for openning, false for closing
            speed: motor speed (radian/s)
            time_step: time step for the simulation (s)
            
            [return]
            true: finished (reach the goal or not moving anymore)
            false: moving
        '''
        if(self.sign==0):
            self.sign = np.sign(target_pos - self.current_pos)
        step = self.sign*speed/240
        
        #check current position of the gripper
        self.current_joints =[]
        diffs=[]
        for j_id,j in enumerate(self.target_joint):            
            joint_state= p.getJointState(self.handId, jointIndex=j)
            self.current_joints.append(joint_state[0])
            if not(self.contact[j_id]):
                diffs.append(np.abs(joint_state[0] - self.prev_joints[j_id]))
        diff = np.sum(diffs)
        if(diff<0.0001):
            self.count_stay+=1
        elif(self.count_stay>0 and diff<0.001):
            self.count_stay+=1
        else:
            self.count_stay=0

        self.prev_joints =self.current_joints
        self.current_pos =self.current_pos+step
        
        if(self.sign>0):
            self.current_pos  = min(self.current_pos,target_pos)
        else:
            self.current_pos  = max(self.current_pos,target_pos)

        if(np.abs(self.current_pos== target_pos)<0.001):
            self.goal_stay+=1
        else:
            self.goal_stay=0

        for j_id,j in enumerate(self.target_joint):
            joint_pose= self.multi[j_id]*self.current_pos+self.offset[j_id]
            p.setJointMotorControl2(bodyIndex=self.handId,
                                    jointIndex=j,
                                    controlMode=p.POSITION_CONTROL, #mode = p.TORQUE_CONTROL
                                    targetPosition=joint_pose,
                                    force=self.forces[j_id],
                                    maxVelocity=5
                                    )            
                                    #positionGain=1,
                                    #velocityGain=0.5)
        
        if(self.count_stay>0.5/time_step or self.goal_stay>2/time_step):
            return True
        else:
            return False
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")
initPos = [0,0,0.37]
initOri = p.getQuaternionFromEuler([0,math.pi,0])
hand = robot_gripper(initPos,initOri)

#meshScale = [0.05, 0.05 ,0.05]
meshScale = [0.001, 0.001 ,0.001]
#meshScale = [1, 1 ,1]
#the visual shape and collision shape can be re-used by all createMultiBody instances (instancing)
ObjPos = [0,0,0.1]
ObjOri = p.getQuaternionFromEuler([0,math.pi/8,math.pi/2])

model_fn = "./objs/obj_000003.obj"
visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,
                                    fileName=model_fn,
                                    rgbaColor=[1, 1, 1, 1],
                                    specularColor=[0.4, .4, 0],
                                    meshScale=meshScale)
collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                          fileName=model_fn,
                                          meshScale=meshScale)

target_obj = p.createMultiBody(baseMass=0.01,
                  baseInertialFramePosition=[0, 0, 0],
                  baseCollisionShapeIndex=collisionShapeId,
                  baseVisualShapeIndex=visualShapeId,
                  basePosition=ObjPos,
                  baseOrientation=ObjOri
                  )

obj_constraint = p.createConstraint(
                    parentBodyUniqueId = target_obj,
                    parentLinkIndex = -1,
                    childBodyUniqueId = -1,
                    childLinkIndex = -1,
                    jointType = p.JOINT_FIXED,
                    jointAxis = [0, 0, 0],
                    parentFramePosition = [0, 0, 0],
                    childFramePosition = ObjPos,
                    childFrameOrientation = ObjOri)
p.changeConstraint(obj_constraint,maxForce=1)

time.sleep(2)
close_angle= -0.05
tograsp=close_angle
speed=0.5
t_wait=0
t_wait2=0
t=0
toolPos=initPos
for i in range (100000):
    grasp = hand.update_grasp(tograsp,speed)
    p.stepSimulation()
    time.sleep(1./240.) #possible to increase the speed of the simulation.
    t=t+1/240
    if(t_wait2>0):
        t_wait2+=1.0/240.0
    if(grasp==True or toolPos>initPos or t>5):
        p.changeConstraint(obj_constraint,maxForce=0)
        t_wait+=1.0/240.0
        if(t_wait>4 and tograsp==close_angle):
           toolPos[2] = min(toolPos[2]+0.0005,0.7)
           p.changeConstraint(hand.base_constraint, toolPos, initOri, maxForce=100000)
        if(toolPos[2]==0.7): t_wait2+=1.0/240.0
        if(t_wait2>2 and tograsp==close_angle):
           hand.reset_internal_val()
           tograsp=1 
           speed = 10    
    if(t_wait2>10):
        break
    
p.disconnect()

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
