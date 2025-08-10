import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import time
from collections import namedtuple
import math
import random

class UR5RobotiqEnv(gym.Env):
    def __init__(self):
        super(UR5RobotiqEnv, self).__init__()

        # Connect to PyBullet
        self.physics_client = p.connect(p.GUI)
        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # Set the simulation time step to 1/300 for faster calculations
        p.setTimeStep(1 / 300)
        # Action space: [x, y, z] target position for the end-effector
        self.action_space = spaces.Box(low=np.array([0.3, -0.3]), high=np.array([0.7, 0.3]), dtype=np.float64)

        # Observation space: [x, y, z] position of the target object
        self.observation_space = spaces.Box(low=np.array([0.3, -0.3]), high=np.array([0.7, 0.3]), dtype=np.float64)

        # Load environment objects
        self.plane_id = p.loadURDF("plane.urdf")
        self.table_id = p.loadURDF("table/table.urdf", [0.5, 0, 0], p.getQuaternionFromEuler([0, 0, 0]))
        self.tray_id = p.loadURDF("tray/tray.urdf", [0.5, 0.9, 0.6], p.getQuaternionFromEuler([0, 0, 0]))
        self.cube_id2 = p.loadURDF("cube.urdf", [0.5, 0.9, 0.3], p.getQuaternionFromEuler([0, 0, 0]), globalScaling=0.6, useFixedBase=True)
      
        # Set GUI viewing angle
        self.set_gui_view()
        # Load the robot
        self.robot = UR5Robotiq85([0, 0, 0.62], [0, 0, 0])
        self.robot.load()

        # Initialize cube
        self.cube_id = None
        # Set the maximum number of steps
        self.max_steps = 100
        self.current_step = 0
        self.gripper_range = [0, 0.085]  # [fully closed, fully open]
        
        # change gripper friction
        for link_id in [12, 17]:
            p.changeDynamics(self.robot.id, link_id,
                     lateralFriction=1000.0,
                     spinningFriction=1.0,
                     frictionAnchor=1)

    def set_gui_view(self):
        """
        Set the GUI camera view (not actual camera capture)
        """
        camera_distance = 1.1     # Distance from target
        camera_yaw = 90           # Left/right rotation
        camera_pitch = -45        # Up/down tilt
        camera_target = [0.5, 0, 0.6]  # Look-at point (center of table)

        p.resetDebugVisualizerCamera(cameraDistance=camera_distance,
                                     cameraYaw=camera_yaw,
                                     cameraPitch=camera_pitch,
                                     cameraTargetPosition=camera_target)

    def draw_boundary(self, x_range, y_range, z_height):
        """
        Draw a boundary box for the specified x and y ranges.
        :param x_range: List containing min and max values for x-coordinate.
        :param y_range: List containing min and max values for y-coordinate.
        :param z_height: Height (z-coordinate) at which the boundary box will be drawn.
        """
        corners = [
            [x_range[0], y_range[0], z_height],  # Bottom-left
            [x_range[1], y_range[0], z_height],  # Bottom-right
            [x_range[1], y_range[1], z_height],  # Top-right
            [x_range[0], y_range[1], z_height],  # Top-left
        ]

        # Draw lines between the corners to form a box
        for i in range(len(corners)):
            p.addUserDebugLine(corners[i], corners[(i + 1) % len(corners)], [1, 0, 0], lineWidth=2)

    def reset(self, seed=None, options=None):
        """
        Reset the environment.
        """
        self.current_step = 0
        self.robot.orginal_position(self.robot)
        # Reset cube position
        x_range = np.arange(0.4, 0.7, 0.2)
        y_range = np.arange(-0.3, 0.3, 0.2)

        cube_start_pos = [
            np.random.choice(x_range),
            np.random.choice(y_range),
            0.63
        ]
        x_draw_range = [0.3, 0.7]
        y_draw_range = [-0.3, 0.3]
        # Draw the boundary box
        self.draw_boundary(x_draw_range, y_draw_range, 0.63)
        cube_start_orn = p.getQuaternionFromEuler([0, 0, 0])
        if self.cube_id:
            p.resetBasePositionAndOrientation(self.cube_id, cube_start_pos, cube_start_orn)
        else:
            self.cube_id = p.loadURDF("./urdf/cube_blue.urdf", cube_start_pos, cube_start_orn)

        # Store the initial position of the cube for comparison
        self.initial_cube_pos = np.array(cube_start_pos[:2])

        # Get initial cube position for observation
        self.target_pos = np.array(cube_start_pos[:2])
        observation = self.target_pos
        info = {}
        return observation, info

    def gripper_close(self):
        grip_value = self.gripper_range[1]  

        while True:
            contact_point = p.getContactPoints(bodyA=self.robot.id)

            force = {}
            if len(contact_point) > 0:
                for i in contact_point:
                    link_index = i[2]
                    if force.get(link_index) is None:
                        force[link_index] = {17: 0, 12: 0}
                    if i[3] == 17:
                        if i[9] > force[link_index][17]:
                            force[link_index][17] = i[9]
                    elif i[3] == 12:
                        if i[9] > force[link_index][12]:
                            force[link_index][12] = i[9]

            #  Stop immediately when force is detected
            for link_index in force:
                if force[link_index][17] > 3 and force[link_index][12] > 3:
                    print(f"[Grasped] Link {link_index}: joint 17 = {force[link_index][17]:.2f}, joint 12 = {force[link_index][12]:.2f}")
                    return True

            #  Print current force status (for debugging)
            for link_index in force:
                for joint in [17, 12]:
                    if force[link_index][joint] > 0:
                        print(f"Link {link_index}, joint {joint} force: {force[link_index][joint]:.2f}")

            if grip_value <= self.gripper_range[0]:  # Already fully closed
                break

            grip_value -= 0.001
            self.robot.move_gripper(grip_value)

            for _ in range(60):
                p.stepSimulation()

        return False

    def step(self, action):
        """
        Perform an action in the environment.
        :param action: [x, y, z] target position for the end-effector
        """
        self.current_step += 1
        action = np.clip(action, self.action_space.low, self.action_space.high)

        eef_state = p.getLinkState(self.robot.id, self.robot.eef_id)
        eef_position = eef_state[0]
        eef_orientation = eef_state[1]

        target_pos = np.array([action[0], action[1], 0.88]) 
        self.robot.move_arm_ik(target_pos, eef_orientation)
        for _ in range(100):
            p.stepSimulation()

        eef_state = self.robot.get_current_ee_position()
        eef_position = np.array(eef_state[0])[:2]

        distance_to_target = abs(np.linalg.norm(eef_position - self.target_pos))
        if distance_to_target <= 0.01:
            steps_taken = self.max_steps - self.current_step
            reward = 100
            reward += max(0, (steps_taken * 1))
            
            print(f"Cube picked. {self.target_pos[0], self.target_pos[1]} picked successfully, distance {distance_to_target}, reward: {reward}")
        
            target_pos = np.array([action[0], action[1], 0.8]) 
            self.robot.move_arm_ik(target_pos, eef_orientation)
            for _ in range(100):
                p.stepSimulation()
                time.sleep(0.01)

            success = self.gripper_close()
    
            if success:
                print("Grasp successful!")
                time.sleep(0.5)
                self.lift_object_slowly(
                    start_pos=np.array([action[0], action[1], 0.8]),
                    end_z=1.0,
                    eef_orientation=eef_orientation
                )
                p.addUserDebugText(f"Success Pick", textColorRGB=[0, 0, 255], textPosition=[0.5, -1.1, 0.9],
                                textSize=2, lifeTime=1)
            else:
                print("Grasp failed.")
          
            time.sleep(0.5)
            done = True
        elif self.current_step >= self.max_steps:
            reward = -10 * (distance_to_target)
            done = True
        else:
            reward = -10 * (distance_to_target)
            done = False
     
        print(f"reward:{reward}\n")
        print(f"Distance difference: {distance_to_target}")
        observation = self.target_pos
        truncated = False
        info = {}
 
        return observation, reward, done, truncated, info

    def lift_object_slowly(self, start_pos, end_z, eef_orientation,
                            steps=30, sim_steps_per_move=5, sleep_time=0.005):
        """
        Faster smooth lifting
        """
        for i in range(steps):
            intermediate_z = start_pos[2] + (end_z - start_pos[2]) * (i + 1) / steps
            lift_pos = np.array([start_pos[0], start_pos[1], intermediate_z])
            self.robot.move_arm_ik(lift_pos, eef_orientation)

            for _ in range(sim_steps_per_move):
                p.stepSimulation()
                if sleep_time > 0:
                    time.sleep(sleep_time)

    def close(self):
        p.disconnect()


class UR5Robotiq85:
    def __init__(self, pos, ori):
        self.base_pos = pos
        self.base_ori = p.getQuaternionFromEuler(ori)
        self.eef_id = 7
        self.arm_num_dofs = 6
        self.arm_rest_poses = [-1.57, -1.54, 1.34, -1.37, -1.57, 0.0]
        self.gripper_range = [0, 0.085]
        self.max_velocity = 10

    def load(self):
        self.id = p.loadURDF('./urdf/ur5_robotiq_85.urdf', self.base_pos, self.base_ori, useFixedBase=True)
        self.__parse_joint_info__()
        self.__setup_mimic_joints__()
        
    def __parse_joint_info__(self):
        jointInfo = namedtuple('jointInfo',
                               ['id', 'name', 'type', 'lowerLimit', 'upperLimit', 'maxForce', 'maxVelocity', 'controllable'])
        self.joints = []
        self.controllable_joints = []

        for i in range(p.getNumJoints(self.id)):
            info = p.getJointInfo(self.id, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = info[2]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = jointType != p.JOINT_FIXED
            if controllable:
                self.controllable_joints.append(jointID)
            self.joints.append(
                jointInfo(jointID, jointName, jointType, jointLowerLimit, jointUpperLimit, jointMaxForce, jointMaxVelocity, controllable)
            )

        self.arm_controllable_joints = self.controllable_joints[:self.arm_num_dofs]
        self.arm_lower_limits = [j.lowerLimit for j in self.joints if j.controllable][:self.arm_num_dofs]
        self.arm_upper_limits = [j.upperLimit for j in self.joints if j.controllable][:self.arm_num_dofs]
        self.arm_joint_ranges = [ul - ll for ul, ll in zip(self.arm_upper_limits, self.arm_lower_limits)]

    def __setup_mimic_joints__(self):
        mimic_parent_name = 'finger_joint'
        mimic_children_names = {
            'right_outer_knuckle_joint': 1,
            'left_inner_knuckle_joint': 1,
            'right_inner_knuckle_joint': 1,
            'left_inner_finger_joint': -1,
            'right_inner_finger_joint': -1
        }
        self.mimic_parent_id = [joint.id for joint in self.joints if joint.name == mimic_parent_name][0]
        self.mimic_child_multiplier = {joint.id: mimic_children_names[joint.name] for joint in self.joints if joint.name in mimic_children_names}

        for joint_id, multiplier in self.mimic_child_multiplier.items():
            c = p.createConstraint(self.id, self.mimic_parent_id, self.id, joint_id,
                                   jointType=p.JOINT_GEAR, jointAxis=[0, 1, 0],
                                   parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
            p.changeConstraint(c, gearRatio=-multiplier, maxForce=100, erp=1)

    def move_gripper(self, open_length):
        """
        Control the gripper to open or close.
        :param open_length: Target width for gripper opening (0 ~ 0.085m)
        """
        open_length = max(self.gripper_range[0], min(open_length, self.gripper_range[1]))
        open_angle = 0.715 - math.asin((open_length - 0.010) / 0.1143)
        p.setJointMotorControl2(self.id, self.mimic_parent_id, p.POSITION_CONTROL, targetPosition=open_angle,
                                force=50, maxVelocity=self.joints[self.mimic_parent_id].maxVelocity)

    def move_arm_ik(self, target_pos, target_orn):
        joint_poses = p.calculateInverseKinematics(
            self.id, self.eef_id, target_pos, target_orn,
            lowerLimits=self.arm_lower_limits,
            upperLimits=self.arm_upper_limits,
            jointRanges=self.arm_joint_ranges,
            restPoses=self.arm_rest_poses,
        )
        for i, joint_id in enumerate(self.arm_controllable_joints):
            p.setJointMotorControl2(self.id, joint_id, p.POSITION_CONTROL, joint_poses[i], maxVelocity=self.max_velocity)

    def get_current_ee_position(self):
        return p.getLinkState(self.id, self.eef_id)

    def orginal_position(self, robot):
        # Set the initial posture for the robot arm to approach the cube
        target_joint_positions = [0, -1.57, 1.57, -1.5, -1.57, 0.0]
        for i, joint_id in enumerate(robot.arm_controllable_joints):
            p.setJointMotorControl2(robot.id, joint_id, p.POSITION_CONTROL, target_joint_positions[i])
        for _ in range(100):
            p.stepSimulation()
        self.move_gripper(0.085)
        for _ in range(3500):
            p.stepSimulation()
