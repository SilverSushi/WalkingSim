import pybullet as p
import pybullet_data
import numpy as np
import os
import time

class SimpleWalkingSim:
    def __init__(self, render=True):
        self.render_mode = render
        self.time_step = 1. / 24.
        self.max_force = 100

        if self.render_mode:
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)

        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.time_step)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self._load_environment()

    def _load_environment(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")

        humanoid_path = os.path.join(pybullet_data.getDataPath(), "humanoid/humanoid.urdf")
        start_pos = [0, 0, 1.4]
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.humanoid = p.loadURDF(humanoid_path, start_pos, start_orientation)

        self.num_joints = p.getNumJoints(self.humanoid)

        # Disable default motor control to allow custom torques
        for i in range(self.num_joints):
            p.setJointMotorControl2(
                self.humanoid,
                i,
                controlMode=p.VELOCITY_CONTROL,
                force=0
            )
        
        if self.render_mode:
            p.resetDebugVisualizerCamera(
                cameraDistance=2.5,
                cameraYaw=50,
                cameraPitch=-30,
                cameraTargetPosition=start_pos
            )
    def step_random(self):
        # Generate random torques for each joint
        action = np.random.uniform(-1, 1, self.num_joints)
        for i in range(self.num_joints):
            p.setJointMotorControl2(
                self.humanoid,
                i,
                controlMode=p.TORQUE_CONTROL,
                force=action[i] * self.max_force
            )
        p.stepSimulation()
        if self.render_mode:
            time.sleep(self.time_step)

    def run_simulation(self, steps=1000):
        for _ in range(steps):
            self.step_random()

    def close(self):
        p.disconnect()

