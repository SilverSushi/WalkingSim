import pybullet as p
import pybullet_data
import numpy as np
import os
import time

class SimpleWalkingSim:
    def __init__(self, render=True, num_agents=25):
        self.render_mode = render
        self.time_step = 1. / 12.
        self.max_force = 100
        self.num_agents = num_agents

        self.physicsClient = p.connect(p.DIRECT)  

        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.time_step)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.humanoids = []
        self._load_environment()

    def _load_environment(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")

        self.humanoids = []
        grid_size = int(np.sqrt(self.num_agents))
        spacing = 3.0  # Increased for better spacing

        for idx in range(self.num_agents):
            row = idx // grid_size
            col = idx % grid_size
            start_pos = [row * spacing, col * spacing, 3.4]
            start_orientation = p.getQuaternionFromEuler([np.pi / 2, 0, 0])
            humanoid_path = os.path.join(pybullet_data.getDataPath(), "humanoid/humanoid.urdf")
            humanoid = p.loadURDF(humanoid_path, start_pos, start_orientation)
            self.humanoids.append(humanoid)

        self.num_joints = p.getNumJoints(self.humanoids[0])
        for humanoid in self.humanoids:
            for j in range(self.num_joints):
                p.setJointMotorControl2(humanoid, j, controlMode=p.VELOCITY_CONTROL, force=0)

        for i in range(len(self.humanoids)):
            for j in range(i + 1, len(self.humanoids)):
                humanoidA = self.humanoids[i]
                humanoidB = self.humanoids[j]
                for linkA in range(-1, self.num_joints):  # -1 = base
                    for linkB in range(-1, self.num_joints):
                        p.setCollisionFilterPair(humanoidA, humanoidB, linkA, linkB, enableCollision=0)

        if self.render_mode:
            p.resetDebugVisualizerCamera(
                cameraDistance=17,
                cameraYaw=45,
                cameraPitch=-35,
                cameraTargetPosition=[spacing * grid_size / 2, spacing * grid_size / 2, 1.0]
            )

    def step_action(self, index, action):
        for i in range(self.num_joints):
            p.setJointMotorControl2(
                self.humanoids[index],
                i,
                controlMode=p.TORQUE_CONTROL,
                force=action[i] * self.max_force
            )
        p.stepSimulation()
        if self.render_mode:
            time.sleep(self.time_step)

    def get_joint_states(self):
        obs = []
        for humanoid in self.humanoids:
            joint_states = [p.getJointState(humanoid, i) for i in range(self.num_joints)]
            positions = [state[0] for state in joint_states]
            velocities = [state[1] for state in joint_states]
            obs.append(np.concatenate([positions, velocities]))
        return np.array(obs)

    def get_positions(self):
        return [p.getBasePositionAndOrientation(h)[0] for h in self.humanoids]

    def disconnect(self):
        p.disconnect()