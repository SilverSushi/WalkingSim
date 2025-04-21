# walking_sim_env.py
import pybullet as p
import pybullet_data
import gym
from gym import spaces
import numpy as np
from .simple_walking_env import SimpleWalkingSim
from utils.reward_function import compute_complex_reward

class WalkingSimEnv(gym.Env):
    def __init__(self, render=True, num_agents=25):
        super(WalkingSimEnv, self).__init__()
        self.num_agents = num_agents
        self.sim = SimpleWalkingSim(render=render, num_agents=num_agents)

        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_agents, self.sim.num_joints), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_agents, self.sim.num_joints * 2), dtype=np.float32)

    def reset(self):
        self.sim._load_environment()
        return self.sim.get_joint_states()

    def step(self, actions):
        obs_batch = []
        rewards = []
        dones = []

        # Ensure actions is a batch of actions, one per humanoid
        for i, action in enumerate(actions):
            self.sim.step_action(i, action)  # Pass humanoid index and action
            obs_batch.append(self.get_observation(i))  # Get the observation for each humanoid
            rewards.append(self.compute_reward(i, action))  # Compute reward for each humanoid
            dones.append(self.check_done(i))  # Check if the humanoid is done (with the correct index)

        return np.array(obs_batch), np.array(rewards), np.array(dones), {}

    def render(self):
        pass

    def get_observation(self, index):
        joint_states = [p.getJointState(self.sim.humanoids[index], i) for i in range(self.sim.num_joints)]
        positions = [state[0] for state in joint_states]
        velocities = [state[1] for state in joint_states]
        return np.concatenate([positions, velocities])

    def check_done(self, index):
        position, _ = p.getBasePositionAndOrientation(self.sim.humanoids[index])  # Use the index for the correct humanoid
        if abs(position[2]) < 0.5:  # Check if the humanoid has fallen
            return True
        return False
    
    def compute_reward(self, index, action):
        obs = self.get_observation(index)
        return compute_complex_reward(obs, action, self.sim)

    def close(self):
        self.sim.close()
