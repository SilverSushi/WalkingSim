import numpy as np
import pybullet as p

def compute_complex_reward(obs, action, sim, max_force=100):
    """
    A more complex reward function for the walking humanoid.
    
    Parameters:
        obs (numpy array): Current observation (positions and velocities of joints).
        action (numpy array): Current action (torques applied to joints).
        sim (SimpleWalkingSim): The walking simulation object.
        max_force (float): The maximum force the joints can apply.
    
    Returns:
        float: A calculated reward value.
    """
    # Unpack the observation
    joint_positions = obs[:sim.num_joints]
    joint_velocities = obs[sim.num_joints:]
    
    # Compute forward velocity (reward for moving forward)
    base_position, base_velocity = p.getBasePositionAndOrientation(sim.humanoids[0])
    forward_velocity = base_velocity[0]  # x component of the velocity
    velocity_reward = forward_velocity  # Reward for moving forward
    #Maybe normalize?

    # Penalize for falling (low z position means falling)
    balance_penalty = -1.0 if base_position[2] < 1.5 else 0.0
    
    # Energy consumption: penalize for excessive action values (large torques)
    energy_penalty = -np.sum(np.square(action)) / max_force  # Penalize large actions
    
    # Smoothness of movement: penalize large changes in joint velocities
    joint_velocity_changes = np.abs(np.diff(joint_velocities))  # Get the velocity change
    smoothness_penalty = -np.sum(joint_velocity_changes)
    
    # Combine all the terms into the total reward
    reward = 0.5 * velocity_reward + balance_penalty + 100 * energy_penalty + 0.1 * smoothness_penalty
    #print(f"Velocity: {velocity_reward}, Balance: {balance_penalty}, Energy: {energy_penalty}, Smoothness: {smoothness_penalty}")
    # Velocity: 0.7072385791537584, Balance: 0.0, Energy: -0.0009827752411365508, Smoothness: -0.20822533587348085
    return reward
