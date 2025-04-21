from env.walking_sim_env import WalkingSimEnv
from models.ea_agent import EAAgent
import numpy as np

if __name__ == '__main__':
    env = WalkingSimEnv(render=True, num_agents=25)
    obs_dim = env.observation_space.shape[1]
    act_dim = env.action_space.shape[1]

    population = [EAAgent(obs_dim, act_dim) for _ in range(25)]

    for gen in range(100):
        obs = env.reset()
        total_rewards = np.zeros(25)
        dones = np.array([False] * 25)

        while not np.all(dones):
            actions = np.array([agent.predict(o) for agent, o in zip(population, obs)])

            print("Sample action:", actions[0])

            obs, rewards, new_dones, _ = env.step(actions)
            total_rewards += rewards * (~dones)
            dones = np.logical_or(dones, new_dones)

        top_indices = np.argsort(total_rewards)[-5:]
        elites = [population[i] for i in top_indices]

        new_population = []
        for _ in range(25):
            parent = np.random.choice(elites)
            child = parent.clone_and_mutate()
            new_population.append(child)
        population = new_population

        print(f"Generation {gen} — Best score: {total_rewards[top_indices[-1]]:.2f}")
