from env.walking_env import SimpleWalkingSim
import time

if __name__ == "__main__":
    while True:
        sim = SimpleWalkingSim(render=True)
        sim.run_simulation(steps=1000)
        sim.close()