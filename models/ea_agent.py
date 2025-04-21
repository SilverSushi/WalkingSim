import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np

class EAAgent(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(EAAgent, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, act_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

    def predict(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            action = self.forward(obs)
        return action.numpy()

    def clone_and_mutate(self, mutation_rate=0.01):
        new_agent = copy.deepcopy(self)
        with torch.no_grad():
            for param in new_agent.parameters():
                param.add_(torch.randn_like(param) * mutation_rate)
        return new_agent
