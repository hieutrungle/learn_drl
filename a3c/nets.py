import os
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions, gamma=0.99):
        super(ActorCritic, self).__init__()

        self.gamma = gamma

        self.pi1 = nn.Linear(*input_dims, 128)
        self.pi = nn.Linear(128, n_actions)

        self.v1 = nn.Linear(*input_dims, 128)
        self.v = nn.Linear(128, 1)

        self.rewards = []
        self.actions = []
        self.states = []

    def remember(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def forward(self, state):
        pi1 = F.relu(self.pi1(state))
        pi = self.pi(pi1)

        v1 = F.relu(self.v1(state))
        v = self.v(v1)

        return pi, v

    def calc_R(self, done, states, gamma):
        states = torch.tensor(states, dtype=torch.float)
        _, v = self.forward(states)
        R = v[-1] * (1 - int(done))

        batch_return = []
        for reward in self.rewards[::-1]:
            R = reward + gamma * R
            batch_return.append(R)
        batch_return.reverse()
        return torch.tensor(batch_return, dtype=torch.float)

    def calc_loss(self, done, gamma=0.99):
        states = torch.tensor(np.array(self.states), dtype=torch.float)
        actions = torch.tensor(np.array(self.actions), dtype=torch.float)
        returns = self.calc_R(done, self.states, gamma)

        pi, values = self.forward(states)
        values = values.squeeze()
        # returns = returns.squeeze()

        critic_loss = (returns - values) ** 2

        probs = torch.softmax(pi, dim=1)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        actor_loss = -log_probs * (returns - values)

        total_loss = (actor_loss + critic_loss).mean()
        return total_loss

    def choose_action(self, observation):
        observation = torch.tensor(np.array([observation]), dtype=torch.float)
        pi, v = self.forward(observation)
        probs = F.softmax(pi, dim=1)
        action_probs = torch.distributions.Categorical(probs)
        action = action_probs.sample().numpy()[0]
        return action
