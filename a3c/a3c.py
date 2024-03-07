import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from nets import ActorCritic
from buffer import A3CMemory
import torch.multiprocessing as mp
import gymnasium as gym


class Agent(mp.Process):
    def __init__(
        self,
        global_actor_critic,
        optimizer,
        input_dims,
        n_actions,
        gamma,
        lr,
        name,
        global_ep_idx,
        env_id,
    ):
        super(Agent, self).__init__()
        self.local_actor_critic = ActorCritic(input_dims, n_actions, gamma)
        self.global_actor_critic = global_actor_critic
        self.name = "w%02i" % name
        self.global_ep_idx = global_ep_idx
        self.env = gym.make(env_id)
        self.optimizer = optimizer
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.gamma = gamma
        self.lr = lr

    def run(self):
        t_step = 1
        while self.global_ep_idx.value < 3000:
            done = False
            truncated = False
            observation, info = self.env.reset()
            score = 0
            self.local_actor_critic.clear_memory()
            while not done and not truncated:
                action = self.local_actor_critic.choose_action(observation)
                observation_, reward, done, truncated, info = self.env.step(action)
                # observation_, reward, done, info = self.env.step(action)
                score += reward
                self.local_actor_critic.remember(observation, action, reward)
                if t_step % 5 == 0 or done:
                    loss = self.local_actor_critic.calc_loss(done)
                    self.optimizer.zero_grad()
                    loss.backward()
                    for local_param, global_param in zip(
                        self.local_actor_critic.parameters(),
                        self.global_actor_critic.parameters(),
                    ):
                        global_param._grad = local_param.grad
                    self.optimizer.step()
                    self.local_actor_critic.load_state_dict(
                        self.global_actor_critic.state_dict()
                    )
                observation = observation_
                t_step += 1
            with self.global_ep_idx.get_lock():
                self.global_ep_idx.value += 1
            print(
                self.name,
                "episode",
                self.global_ep_idx.value,
                "reward",
                score,
                "time_step",
                t_step,
            )
