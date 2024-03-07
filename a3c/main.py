import gymnasium as gym
import numpy as np
from a3c import Agent
from utils import plot_learning_curve
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.distributions
from nets import ActorCritic


class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(
            params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
        )

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = torch.zeros(1)
                state["exp_avg"] = p.data.new().resize_as_(p.data).zero_()
                state["exp_avg_sq"] = p.data.new().resize_as_(p.data).zero_()

                state["exp_avg"].share_memory_()
                state["exp_avg_sq"].share_memory_()
                # state["step"].share_memory()


if __name__ == "__main__":
    lr = 1e-4
    env_id = "CartPole-v1"
    n_actions = 2
    input_dims = [4]
    global_actor_critic = ActorCritic(input_dims, n_actions)
    global_actor_critic.share_memory()
    optimizer = SharedAdam(global_actor_critic.parameters(), lr=lr, betas=(0.92, 0.999))
    global_ep = mp.Value("i", 0)
    n_threads = 8
    # n_threads = mp.cpu_count()
    res_queue = mp.Queue()
    workers = [
        Agent(
            global_actor_critic,
            optimizer,
            input_dims,
            n_actions,
            gamma=0.99,
            lr=lr,
            name=i,
            global_ep_idx=global_ep,
            env_id=env_id,
        )
        for i in range(n_threads)
    ]

    [w.start() for w in workers]
    res = []
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]

    # plot_learning_curve(res)
