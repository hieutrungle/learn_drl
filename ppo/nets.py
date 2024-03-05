import os
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.optim as optim


class CriticNetwork(nn.Module):
    """
    The critic network is used to evaluate the value of the state-action pair.
    """

    def __init__(
        self,
        input_dims,
        alpha,
        fc1_dims=512,
        fc2_dims=512,
        name="critic",
        chkpt_dir="results/ppo",
    ):
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(
            self.checkpoint_dir, self.model_name + "_ppo.h5"
        )

        self.critic = nn.Sequential(
            nn.Linear(*input_dims, self.fc1_dims),
            nn.ReLU(),
            nn.Linear(self.fc1_dims, self.fc2_dims),
            nn.ReLU(),
            nn.Linear(self.fc2_dims, 1),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)
        return value

    def save_checkpoint(self):
        print("... saving checkpoint ...")
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("... loading checkpoint ...")
        self.load_state_dict(torch.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    """
    The actor network is used to determine the best action for a given state.
    """

    def __init__(
        self,
        input_dims,
        n_actions,
        fc1_dims=512,
        fc2_dims=512,
        alpha=0.0001,
        name="actor",
        chkpt_dir="results/ppo",
    ):
        super(ActorNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(
            self.checkpoint_dir, self.model_name + "_ppo.h5"
        )

        self.actor = nn.Sequential(
            nn.Linear(*input_dims, self.fc1_dims),
            nn.ReLU(),
            nn.Linear(self.fc1_dims, self.fc2_dims),
            nn.ReLU(),
            nn.Linear(self.fc2_dims, self.n_actions),
            nn.Softmax(dim=-1),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        probs = self.actor(state)
        distribution = Categorical(probs)
        return distribution

    def save_checkpoint(self):
        print("... saving checkpoint ...")
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("... loading checkpoint ...")
        self.load_state_dict(torch.load(self.checkpoint_file))
