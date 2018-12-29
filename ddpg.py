import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

import copy

from models import Policy, Critic

# from collections import deque, namedtuple

BUFFER_SIZE = int(1e4)  # replay buffer size
BATCH_SIZE = 512        # minibatch size

TAU = 1e-3              # for soft update of target parameters
ACTOR_LR = 1e-4               # learning rate 
CRITIC_LR = 1e-3

GAMMA = 0.99

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            

class DDPGAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, num_agents, state_size, action_size, agent_id):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.agent_id = agent_id
        
        network_params = (state_size, action_size, num_agents)
        offset = num_agents * agent_id * 2 #offset for seed
        # Q-Network
        self.qnetwork_local = [Critic(*network_params, seed=offset+i).to(device) for i in range(2)]
        self.qnetwork_target = [Critic(*network_params, seed=offset+i).to(device) for i in range(2)]
        self.q_optimizer = [optim.Adam(net.parameters(), lr=CRITIC_LR, weight_decay=0) 
                            for net in self.qnetwork_local]
        self.huber_loss = torch.nn.SmoothL1Loss()
        # Policy Networks
        self.policy_local = Policy(*network_params, seed=offset).to(device)
        self.policy_target = Policy(*network_params, seed=offset).to(device)
        self.policy_optimizer = optim.Adam(self.policy_local.parameters(), lr=ACTOR_LR, weight_decay=0)

        self.__copy_parameters__()
        
        self.noise = OUNoise(action_size, seed=agent_id)
        self.noise_mul = 0.5
        # self.noise = GaussianNoise(action_size, seed=42)

        self.train_t = 0 # training iterations

    def __copy_parameters__(self):
        for i in range(2):
            self.soft_update(self.qnetwork_local[i], self.qnetwork_target[i], 1.0)
        self.soft_update(self.policy_local, self.policy_target, 1.0)
    
    # def step(self, state, action, reward, next_state, done):
    #     # Save experience in replay memory
    #     self.memory.add(state, action, reward, next_state, done)

    def act(self, state):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.policy_local.eval()
        with torch.no_grad():
            action_values = self.policy_local(state)
        self.policy_local.train()
        
        action_values += self.noise_mul * self.noise.sample()
        action_values = torch.clamp(action_values, -1, 1)
        return action_values #.squeeze()

    def learn(self, states, actions, rewards, next_states, dones, next_actions, actions_pred):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        # import pdb; pdb.set_trace()
        N = len(states)
        
        # import pdb; pdb.set_trace()
        target = [net(next_states.view(N,-1), next_actions).squeeze() for net in self.qnetwork_target]
        target = rewards[:,self.agent_id,:].squeeze() + \
                 GAMMA * torch.min(*target) * (1 - dones[:,self.agent_id,:].squeeze())
        target = target.detach()

        for i in range(2):
            y = self.qnetwork_local[i](states.view(N,-1), actions.view(N,-1)).squeeze()
            # import pdb; pdb.set_trace()
            # critic_loss = F.mse_loss(y, target)
            critic_loss = self.huber_loss(y, target)
            
            self.q_optimizer[i].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.qnetwork_local[i].parameters(), 1)
            self.q_optimizer[i].step()
            self.soft_update(self.qnetwork_local[i], self.qnetwork_target[i], TAU)
        
        if self.train_t % 2 == 0:
            actions_pred = [a if i==self.agent_id else a.detach() for i,a in enumerate(actions_pred)]
            # import pdb; pdb.set_trace()
            actions_pred = torch.cat(actions_pred, dim=-1)
            # import pdb; pdb.set_trace()
            policy_loss = -self.qnetwork_local[0](states.view(N,-1), actions_pred).mean()
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_local.parameters(), 1)
            self.policy_optimizer.step()
            # ------------------- update target network ------------------- #  
            self.soft_update(self.policy_local, self.policy_target, TAU)

        self.train_t += 1
        
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def reset(self):
        self.noise.reset()
        if self.train_t >= 7500:
            self.noise_mul = 0
            
    def save_network(self):
        self.bestq_local = [net.state_dict() for net in self.qnetwork_local]
        self.bestq_target = [net.state_dict() for net in self.qnetwork_target]
        self.bestpolicy_local = self.policy_local.state_dict()
        self.bestpolicy_target = self.policy_target.state_dict()
        
    def load_network(self):
        for i in range(2):
            self.qnetwork_local[i].state_dict().update(self.bestq_local[i])
            self.qnetwork_target[i].state_dict().update(self.bestq_target[i])
        self.policy_local.state_dict().update(self.bestpolicy_local)
        self.policy_target.state_dict().update(self.bestpolicy_target)
        
            
#     def update_priority(self):
#         n = len(self.memory)
#         states = self.memory.states[:n]
#         next_states = self.memory.next_states[:n]
#         actions = self.memory.actions[:n]
#         rewards = self.memory.rewards[:n]
#         dones = self.memory.dones[:n]
        
#         target = self.qnetwork_target(next_states, self.policy_target(next_states)).squeeze()
#         target = rewards.squeeze() + GAMMA * target * (1 - dones.squeeze())
#         y = self.qnetwork_local(states, actions).squeeze()

#         error = torch.abs(y - target).detach().cpu().numpy()
#         self.memory.prob = error / error.sum()

class GaussianNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, sigma=0.2, decay=0.999):
        """Initialize parameters and noise process."""
        self.size = size
        self.sigma = sigma
        self.decay = decay
        self.seed = torch.manual_seed(seed)
        # self.seed = np.random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.sigma = self.decay * self.sigma

    def sample(self):
        """Update internal state and return it as a noise sample."""
        return self.sigma * torch.randn(self.size, device=device)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * torch.ones(size, device=device)
        self.theta = theta
        self.sigma = sigma
        self.seed = torch.manual_seed(seed)
        # self.seed = np.random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * torch.randn(len(x), device=device)
        self.state = x + dx
        return self.state