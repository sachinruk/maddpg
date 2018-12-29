import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from ddpg import DDPGAgent

import copy

BUFFER_SIZE = int(1e4)  # replay buffer size
BATCH_SIZE = 256        # minibatch size

TAU = 1e-3              # for soft update of target parameters
ACTOR_LR = 1e-4               # learning rate 
CRITIC_LR = 1e-3

GAMMA = 0.99 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            

class MADDPGAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, num_agents, state_size, action_size):
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

        self.agents = [DDPGAgent(num_agents, state_size, action_size, i) for i in range(num_agents)]

        network_params = (state_size, action_size, num_agents)
        self.memory = ReplayBuffer(*network_params, BUFFER_SIZE, BATCH_SIZE)

    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

    def act(self, states):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        actions = []
        for state, agent in zip(states, self.agents):
            actions.append(agent.act(state).detach().cpu().numpy())

        return np.concatenate(actions) #.squeeze()

    def learn(self):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        
        states, actions, rewards, next_states, dones = self.memory.sample()
        # N = len(states)

        actions_pred = []
        next_actions = []
        # for i, (p, pt) in enumerate(zip(self.policy_local, self.policy_target)):
        #     actions_pred.append(p(states[:,i,:]))
        #     next_actions.append(pt(next_states[:,i,:]))
        for i,agent in enumerate(self.agents):
            actions_pred.append(agent.policy_local(states[:,i,:]))
            next_actions.append(agent.policy_target(next_states[:,i,:]))
        next_actions = torch.cat(next_actions, dim=-1)
        future_noise = torch.clamp(0.1*torch.randn(next_actions.shape),-0.1,0.1).to(device)
        next_actions += future_noise
        next_actions = torch.clamp(next_actions, -1.0, 1.0)

        for agent in self.agents:
            agent.learn(states, actions, rewards, next_states, dones, next_actions, actions_pred)

    def reset(self):
        for agent in self.agents:
            agent.reset()
            
    def save_network(self):
        for agent in self.agents:
            agent.save_network()
        
    def load_network(self):
        for agent in self.agents:
            agent.load_network()
            

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, state_size, action_size, num_agents, buffer_size, batch_size, priority=False):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer (chosen as multiple of num agents)
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.states = torch.zeros(buffer_size, num_agents, state_size).to(device)
        self.next_states = torch.zeros(buffer_size, num_agents, state_size).to(device)
        self.actions = torch.zeros(buffer_size, num_agents, action_size).to(device)
        self.rewards = torch.zeros([buffer_size, num_agents, 1]).to(device)
        self.dones = torch.zeros([buffer_size, num_agents, 1]).to(device)
#         self.prob = None

        self.ptr = 0
        self.n = 0
        self.buffer_size = buffer_size
        self.batch_size = batch_size
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        # n = len(reward)
        self.states[self.ptr] = torch.from_numpy(state).to(device)
        self.next_states[self.ptr] = torch.from_numpy(next_state).to(device)
        self.actions[self.ptr] = torch.from_numpy(action).to(device)
        self.rewards[self.ptr] = torch.from_numpy(reward).to(device)
        self.dones[self.ptr] = torch.from_numpy(done).to(device)
        
        self.ptr += 1
        if self.ptr >= self.buffer_size:
            self.ptr = 0
            self.n = self.buffer_size

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
#         if self.prob is not None:
#             idx = np.random.choice(len(self.prob), self.batch_size, replace=False, p=self.prob)
#         else:
        idx = np.random.choice(len(self), self.batch_size, replace=False)
        
        states = self.states[idx]
        next_states = self.next_states[idx]
        actions = self.actions[idx]
        rewards = self.rewards[idx]
        dones = self.dones[idx]
        
        return (states, actions, rewards, next_states, dones)
        
    def __len__(self):
        if self.n == 0:
            return self.ptr
        else:
            return self.n