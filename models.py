import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Policy(nn.Module):
    def __init__(self, input_dim, action_dim, num_agents, h=(256, 256), seed=42):
        super(Policy, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_dim = input_dim
        self.num_agents = num_agents
        nodes = (input_dim,)+h+(action_dim,)
        self.bn = nn.BatchNorm1d(input_dim)
        self.fc_mu = nn.ModuleList([nn.Linear(x,y) for x,y in zip(nodes[:-1], nodes[1:])])
        
        self.reset_parameters()
        
    def reset_parameters(self):
        for layer in self.fc_mu[:-1]:
            f = layer.in_features
            lim = 1 / np.sqrt(f)
            layer.weight.data.uniform_(-lim,lim)
        self.fc_mu[-1].weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, x):
        # in_shape = x.shape
        # if len(x) > 1:
        #     import pdb; pdb.set_trace()
        # x = self.bn(x.reshape(-1, self.input_dim)).reshape(*in_shape)
        x = self.bn(x)
        for layer in self.fc_mu[:-1]:
            x = F.relu(layer(x))
            
        return torch.tanh(self.fc_mu[-1](x))
    
class Critic(nn.Module):
    def __init__(self, input_dim, action_dim, num_agents, h=(256, 256), seed=42):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_dim = (input_dim + action_dim) * num_agents
        
        self.bn = nn.BatchNorm1d(self.input_dim)
        nodes = (self.input_dim,)+h+(1,)
        self.fc = nn.ModuleList([nn.Linear(x,y) for x,y in zip(nodes[:-1], nodes[1:])])
        self.reset_parameters()
        
    def reset_parameters(self):
        for layer in self.fc[:-1]:
            f = layer.in_features
            lim = 1 / np.sqrt(f)
            layer.weight.data.uniform_(-lim,lim)
        
        self.fc[-1].weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, x, action):
        # if len(x) > 1:
        #     import pdb; pdb.set_trace()
        x = torch.cat([x, action], dim=-1)
        x = self.bn(x)
        for layer in self.fc[:-1]:
            x = F.relu(layer(x))

        return self.fc[-1](x)