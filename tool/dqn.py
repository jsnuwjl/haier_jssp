import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.init import kaiming_uniform_, zeros_


def init_one_layer(in_features, out_features):
    layer = nn.Linear(in_features, out_features)
    kaiming_uniform_(layer.weight)
    zeros_(layer.bias)
    return layer


class Net(nn.Module):
    def __init__(self, layer_sizes, device):
        super(Net, self).__init__()
        self.layer_sizes = layer_sizes
        self.layers = []
        for layer_idx in range(1, len(self.layer_sizes)):
            layer = init_one_layer(self.layer_sizes[layer_idx - 1], self.layer_sizes[layer_idx])
            self.layers.append(layer)
        self.layers = nn.ModuleList(self.layers)
        self.to(device)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.leaky_relu(layer(x))
        x = self.layers[-1](x)
        return x


class DQN:
    def __init__(self, config):
        self.config = config
        self.device = config['device']
        self.n_states = config['layer_sizes'][0]
        self.n_actions = config['layer_sizes'][-1]
        self.epsilon = config['epsilon']
        self.epsilon_decay_coefficient = config['epsilon_decay_coefficient']
        self.epsilon_decay_interval = config['epsilon_decay_interval']
        self.memory_size = config['memory_size']
        self.batch_size = config['batch_size']
        self.gamma = config['gamma']
        self.target_replace_interval = config['target_replace_interval']
        self.eval_net = Net(config['layer_sizes'], config['device'])
        self.target_net = Net(config['layer_sizes'], config['device'])

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.episode = 0
        self.memory = np.zeros((self.memory_size, self.n_states * 2 + 3), dtype=np.float32)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=config['lr'], weight_decay=config['l2'])
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.tensor(x, device=self.device, dtype=torch.float32), 0)
        if np.random.uniform() > self.epsilon:
            with torch.no_grad():
                actions_value = self.eval_net(x)
                action = torch.max(actions_value, dim=1)[1].cpu().numpy()
                action = action[0]
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def store_transition(self, s, a, r, s_, done):
        transition = np.hstack((s, [a, r, done], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def add_episode(self):
        self.episode += 1
        if self.episode % self.epsilon_decay_interval == 0:
            self.epsilon = self.epsilon_decay_coefficient * (self.epsilon + 0.1)
        if self.epsilon < 0.01:
            self.epsilon = 0

    def learn(self):
        if self.learn_step_counter % self.target_replace_interval == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(min(self.memory_size, self.memory_counter), self.batch_size)
        b_memory = self.memory[sample_index, :]
        b_s = torch.tensor(b_memory[:, :self.n_states], device=self.device, dtype=torch.float32)
        b_a = torch.tensor(b_memory[:, self.n_states:self.n_states + 1].astype(int), device=self.device,
                           dtype=torch.long)
        b_r = torch.tensor(b_memory[:, self.n_states + 1:self.n_states + 2], device=self.device, dtype=torch.float32)
        b_d = torch.tensor(b_memory[:, self.n_states + 2:self.n_states + 3], device=self.device, dtype=torch.float32)
        b_s_ = torch.tensor(b_memory[:, -self.n_states:], device=self.device, dtype=torch.float32)

        q_eval = self.eval_net(b_s).gather(1, b_a)
        with torch.no_grad():
            q_next = self.target_net(b_s_)
        q_target = b_r + self.gamma * q_next.max(dim=1)[0].view(self.batch_size, 1) * (1. - b_d)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, path):
        params = {'state_dict': self.eval_net.state_dict(), 'config': self.config, 'epsilon': self.epsilon}
        torch.save(params, path)
