# Date: 2024/8/1 9:06
# Author: cls1277
# Email: cls1277@163.com

from collections import deque
import random
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import numpy as np
from utils.prioritized_replay_memory import PrioritizedReplayMemory
from utils.transition_memory import TransitionMemory

class DQNAgent:

    def __init__(self, epsilon, min_epsilon, decay_rate, learning_rate, tau, gamma, batch_size,
                 q_network, target_network, max_memory_length, agent_index=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.experience_memory = deque(maxlen=max_memory_length)
        self.prioritized_memory = PrioritizedReplayMemory(max_length=max_memory_length, alpha=0.6,
                                                          beta=0.4, beta_annealing_steps=500000)
        self.last_observation = None
        self.last_action = None
        self.agent_index = agent_index
        self.q_network = q_network.to(self.device)
        self.target_network = target_network.to(self.device)
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.loss_history = []
        self.total_training_episodes = 0

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.decay_rate, self.min_epsilon)

    def policy(self, observation, done):
        observation = torch.tensor(observation, dtype=torch.float32).to(self.device)
        if self.agent_index == 0:
            max_action = 12
        elif self.agent_index == 1:
            max_action = 3
        else:
            print("Error: invalid agent type")
            return

        if done:
            return None
        elif random.random() <= self.epsilon:
            action = random.randint(0, max_action)
        else:
            self.q_network.eval()
            with torch.no_grad():
                qs = self.q_network(observation)
                action = np.argmax(qs.cpu().detach().numpy())
        self.q_network.train()
        return action

    def save_model(self, filename):
        print("Saving Q network...")
        if not os.path.isdir('checkpoints'):
            os.mkdir('checkpoints')
        network_state = {
            'net': self.q_network.state_dict(),
            'target': self.target_network.state_dict(),
            'epsilon': self.epsilon,
            'total_training_episodes': self.total_training_episodes
        }
        torch.save(network_state, f'./checkpoints/{filename}.pth')
        print("Save complete!")

    def load_model(self, filename):
        print("Loading model from checkpoint...")
        checkpoint = torch.load(f'./checkpoints/{filename}.pth')  # load checkpoint
        self.q_network.load_state_dict(checkpoint['net'])
        self.target_network.load_state_dict(checkpoint['target'])
        self.epsilon = checkpoint['epsilon']
        self.total_training_episodes = checkpoint['total_training_episodes']
        print("Load complete!")

    def push_memory(self, memory):
        assert (isinstance(memory, TransitionMemory))
        self.experience_memory.append(memory)

    def do_training_update(self):
        if self.batch_size == 0 or len(self.experience_memory) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.sample_random_experience(n=self.batch_size)
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q = self.target_network(next_states).detach()
        max_next_q = next_q.max(1)[0].unsqueeze(1)
        target_q = rewards + (1 - dones) * self.gamma * max_next_q
        target_q = target_q.detach()
        target_q = target_q.squeeze()
        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        self.loss_history.append(loss.item())
        loss.backward()
        self.optimizer.step()

    def sample_random_experience(self, n):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        experience_sample = random.sample(self.experience_memory, n)
        for memory in experience_sample:
            states.append(memory.state)
            actions.append(memory.action)
            rewards.append([memory.reward])
            next_states.append(memory.next_state)
            dones.append([memory.done])

        return (torch.tensor(states, dtype=torch.float32).to(self.device),
                torch.tensor(actions, dtype=torch.long).to(self.device),
                torch.tensor(rewards, dtype=torch.float32).to(self.device),
                torch.tensor(next_states, dtype=torch.float32).to(self.device),
                torch.tensor(dones, dtype=torch.int8).to(self.device))

    def do_prioritized_training_update(self, frame):
        if self.batch_size == 0 or len(self.prioritized_memory.memory) < self.batch_size:
            return
        states, actions, rewards, next_states, dones, importance_sampling_weights, selected_indices = self.prioritized_memory.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        importance_sampling_weights = importance_sampling_weights.to(self.device)
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q = self.target_network(next_states).detach()
        max_next_q = next_q.max(1)[0].unsqueeze(1)
        target_q = rewards + (1 - dones) * self.gamma * max_next_q
        target_q = target_q.detach()
        target_q = target_q.squeeze()
        loss = (current_q - target_q).pow(2)
        loss = loss * importance_sampling_weights
        new_priorities = loss + 0.00001
        loss = torch.mean(loss)
        self.optimizer.zero_grad()
        self.loss_history.append(loss.item())
        loss.backward()
        self.optimizer.step()
        self.prioritized_memory.update_priorities(selected_indices, new_priorities.detach().cpu().numpy())
        self.prioritized_memory.anneal_beta(frame)

    def update_target_network(self):
        for source_parameters, target_parameters in zip(self.q_network.parameters(), self.target_network.parameters()):
            target_parameters.data.copy_(self.tau * source_parameters.data + (1.0 - self.tau) * target_parameters.data)


class ATMAgentDQN(nn.Module):

    def __init__(self):
        super(ATMAgentDQN, self).__init__()
        self.fc1 = nn.Linear(36, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 13)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class VTMAgentDQN(nn.Module):

    def __init__(self):
        super(VTMAgentDQN, self).__init__()
        self.fc1 = nn.Linear(36, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x