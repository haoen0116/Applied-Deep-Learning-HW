import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agent_dir.agent import Agent
from environment import Environment
from torch.distributions import Categorical

import matplotlib.pyplot as plt


class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_num, hidden_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_num)
        # #加的
        self.saved_log_probs = []

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        action_prob = F.softmax(x, dim=1)
        return action_prob

class actor_critic(nn.Module):
    def __init__(self):
        super(actor_critic, self).__init__()
        self.affine = nn.Linear(8, 84)
        self.action_layer = nn.Linear(84, 4)
        self.value_layer = nn.Linear(84, 1)

        self.state_values = []
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, state):
        state = state.cpu().numpy()
        state = torch.from_numpy(state).float()
        k=self.affine(state)
        state = F.relu(k)
        state_value = self.value_layer(state)

        action_probs = F.softmax(self.action_layer(state))
        action_distrib = Categorical(action_probs)
        action = action_distrib.sample()

        self.saved_log_probs.append(action_distrib.log_prob(action))
        self.state_values.append(state_value)
        #print("action.item()", type(action.item()))   # int
        return action.item()

class AgentPG(Agent):
    def __init__(self, env, args):
        self.env = env
        # self.model = PolicyNet(state_dim=self.env.observation_space.shape[0],
        #                        action_num=self.env.action_space.n,
        #                        hidden_dim=64)
        self.model = actor_critic()
        if args.test_pg:
            self.load('pg_improve.cpt')
        # discounted reward
        self.gamma = 0.99

        # training hyperparameters
        # self.num_episodes = 100000 # total training episodes (actually too large...)
        self.num_episodes = 1000  # total training episodes (actually too large...)
        self.display_freq = 10  # frequency to display training progress

        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-3)

        # saved rewards and actions
        self.rewards, self.saved_actions = [], []
        # print(self.model)

    def save(self, save_path):
        print('save model to', save_path)
        torch.save(self.model.state_dict(), save_path)

    def load(self, load_path):
        print('load model from', load_path)
        self.model.load_state_dict(torch.load(load_path))

    def init_game_setting(self):
        self.rewards, self.saved_actions = [], []

    def make_action(self, state, test=False):
        # TODO:
        # Use your model to output distribution over actions and sample from it.
        # HINT: google torch.distributions.Categorical
        # print(type(state)) #ndarray

        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.model(state)
        action = probs
        return action

    def update(self):
        R = 0
        policy_loss = []
        returns = []
        # print("self.rewards",self.rewards)
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.num_episodes)

        for log_prob, value, R in zip(self.model.saved_log_probs, self.model.state_values, returns):
            policy_loss.append(-log_prob * R)
        # print("policy_loss",policy_loss)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        del self.rewards[:]
        del self.model.saved_log_probs[:]

    def train(self):
        avg_reward = None  # moving average of reward
        avg_reward_list = []
        epoch_list = []
        for epoch in range(self.num_episodes):
            state = self.env.reset()
            self.init_game_setting()
            done = False
            while (not done):
                action = self.make_action(state)
                state, reward, done, _ = self.env.step(action)

                self.saved_actions.append(action)
                self.rewards.append(reward)

            # for logging
            last_reward = np.sum(self.rewards)
            avg_reward = last_reward if not avg_reward else avg_reward * 0.9 + last_reward * 0.1
            # avg_reward_list.append(avg_reward)
            # update model
            self.update()

            if epoch % self.display_freq == 0:
                print('Epochs: %d/%d | Avg reward: %f ' %
                      (epoch, self.num_episodes, avg_reward))
                epoch_list.append(epoch)
                avg_reward_list.append(avg_reward)
            if avg_reward > 50:  # to pass baseline, avg. reward > 50 is enough.
                self.save('pg_improve.cpt')
                break


        x = epoch_list
        y = avg_reward_list
        plt.figure()
        plt.plot(x, y, 'o-')
        plt.xlabel('Epochs')
        plt.ylabel('Average reward')
        plt.show()
        plt.savefig("learning curve pg.jpg")
