import random
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from collections import namedtuple

use_cuda = torch.cuda.is_available()
Transition = namedtuple('Transition',
                        ('state_image', 'state_other', 'action', 'next_state', 'reward'))

class DQN(nn.Module):
    '''
    This architecture is the one from OpenAI Baseline, with small modification.
    '''
    def __init__(self, channels, num_actions, other_state):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(2304, 512)
        self.fc2 = nn.Linear(512, 256)

        self.other_state1 = nn.Linear(other_state, 16)
        self.other_state2 = nn.Linear(16, 64)

        self.head = nn.Linear(256 + 64, num_actions)

        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU(0.01)

    def forward(self, x, s2, s3):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.lrelu(self.fc(x.view(x.size(0), -1)))
        x = self.relu(self.fc2(x))

        x2 = torch.cat((s2, s3))
        x2 = self.relu(self.other_state1(x2))
        x2 = self.relu(self.other_state2(x2))
        x = torch.cat((x, x2))

        q = self.head(x)
        return q


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class AgentDQN:
    def __init__(self, env, eval):
        self.env = env
        self.input_channels = 3
        self.other_state = 2
        self.num_actions = 8
        self.MEMORY_CAPACITY = 100000
        self.replayMem = ReplayMemory(self.MEMORY_CAPACITY)

        # build target, online network
        self.target_net = DQN(self.input_channels, self.num_actions, self.other_state)
        self.target_net = self.target_net.cuda() if use_cuda else self.target_net
        self.online_net = DQN(self.input_channels, self.num_actions, self.other_state)
        self.online_net = self.online_net.cuda() if use_cuda else self.online_net

        if eval:
            self.load('dqn')
        
        # discounted reward
        self.GAMMA = 0.99 
        
        # training hyperparameters
        self.train_freq = 4     # frequency to train the online network
        self.learning_start = 5000     # before we start to update our network, we wait a few steps first to fill the replay.
        self.batch_size = 64
        self.num_timesteps = 10000    # total training steps
        self.num_timesteps_for_each = 50000
        self.display_freq = 10  # frequency to display training progress
        self.save_freq = 20000     # frequency to save the model
        self.target_update_freq = 2000  # frequency to update target network

        self.EPS_START = 0.9    # random sample "random generate" at first
        self.EPS_END = 0.05     # random sample "random generate" at end
        self.EPS_DECAY = 200000

        # optimizer
        self.optimizer = optim.RMSprop(self.online_net.parameters(), lr=1e-4)

        self.steps = 0  # num. of passed steps. this may be useful in controlling exploration

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.steps_done = 0

    def save(self, save_path):
        print('save model to', save_path)
        torch.save(self.online_net.state_dict(), save_path + '_online.cpt')
        torch.save(self.target_net.state_dict(), save_path + '_target.cpt')

    def load(self, load_path):
        print('load model from', load_path)
        if use_cuda:
            self.online_net.load_state_dict(torch.load(load_path + '_online.cpt'))
            self.target_net.load_state_dict(torch.load(load_path + '_target.cpt'))
        else:
            self.online_net.load_state_dict(torch.load(load_path + '_online.cpt', map_location=lambda storage, loc: storage))
            self.target_net.load_state_dict(torch.load(load_path + '_target.cpt', map_location=lambda storage, loc: storage))

    def init_game_setting(self):
        # we don't need init_game_setting in DQN
        pass
    
    def make_action(self, state, test=False):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                print(state)
                action = self.online_net.forward(state[0].float(), state[1].float(), state[2].float()).max(1)[1].view(1, 1)
                return action.item()
        else:
            action = torch.tensor([[random.randrange(self.num_actions)]], device=self.device, dtype=torch.long)
            return action.item()

    def update(self):
        if len(self.replayMem) < self.batch_size:
            return
        transitions = self.replayMem.sample(self.batch_size)

        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                      device=self.device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        # print('batch.reward', batch.reward)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.online_net(state_batch.float()).gather(1, action_batch)

        with torch.no_grad():
            next_state_values = torch.zeros(self.batch_size, device=self.device)
            next_state_values[non_final_mask] = self.target_net(non_final_next_states.float()).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self):
        episodes_done_num = 0   # passed episodes
        total_reward = 0    # compute average reward
        loss = 0 
        while True:     # self.steps > self.num_timesteps --> break
            state = self.env.reset(random=True)
            print(state)
            self.env.render(intestine_update=True)
            # State: (80,80,4) --> (1,4,80,80)
            state = torch.from_numpy(state).permute(2, 0, 1).unsqueeze(0)
            state = state.cuda() if use_cuda else state
            
            done = False
            while not done:
                self.env.render(intestine_update=False)
                # select and perform action
                # print('a')
                action = self.make_action(state)        # [[action]]
                # print('b')
                next_state, reward, done = self.env.step(action)
                print('reward: ', reward)
                total_reward += reward

                # process new state
                # print('d')
                next_state = torch.from_numpy(next_state[0]).permute(2, 0, 1).unsqueeze(0)
                next_state = next_state.cuda() if use_cuda else next_state
                if done:
                    next_state = None

                # print('e')
                self.replayMem.push(state,
                                    torch.tensor([[action]]).to(self.device),
                                    next_state[0], next_state[1:],
                                    torch.tensor([reward]).to(self.device).float())

                # move to the next state
                state = next_state

                # Perform one step of the optimization
                if self.steps > self.learning_start and self.steps % self.train_freq == 0:
                    loss = self.update()

                # update target network
                if self.steps > self.learning_start and self.steps % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.online_net.state_dict())

                # save the model
                if self.steps % self.save_freq == 0:
                    self.save('dqn')

                self.steps += 1

                if self.steps > self.num_timesteps_for_each:
                    print('self.steps > self.num_timestepsm, Break!')
                    self.steps = 0
                    break

            if episodes_done_num % self.display_freq == 0:
                print('Episode: %d | Steps: %d/%d | Avg reward: %f | loss: %f '%
                        (episodes_done_num, self.steps, self.num_timesteps, total_reward / self.display_freq, loss))
                total_reward = 0

            episodes_done_num += 1
            if episodes_done_num > self.num_timesteps:
                print('episodes_done_num > self.num_timestepsm, Break!')
                break
        self.save('dqn')


if __name__ == '__main__':
    from env_NewPath_image_runV5 import Intestine_path
    env = Intestine_path()
    agent = AgentDQN(env, eval=False)
    agent.train()
