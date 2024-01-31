import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer

device = T.device("cuda:0" if T.cuda.is_available() else "cpu")


class DuelingDeepQNetwork(nn.Module):
    def __init__(self, alpha, state_dim, action_dim, fc1_dim, fc2_dim):
        super(DuelingDeepQNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.V = nn.Linear(fc2_dim, 1)
        self.A = nn.Linear(fc2_dim, action_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.to(device)

    def forward(self, state):
        x = T.relu(self.fc1(state))
        x = T.relu(self.fc2(x))

        V = self.V(x)
        A = self.A(x)
        Q = V + A - T.mean(A, dim=-1, keepdim=True)

        return Q

    def save_checkpoint(self, checkpoint_file):
        T.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(T.load(checkpoint_file))


class D3QN:
    def __init__(self, alpha, state_dim, action_dim, fc1_dim, fc2_dim, ckpt_dir,
                 gamma=0.99, tau=0.005, epsilon=1.0, eps_end=0.01, eps_dec=5e-7,
                 max_size=1000000, batch_size=256):
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.batch_size = batch_size
        self.checkpoint_dir = ckpt_dir
        self.action_space = [i for i in range(action_dim)]

        self.q_eval = DuelingDeepQNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
                                          fc1_dim=fc1_dim, fc2_dim=fc2_dim)
        self.q_target = DuelingDeepQNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
                                            fc1_dim=fc1_dim, fc2_dim=fc2_dim)

        self.memory = ReplayBuffer(state_dim=state_dim, action_dim=action_dim,
                                   max_size=max_size, batch_size=batch_size)

        self.update_network_parameters(tau=1.0)
        self.lane_left = 0
        self.lane_right = 0
        self.time = 0.5
        self.maxacc = 1.5
        self.maxbrake = 4.0
        self.minbrake = 3.5

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for q_target_params, q_eval_params in zip(self.q_target.parameters(), self.q_eval.parameters()):
            q_target_params.data.copy_(tau * q_eval_params + (1 - tau) * q_target_params)

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min

    def choose_action(self, observation, step, state_feature, isTrain=True):
        ego_vehicle_position_x, ego_vehicle_lane, ego_vehicle_longitudinalv, ego_vehicle_lateralv,\
        ego_front_position_x, ego_front_lane, ego_front_longitudinalv, ego_front_lateralv,\
        ego_rear_position_x, ego_rear_lane, ego_rear_longitudinalv, ego_rear_lateralv,\
        left_front_position_x, left_front_lane, left_front_longitudinalv, left_front_lateralv,\
        left_rear_position_x, left_rear_lane, left_rear_longitudinalv, left_rear_lateralv,\
        right_front_position_x, right_front_lane, right_front_longitudinalv, right_front_lateralv,\
        right_rear_position_x, right_rear_lane, right_rear_longitudinalv, right_rear_lateralv = state_feature
        if step == 0:
            self.lane_left = 0
            self.lane_right = 0
        state = T.tensor([observation], dtype=T.float).to(device)
        q_vals = self.q_eval.forward(state)
        action = T.argmax(q_vals).item()
        if (np.random.random() < self.epsilon) and isTrain:
            action = np.random.choice(self.action_space)
        if action in (0,3,6,9):
            self.lane_left += 1
            self.lane_right = 0
        elif action in (2,5,8,11):
            self.lane_right += 1
            self.lane_left = 0
        else:
            self.lane_left = 0
            self.lane_right = 0
        if self.lane_left < 6 and self.lane_right < 6:
            if action in (0,3,6,9):
                if abs(ego_front_position_x-ego_vehicle_position_x)-5<self.RSSmodel_distance(ego_vehicle_longitudinalv,ego_front_longitudinalv):
                    action = 3
            elif action in (2,5,8,11):
                if abs(ego_front_position_x-ego_vehicle_position_x)-5<self.RSSmodel_distance(ego_vehicle_longitudinalv,ego_front_longitudinalv):
                    action = 5         
        elif self.lane_left == 6:
            if abs(left_front_position_x-ego_vehicle_position_x)-5<self.RSSmodel_distance(ego_vehicle_longitudinalv,left_front_longitudinalv) or abs(left_rear_position_x-ego_vehicle_position_x)-5<self.RSSmodel_distance(left_rear_longitudinalv,ego_vehicle_longitudinalv):
                state = T.tensor([observation], dtype=T.float).to(device)
                q_vals = self.q_eval.forward(state)
                q_vals[0][0] = -100
                q_vals[0][3] = -100
                q_vals[0][6] = -100
                q_vals[0][9] = -100
                q_vals[0][2] = -100
                q_vals[0][5] = -100
                q_vals[0][8] = -100
                q_vals[0][11] = -100
                action = T.argmax(q_vals).item()
                if (np.random.random() < self.epsilon) and isTrain:
                    action = np.random.choice([1,4,7,10])
            self.lane_left = 0
        elif self.lane_right == 6:
            if abs(right_front_position_x-ego_vehicle_position_x)-5<self.RSSmodel_distance(ego_vehicle_longitudinalv,right_front_longitudinalv) or abs(right_rear_position_x-ego_vehicle_position_x)-5<self.RSSmodel_distance(right_rear_longitudinalv,ego_vehicle_longitudinalv):
                state = T.tensor([observation], dtype=T.float).to(device)
                q_vals = self.q_eval.forward(state)
                q_vals[0][2] = -100
                q_vals[0][5] = -100
                q_vals[0][8] = -100
                q_vals[0][11] = -100
                q_vals[0][0] = -100
                q_vals[0][3] = -100
                q_vals[0][6] = -100
                q_vals[0][9] = -100
                action = T.argmax(q_vals).item()
                if (np.random.random() < self.epsilon) and isTrain:
                    action = np.random.choice([1,4,7,10])
            self.lane_right = 0
        if action in (1,4,7,10):
            if abs(ego_front_position_x-ego_vehicle_position_x)-5<self.RSSmodel_distance(ego_vehicle_longitudinalv,ego_front_longitudinalv):
                action = 4
        return action, self.lane_left, self.lane_right

    def learn(self):
        if not self.memory.ready():
            return

        states, actions, rewards, next_states, terminals = self.memory.sample_buffer()
        batch_idx = T.arange(self.batch_size, dtype=T.long).to(device)
        states_tensor = T.tensor(states, dtype=T.float).to(device)
        actions_tensor = T.tensor(actions, dtype=T.long).to(device)
        rewards_tensor = T.tensor(rewards, dtype=T.float).to(device)
        next_states_tensor = T.tensor(next_states, dtype=T.float).to(device)
        terminals_tensor = T.tensor(terminals).to(device)

        with T.no_grad():
            q_ = self.q_target.forward(next_states_tensor)
            max_actions = T.argmax(self.q_eval.forward(next_states_tensor), dim=-1)
            q_[terminals_tensor] = 0.0
            target = rewards_tensor + self.gamma * q_[batch_idx, max_actions]
        q = self.q_eval.forward(states_tensor)[batch_idx, actions_tensor]

        loss = F.mse_loss(q, target.detach())
        self.q_eval.optimizer.zero_grad()
        loss.backward()
        self.q_eval.optimizer.step()

        self.update_network_parameters()
        self.decrement_epsilon()

    def save_models(self, episode):
        self.q_eval.save_checkpoint(self.checkpoint_dir + 'Q_eval/D3QN_q_eval_{}.pth'.format(episode))
        print('Saving Q_eval network successfully!')
        self.q_target.save_checkpoint(self.checkpoint_dir + 'Q_target/D3QN_Q_target_{}.pth'.format(episode))
        print('Saving Q_target network successfully!')

    def load_models(self, episode):
        self.q_eval.load_checkpoint(self.checkpoint_dir + 'Q_eval/D3QN_q_eval_{}.pth'.format(episode))
        print('Loading Q_eval network successfully!')
        self.q_target.load_checkpoint(self.checkpoint_dir + 'Q_target/D3QN_Q_target_{}.pth'.format(episode))
        print('Loading Q_target network successfully!')
    def RSSmodel_distance(self,vrear,vfront):
        distance = max(0,(vrear*self.time+0.5*self.maxacc*self.time**2+(vrear+self.time*self.maxacc)**2/(2*self.minbrake)-vfront**2/(2*self.maxbrake)))
        return distance