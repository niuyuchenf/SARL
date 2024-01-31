import os
from sumohighway import HighwayEnv
import numpy as np
import argparse
from utils import create_directory, plot_learning_curve
from D3QN import D3QN


parser = argparse.ArgumentParser()
parser.add_argument('--max_episodes', type=int, default=3000)
parser.add_argument('--ckpt_dir', type=str, default='./checkpoints/D3QN/')
parser.add_argument('--reward_path', type=str, default='./output_images/reward.png')
parser.add_argument('--epsilon_path', type=str, default='./output_images/epsilon.png')

args = parser.parse_args(args=[])

def main():
    env = HighwayEnv()
    agent = D3QN(alpha=0.0003, state_dim=28, action_dim=12,
                 fc1_dim=256, fc2_dim=256, ckpt_dir=args.ckpt_dir, gamma=0.98, tau=0.005, epsilon=1.0,
                 eps_end=0.45, eps_dec=5e-4, max_size=1000000, batch_size=1024)
    create_directory(args.ckpt_dir, sub_dirs=['Q_eval', 'Q_target'])
    total_rewards, avg_rewards, epsilon_history = [], [], []

    for episode in range(args.max_episodes):
        total_reward = 0
        done = False
        observation,state = env.reset()
        step = 0
        while not done:
            action, lane_left, lane_right = agent.choose_action(observation, step, state, isTrain=True)
            step = 1
            observation_, state, reward, done, collision = env.step(action)
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
            total_reward += reward
            observation = observation_

        total_rewards.append(total_reward)
        avg_reward = np.mean(total_rewards[-100:])
        avg_rewards.append(avg_reward)
        epsilon_history.append(agent.epsilon)
        
        if (episode + 1) % 100 == 0:
            agent.save_models(episode+1)
            print('EP:{} Reward:{} Avg_reward:{} Epsilon:{}'.
              format(episode+1, total_reward, avg_reward, agent.epsilon))
            
    np.save(r'path.npy',np.array(avg_rewards))
    episodes = [i+1 for i in range(args.max_episodes)]
    plot_learning_curve(episodes, avg_rewards, title='Reward', ylabel='reward',
                        figure_file=args.reward_path)
    plot_learning_curve(episodes, epsilon_history, title='Epsilon', ylabel='epsilon',
                        figure_file=args.epsilon_path)


if __name__ == '__main__':
    main()