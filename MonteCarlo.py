import gymnasium as gym
import random
import numpy as np
import matplotlib.pyplot as plt
import pygame
import pickle

def run(episodes):
    env = gym.make("FrozenLake-v1", map_name = "8x8", is_slippery=True, render_mode="rgb_array")

    returns_sum = np.zeros((env.observation_space.n, env.action_space.n))
    returns_count = np.zeros((env.observation_space.n, env.action_space.n))
    Q = np.zeros((env.observation_space.n, env.action_space.n))

    epsilon = 1.0
    decrease_rate = 0.000044
    rng = np.random.default_rng()
    gamma = 0.9

    rewardsArray = np.zeros(episodes)
    episode_rewards = np.zeros(episodes)


    for i in range(episodes):
        episode = []
        state = env.reset()[0]
        done = False
        while(not done):
            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])

            new_state, reward, done, _, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = new_state

        rewardsArray[i] = sum([x[2] for x in episode])

        episode_rewards[i] = np.sum(rewardsArray[:i + 1]) / (i + 1)

        if epsilon > 0:
            epsilon -= decrease_rate

        t = 0
        while t < len(episode):
            state, action, reward = episode[t]
            sa_pair = (state, action)
            
            first_occurrence_idx = -1
            for i in range(t, len(episode)):
                if episode[i][0] == state and episode[i][1] == action:
                    first_occurrence_idx = i
                    break
            
            if first_occurrence_idx == -1:
                t += 1
                continue
            
            G = 0
            for i in range(first_occurrence_idx, len(episode)):
                G += episode[i][2] * (gamma ** (i - first_occurrence_idx))
            
            returns_sum[sa_pair] += G
            returns_count[sa_pair] += 1
            
            Q[state][action] = returns_sum[sa_pair] / returns_count[sa_pair]
            
            t = first_occurrence_idx + 1

    # print(Q)

    # mean_reward = np.mean(episode_rewards)
    # std_dev = np.std(episode_rewards)
    # print("Mean reward per run:", mean_reward)
    # print("Standard deviation of rewards:", std_dev)

    num_test_episodes = 1000
    total_rewards = 0

    for episode in range(num_test_episodes):

        state = env.reset()[0]
        terminate = False
        trunc = False
        episode_reward = 0
        
        while not (terminate or trunc):
            action = np.argmax(Q[state])
            next_state, reward, terminate, trunc, _ = env.step(action)
            episode_reward += reward
            state = next_state

        total_rewards += episode_reward

    avg_reward = total_rewards / num_test_episodes
    # print("Average reward over", num_test_episodes, "test episodes:", avg_reward)

    # sum_rewards1 = np.zeros(episodes)
    # for i in range(episodes):
    #     sum_rewards1[i] = np.sum(rewardsArray[max(0, i - 100):(i + 1)]) / min(i + 1, 100)
    # plt.plot(sum_rewards1)
    # plt.xlabel('Episodes grouped by 100')
    # plt.ylabel('Reward sum')
    # plt.savefig('frozenLakeMC.png')
    # plt.close()
    return avg_reward
    return Q

if __name__ == '__main__':
    rewards = []
    num_runs = 1000
    for i in range(num_runs):
        rewards.append(run(25000))
        print(i)
    
    mean_reward = np.mean(rewards)
    std_deviation = np.std(rewards)
    print("Mean reward over", num_runs, "runs:", mean_reward)
    print("Standard deviation of rewards:", std_deviation)
