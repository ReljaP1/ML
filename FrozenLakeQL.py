import gymnasium as gym
import random
import numpy as np
import matplotlib.pyplot as plot
import pygame
import pickle

def run(episodes):

    env = gym.make("FrozenLake-v1", map_name = "8x8", is_slippery=True, render_mode="rgb_array")

    q = np.zeros((env.observation_space.n, env.action_space.n))

    alpha = 0.1
    gamma = 0.9

    epsilon = 1
    decrease_rate = 0.000044
    rng = np.random.default_rng()

    rewardsArray = np.zeros(episodes)
    episode_rewards = np.zeros(episodes)


    for i in range(episodes):
        state = env.reset()[0]
        terminate = False
        trunc = False

        while(not terminate and not trunc):

            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state,:])
            
            new_state, reward, terminate, trunc, _ = env.step(action)

            q[state, action] = q[state, action] + alpha * (reward + gamma * np.max(q[new_state,:]) - q[state, action])

            state = new_state

            if terminate:
                break

        if epsilon > 0:
            epsilon -= decrease_rate

        if reward != 0:
            rewardsArray[i] = 1

        episode_rewards[i] = np.sum(rewardsArray[:i + 1]) / (i + 1)

    # print(q)

    mean_reward = np.mean(episode_rewards)
    std_dev = np.std(episode_rewards)
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
            action = np.argmax(q[state,:])
            next_state, reward, terminate, trunc, _ = env.step(action)
            episode_reward += reward
            state = next_state

        total_rewards += episode_reward

    avg_reward = total_rewards / num_test_episodes
    # print("Average reward over", num_test_episodes, "test episodes:", avg_reward)

    env.close()
    sum_rewards1 = np.zeros(episodes)
    for i in range(episodes):
        sum_rewards1[i] = np.sum(rewardsArray[max(0, i - 100):(i + 1)])
    plot.plot(sum_rewards1)
    plot.xlabel('Episodes grouped by 100')
    plot.ylabel('Reward sum')
    plot.savefig('frozenLakeQL.png')
    plot.close()
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
