import gymnasium as gym
import numpy as np
from td3 import Agent
from utils import plot_learning_curve

if __name__ == "__main__":

    env = gym.make("BipedalWalker-v3")
    agent = Agent(
        input_dims=env.observation_space.shape,
        alpha=0.001,
        beta=0.001,
        env=env,
        n_actions=env.action_space.shape[0],
    )
    # print(f"observation space: {env.observation_space.shape}")
    # print(f"action space: {env.action_space.shape}")
    # exit()
    n_games = 2000
    filename = "results/plots/bipedal_walker.png"

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()
        evaluate = True
    else:
        evaluate = False

    for i in range(n_games):
        observation, info = env.reset()

        done = False
        truncated = False
        score = 0
        while not done and not truncated:
            action = agent.choose_action(observation, evaluate)
            next_observation, reward, done, truncated, info = env.step(action)
            agent.remember(observation, action, reward, next_observation, done)
            if not load_checkpoint:
                agent.learn()
            score += reward
            observation = next_observation
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()
        print(f"episode {i}, score: {score}, avg score {avg_score}")

    if not load_checkpoint:
        x = [i + 1 for i in range(n_games)]
        plot_learning_curve(x, score_history, filename)

    env.close()
