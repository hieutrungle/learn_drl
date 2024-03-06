import gymnasium as gym

# import pybullet_envs
import numpy as np
from sac import Agent
from utils import plot_learning_curve

if __name__ == "__main__":

    env = gym.make("InvertedPendulum-v4")
    agent = Agent(
        input_dims=env.observation_space.shape,
        alpha=0.0003,
        beta=0.0003,
        env=env,
        n_actions=env.action_space.shape[0],
    )
    n_games = 2000
    filename = "results/plots/InvertedPendulumBulletEnv.png"

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()
        env.render(mode="human")
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
