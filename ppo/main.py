import gymnasium as gym
import numpy as np
from ppo import Agent
from utils import plot_learning_curve

if __name__ == "__main__":

    env = gym.make("CartPole-v1")
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003

    agent = Agent(
        input_dims=env.observation_space.shape,
        batch_size=batch_size,
        alpha=alpha,
        n_epochs=n_epochs,
        N=N,
        n_actions=env.action_space.n,
    )
    n_games = 300
    filename = "results/plots/cartpole.png"

    best_score = env.reward_range[0]
    score_history = []
    learn_iters = 0
    n_steps = 0
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
            action, probs, value = agent.choose_action(observation)
            next_observation, reward, done, truncated, info = env.step(action)
            agent.remember(observation, action, probs, value, reward, done)
            score += reward
            n_steps += 1
            if n_steps % N == 0:
                learn_iters += 1
                if not load_checkpoint:
                    agent.learn()
            observation = next_observation
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()
        print(
            f"episode {i}, score: {score}, avg score {avg_score}, 'time_steps': {n_steps}, 'learn_iters': {learn_iters}"
        )

    if not load_checkpoint:
        x = [i + 1 for i in range(n_games)]
        plot_learning_curve(x, score_history, filename)

    env.close()
