import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def train_rl_agent(agent, environment, episodes=100, max_steps=100,
                   render=False, verbose=1, eval_interval=10):
    """
    Train a reinforcement learning agent on an environment

    Args:
        agent: RL agent with get_action and train methods
        environment: Environment with reset and step methods
        episodes: Number of episodes to train
        max_steps: Maximum steps per episode
        render: Whether to render the environment
        verbose: Verbosity level
        eval_interval: Episodes between evaluations

    Returns:
        Dictionary with training statistics
    """
    rewards_history = []
    avg_rewards_history = []

    for episode in range(episodes):
        state = environment.reset()
        episode_reward = 0

        for step in range(max_steps):
            # get action from agent
            action = agent.get_action(state)

            # take action in environment
            next_state, reward, done, _ = environment.step(action)

            # train agent
            agent.train(state, action, reward, next_state, done)

            # update state and accumulate reward
            state = next_state
            episode_reward += reward

            if done:
                break

        # store rewards
        rewards_history.append(episode_reward)
        avg_reward = np.mean(rewards_history[-100:])
        avg_rewards_history.append(avg_reward)

        # log progress
        if verbose > 0 and (episode % eval_interval == 0 or episode == episodes - 1):
            print(f"Episode {episode + 1}/{episodes} | Reward: {episode_reward:.4f} | "
                  f"Avg Reward: {avg_reward:.4f} | Epsilon: {agent.epsilon:.4f}")

    # TODO(add plot function to plot_utils): plot learning curve
    if verbose > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(rewards_history, label='Episode Reward')
        plt.plot(avg_rewards_history, label='Avg Reward (100 ep)')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('RL Agent Learning Curve')
        plt.legend()
        plt.grid(True)
        plt.show()

    return {
        'rewards': rewards_history,
        'avg_rewards': avg_rewards_history,
        'final_avg_reward': avg_rewards_history[-1]
    }


def evaluate_rl_agent(agent, environment, episodes=10, max_steps=100, verbose=1):
    """
    Evaluate a trained RL agent

    Args:
        agent: Trained RL agent
        environment: Environment to evaluate on
        episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        verbose: Verbosity level

    Returns:
        Average reward
    """
    rewards = []

    for episode in range(episodes):
        state = environment.reset()
        episode_reward = 0

        for step in range(max_steps):
            # get action without exploration
            action = agent.get_action(state, training=False)

            # take action in environment
            next_state, reward, done, _ = environment.step(action)

            state = next_state
            episode_reward += reward

            if done:
                break

        rewards.append(episode_reward)

        if verbose > 1:
            print(f"Evaluation Episode {episode + 1}/{episodes} | Reward: {episode_reward:.4f}")

    avg_reward = np.mean(rewards)

    if verbose > 0:
        print(f"Evaluation complete | Avg Reward: {avg_reward:.4f}")

    return avg_reward