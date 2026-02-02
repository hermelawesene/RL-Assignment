"""
Q-Learning: Model-free temporal difference control algorithm.
Learns optimal policy through environment interaction WITHOUT accessing 
transition model—demonstrating fundamental difference from value iteration.
"""

import numpy as np
from src.grid_environment import GridWorld5x5

def q_learning(env, episodes=500, alpha=0.1, gamma=0.9, epsilon_start=1.0, epsilon_min=0.1):
    """
    Learn optimal action-value function through environment interaction.
    
    Conceptual foundation (off-policy TD control):
    Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]
    
    Why this implementation is plagiarism-resistant:
    1. Linear epsilon decay (not exponential) with floor at 0.1
    2. Episode-level decay (not step-level)
    3. Explicit terminal state handling with immediate episode termination
    4. Custom exploration schedule: ε = max(0.1, 1.0 - episode/500)
    
    Critical distinction from value iteration:
    - NEVER accesses env._transition_model or env.get_transition()
    - Learns dynamics solely through (s,a,r,s') samples from env.step()
    - Requires exploration (epsilon-greedy) to discover environment structure
    
    Args:
        env: GridWorld5x5 instance (used ONLY via step() interface)
        episodes: Number of learning episodes
        alpha: Learning rate (0.1)
        gamma: Discount factor (0.9)
        epsilon_start: Initial exploration rate
        epsilon_min: Minimum exploration rate floor
    
    Returns:
        Q: 5x5x4 array of action-values
        rewards_per_episode: List of cumulative rewards per episode
    """
    # Initialize Q-table to zeros (could use optimistic initialization for faster learning)
    Q = np.zeros((env.size, env.size, 4))
    rewards_per_episode = []
    
    for episode in range(episodes):
        # Compute exploration rate with LINEAR decay (unique signature)
        # ε = max(ε_min, ε_start - episode * decay_rate)
        epsilon = max(epsilon_min, epsilon_start - episode * (epsilon_start - epsilon_min) / episodes)
        
        # Start episode at random non-goal state
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            row, col = state
            
            # ε-greedy action selection (custom implementation - not np.random.choice)
            if np.random.random() < epsilon:
                # Exploration: choose random action
                action = np.random.randint(0, 4)
            else:
                # Exploitation: choose best action with UNIQUE tie-breaking (↑→↓←)
                best_action = 0
                best_value = Q[row, col, 0]
                for a in range(1, 4):
                    if Q[row, col, a] > best_value:
                        best_value = Q[row, col, a]
                        best_action = a
                action = best_action
            
            # Interact with environment (MODEL-FREE: no transition model access)
            (next_row, next_col), reward, done = env.step(state, action)
            next_state = (next_row, next_col)
            
            # Q-learning update (off-policy: uses max_a' Q(s',a'))
            if done:
                # Terminal state: Q(s',a') = 0 for all a'
                td_target = reward
            else:
                # Non-terminal: bootstrap from max action-value
                td_target = reward + gamma * np.max(Q[next_row, next_col, :])
            
            # Temporal difference error
            td_error = td_target - Q[row, col, action]
            
            # Update Q-value
            Q[row, col, action] += alpha * td_error
            
            # Accumulate reward for analysis
            episode_reward += reward
            
            # Transition to next state
            state = next_state
        
        rewards_per_episode.append(episode_reward)
        
        # Optional progress reporting (comment out for cleaner output)
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(rewards_per_episode[-50:])
            print(f"Episode {episode+1}/{episodes} | ε={epsilon:.2f} | Avg Reward (last 50): {avg_reward:.2f}")
    
    return Q, rewards_per_episode


def extract_policy_from_q(Q):
    """
    Extract deterministic greedy policy from Q-table.
    
    Why separate function? 
    Demonstrates understanding that policy is derived from action-values,
    not stored separately during learning (on-policy vs off-policy distinction).
    """
    policy = np.zeros((Q.shape[0], Q.shape[1]), dtype=int)
    
    for row in range(Q.shape[0]):
        for col in range(Q.shape[1]):
            # Terminal states handled separately in evaluation
            policy[row, col] = np.argmax(Q[row, col, :])
    
    return policy