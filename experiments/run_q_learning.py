"""
Experiment script for Q-Learning.
Demonstrates model-free learning through environment interaction.
"""

import numpy as np
import sys
sys.path.append('..')

from src.grid_environment import GridWorld5x5
from src.q_learning import q_learning, extract_policy_from_q
from utils.visualizer import plot_policy, plot_learning_curve

def main():
    print("="*60)
    print("Q-LEARNING: Model-Free Temporal Difference Control")
    print("="*60)
    print("Concept: Learns optimal policy through environment interaction")
    print("WITHOUT accessing transition model (p(s',r|s,a) unknown).\n")
    
    # Initialize environment
    env = GridWorld5x5(goal_pos=(4, 4))
    
    # Run Q-learning
    Q, rewards = q_learning(
        env,
        episodes=500,
        alpha=0.1,
        gamma=0.9,
        epsilon_start=1.0,
        epsilon_min=0.1
    )
    
    # Extract greedy policy from Q-table
    policy = extract_policy_from_q(Q)
    
    # Mark goal state as terminal in policy visualization
    policy[4, 4] = -1
    
    print(f"\nLearned Q-value at start state (0,0):")
    print(f"  Up: {Q[0,0,0]:.2f} | Right: {Q[0,0,1]:.2f} | Down: {Q[0,0,2]:.2f} | Left: {Q[0,0,3]:.2f}")
    
    # Visualize learning progress
    plot_learning_curve(
        rewards,
        window=25,
        title="Q-Learning: Convergence of Cumulative Reward",
        save_path="results/ql_learning_curve.png"
    )
    
    # Visualize final policy
    plot_policy(
        policy,
        title="Q-Learning: Learned Optimal Policy",
        save_path="results/ql_policy.png"
    )
    
    # Print policy grid
    print("\nLearned Policy (Grid Representation):")
    print("  Col: 0    1    2    3    4")
    for row in range(5):
        row_actions = []
        for col in range(5):
            if (row, col) == (4, 4):
                row_actions.append(" G ")
            else:
                action = policy[row, col]
                row_actions.append(f" {env.action_names[action]} ")
        print(f"Row {row}: {' | '.join(row_actions)}")
    
    print("\n✓ Q-learning completed successfully")
    print("✓ Policy extracted from Q-table and visualized")
    print("\nKey Insight: Policy learned THROUGH interaction without MDP model,")
    print("demonstrating model-free paradigm.")

if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    main()