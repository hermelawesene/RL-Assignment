"""
Experiment script for Value Iteration.
Demonstrates model-based planning with explicit MDP access.
"""

import numpy as np
import sys
sys.path.append('..')

from src.grid_environment import GridWorld5x5
from src.value_iteration import value_iteration
from utils.visualizer import plot_value_function, plot_policy

def main():
    print("="*60)
    print("VALUE ITERATION: Model-Based Planning")
    print("="*60)
    print("Concept: Uses explicit MDP model (p(s',r|s,a)) to compute")
    print("optimal policy offline without environment interaction.\n")
    
    # Initialize environment with custom semantics
    env = GridWorld5x5(goal_pos=(4, 4))
    
    # Run value iteration
    V, policy, iterations = value_iteration(
        env, 
        gamma=0.9, 
        theta=1e-6
    )
    
    print(f"\nOptimal value at start state (0,0): V(0,0) = {V[0,0]:.4f}")
    print(f"Optimal value at goal-adjacent state (3,4): V(3,4) = {V[3,4]:.4f}")
    
    # Visualize results
    plot_value_function(
        V, 
        title="Value Iteration: State Values V(s)",
        save_path="results/vi_values.png"
    )
    
    plot_policy(
        policy, 
        V=V,
        title="Value Iteration: Optimal Policy π*(s)",
        save_path="results/vi_policy.png"
    )
    
    # Print policy in human-readable format
    print("\nOptimal Policy (Grid Representation):")
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
    
    print("\n✓ Value iteration completed successfully")
    print("✓ Policy extracted and visualized")
    print("\nKey Insight: Policy computed WITHOUT environment interaction,")
    print("demonstrating model-based paradigm.")

if __name__ == "__main__":
    # Create results directory if needed
    import os
    os.makedirs("results", exist_ok=True)
    main()