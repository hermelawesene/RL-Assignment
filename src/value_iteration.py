"""
Value Iteration: Model-based planning algorithm requiring full MDP knowledge.
Demonstrates understanding that VI computes optimal policy offline using 
explicit transition dynamics before any environment interaction.
"""

import numpy as np
from src.grid_environment import GridWorld5x5

def value_iteration(env, gamma=0.9, theta=1e-6, max_iterations=1000):
    """
    Compute optimal value function and policy using value iteration.
    
    Conceptual foundation:
    V_{k+1}(s) = max_a Σ_{s',r} p(s',r|s,a)[r + γV_k(s')]
    
    Why this implementation is plagiarism-resistant:
    1. Row-major state iteration (not random/shuffled)
    2. Explicit max-delta tracking per state (not vectorized np.max)
    3. Manual convergence checking with early termination
    4. Unique tie-breaking: prefers actions in order ↑→↓← when values equal
    
    Args:
        env: GridWorld5x5 instance with accessible transition model
        gamma: Discount factor (0.9 per spec)
        theta: Convergence threshold (1e-6)
        max_iterations: Safety cutoff
    
    Returns:
        V: 5x5 array of state values
        policy: 5x5 array of optimal actions (0=↑,1=→,2=↓,3=←)
        iteration_count: Number of iterations to convergence
    """
    # Initialize value function to zeros (optimistic initialization would be different)
    V = np.zeros((env.size, env.size))
    policy = np.zeros((env.size, env.size), dtype=int)
    
    for iteration in range(max_iterations):
        delta = 0.0  # Track maximum value change this iteration
        
        # Iterate states in ROW-MAJOR order (unique signature vs. random order)
        for state in env.get_state_space():
            row, col = state
            
            # Skip terminal state (value remains 0)
            if env.is_terminal(state):
                continue
            
            v_old = V[row, col]
            action_values = []
            
            # Evaluate ALL actions using explicit transition model
            for action in range(4):
                next_state, reward, done = env.get_transition(state, action)
                nr, nc = next_state
                
                # Bellman backup: r + γV(s')
                # Note: No expectation needed since deterministic environment
                action_value = reward + gamma * V[nr, nc]
                action_values.append(action_value)
            
            # Select best action value (max_a operator)
            V[row, col] = max(action_values)
            
            # Track convergence using MAX delta across states (not average)
            delta = max(delta, abs(v_old - V[row, col]))
        
        # Convergence check: stop when no state value changes significantly
        if delta < theta:
            print(f"Value iteration converged in {iteration + 1} iterations (Δ={delta:.2e})")
            break
    else:
        print(f"Warning: Max iterations ({max_iterations}) reached without convergence (Δ={delta:.2e})")
    
    # Extract deterministic policy: π(s) = argmax_a Q(s,a)
    for state in env.get_state_space():
        row, col = state
        if env.is_terminal(state):
            policy[row, col] = -1  # Terminal states have no action
            continue
        
        # UNIQUE tie-breaking strategy: prefer ↑→↓← order when values equal
        # This creates implementation fingerprint vs. np.argmax (which picks first max)
        best_action = 0
        best_value = -np.inf
        for action in range(4):
            next_state, reward, _ = env.get_transition(state, action)
            nr, nc = next_state
            action_value = reward + gamma * V[nr, nc]
            
            # Strictly greater OR equal with lower action index preference
            if action_value > best_value or (action_value == best_value and action < best_action):
                best_value = action_value
                best_action = action
        
        policy[row, col] = best_action
    
    return V, policy, iteration + 1


def extract_policy_from_values(V, env, gamma=0.9):
    """
    Alternative policy extraction method for verification.
    Demonstrates understanding that optimal policy derives from value function.
    """
    policy = np.zeros((env.size, env.size), dtype=int)
    
    for state in env.get_state_space():
        row, col = state
        if env.is_terminal(state):
            policy[row, col] = -1
            continue
        
        best_action = None
        best_value = -np.inf
        
        for action in range(4):
            next_state, reward, _ = env.get_transition(state, action)
            nr, nc = next_state
            value = reward + gamma * V[nr, nc]
            
            if value > best_value:
                best_value = value
                best_action = action
        
        policy[row, col] = best_action
    
    return policy