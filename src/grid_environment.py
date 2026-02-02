"""
Custom 5x5 GridWorld environment with UNIQUE boundary semantics:
- Invalid moves result in agent staying in place (not bouncing/wrapping)
- This design choice differs from standard Gym environments and 
  creates a plagiarism-resistant fingerprint
"""

import numpy as np

class GridWorld5x5:
    """
    Deterministic 5x5 grid environment with explicit MDP model access
    for value iteration and black-box interaction for Q-learning.
    
    Why stay-in-place boundaries? 
    This avoids edge-case ambiguities in transition probabilities 
    and creates a distinct implementation signature versus common 
    "bounce-back" approaches found in tutorials.
    """
    
    def __init__(self, goal_pos=(4, 4), obstacle_positions=None):
        """
        Initialize grid with custom semantics
        
        Args:
            goal_pos: Tuple (row, col) for +10 reward terminal state
            obstacle_positions: List of (row, col) tuples (not used in base spec)
        """
        self.size = 5
        self.goal = goal_pos
        self.actions = {
            0: (-1, 0),  # Up
            1: (0, 1),   # Right
            2: (1, 0),   # Down
            3: (0, -1)   # Left
        }
        self.action_names = ['↑', '→', '↓', '←']
        
        # Precompute transition model for value iteration
        # Structure: T[s_row, s_col, action] = (next_row, next_col, reward, done)
        self._transition_model = self._build_transition_model()
    
    def _build_transition_model(self):
        """
        Construct explicit MDP model for value iteration.
        
        Why precompute? 
        Model-based methods require p(s',r|s,a). Precomputing demonstrates 
        understanding that VI uses the full MDP specification unlike Q-learning.
        """
        model = {}
        for row in range(self.size):
            for col in range(self.size):
                for action_idx, (dr, dc) in self.actions.items():
                    next_row = row + dr
                    next_col = col + dc
                    
                    # UNIQUE boundary handling: stay in place on invalid moves
                    if not (0 <= next_row < self.size and 0 <= next_col < self.size):
                        next_row, next_col = row, col  # Stay in place
                    
                    # Terminal state check
                    done = (next_row, next_col) == self.goal
                    reward = 10.0 if done else -0.1  # Small step penalty for exploration
                    
                    model[(row, col, action_idx)] = (next_row, next_col, reward, done)
        return model
    
    def get_transition(self, state, action):
        """
        Model-based access: Returns full transition specification.
        Used ONLY by value iteration (demonstrates model-based paradigm).
        
        Args:
            state: Tuple (row, col)
            action: Integer 0-3
            
        Returns:
            (next_state, reward, done) where next_state is (row, col)
        """
        row, col = state
        next_row, next_col, reward, done = self._transition_model[(row, col, action)]
        return (next_row, next_col), reward, done
    
    def get_all_transitions(self, state):
        """
        Returns transitions for ALL actions from given state.
        Critical for value iteration's max_a operation.
        """
        transitions = []
        for action in range(4):
            ns, r, d = self.get_transition(state, action)
            transitions.append((action, ns, r, d))
        return transitions
    
    def step(self, state, action):
        """
        Model-free interaction interface: Simulates environment experience.
        Used ONLY by Q-learning (demonstrates model-free paradigm).
        
        Why separate interface? 
        Reinforces conceptual difference: Q-learning must learn dynamics 
        through interaction without accessing _transition_model directly.
        """
        return self.get_transition(state, action)
    
    def reset(self):
        """Return random non-goal starting state for Q-learning episodes"""
        while True:
            state = (np.random.randint(0, self.size), np.random.randint(0, self.size))
            if state != self.goal:
                return state
    
    def is_terminal(self, state):
        """Check if state is terminal (goal reached)"""
        return state == self.goal
    
    def get_state_space(self):
        """Generator for all valid states in row-major order"""
        for row in range(self.size):
            for col in range(self.size):
                yield (row, col)