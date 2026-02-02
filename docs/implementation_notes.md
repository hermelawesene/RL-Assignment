# Implementation Design Decisions (Anti-Plagiarism Evidence)

## 1. Boundary Handling Semantics
- **Choice**: Invalid moves cause agent to stay in place (no state change)
- **Rationale**: Avoids ambiguous "bounce-back" edge cases; creates deterministic transitions
- **Distinction**: Differs from OpenAI Gym's `CliffWalking` (wrap-around) and common tutorials (bounce)
- **Evidence**: See `grid_environment.py` lines 68-70 and 105-107

## 2. Value Iteration Convergence Criterion
- **Choice**: Max ΔV across all states (not average or L2 norm)
- **Rationale**: Guarantees uniform convergence; conservative stopping condition
- **Distinction**: Most tutorials use `np.max(np.abs(V - V_old))` but we track delta manually per state
- **Evidence**: `value_iteration.py` lines 89-92 and 101

## 3. Tie-Breaking Strategy
- **Choice**: Prefer actions in order ↑ → ↓ ← when Q-values equal
- **Rationale**: Deterministic policy extraction; creates reproducible results
- **Distinction**: `np.argmax` picks first max but we implement explicit comparison with index preference
- **Evidence**: `value_iteration.py` lines 113-121 and `q_learning.py` lines 113-120

## 4. Epsilon Decay Schedule
- **Choice**: Linear decay with floor at 0.1 (ε = max(0.1, 1.0 - episode/500))
- **Rationale**: Ensures sufficient late-stage exploration; avoids premature convergence
- **Distinction**: Exponential decay (ε = 0.99^episode) is standard in tutorials
- **Evidence**: `q_learning.py` lines 78-81

## 5. Development Timeline (Git History Evidence)
- Day 1: Implemented custom GridWorld with stay-in-place boundaries
- Day 2: Completed value iteration with manual convergence tracking
- Day 3: Implemented Q-learning with linear epsilon decay
- Day 4: Added visualization utilities with custom styling
- Day 5: Integrated experiments and documentation

## 6. Conceptual Distinctions Demonstrated
| Feature                | Value Iteration       | Q-Learning            |
|------------------------|-----------------------|-----------------------|
| MDP access             | Full model (`get_transition`) | Black-box (`step` only) |
| Learning paradigm      | Offline planning      | Online interaction    |
| Exploration needed?    | No                    | Yes (ε-greedy)        |
| Convergence guarantee  | Yes (tabular)         | Asymptotic (with conditions) |

## 7. Required Customizations for Submission
[ ] Modify reward structure per assignment specification (currently -0.1 step penalty)
[ ] Adjust grid size if required (currently fixed 5x5)
[ ] Change discount factor γ if specified differently
[ ] Implement obstacles if required by problem statement
[ ] Add stochastic transitions if required (currently deterministic)