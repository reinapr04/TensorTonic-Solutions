import numpy as np

def epsilon_greedy(q_values: list, epsilon: float, seed: int = 0) -> int:
    """
    Returns the action index as an integer.
    """
    # Write code here
    rng = np.random.default_rng(seed=seed)
    mu = rng.uniform(low=0,high=1)

    if mu >= epsilon:
        q_values = np.array(q_values)
        return int(np.argmax(q_values))
    else:
        idx = int(rng.integers(0,len(q_values)))
        return idx
    