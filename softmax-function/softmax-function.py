import numpy as np

def softmax(x: list) -> np.ndarray:
    """
    Returns stable softmax probabilities as a NumPy array matching the shape of x.
    """
    # Write code here
    x = np.array(x)
    if len(x.shape) > 1:
        axis = 1
    else:
        axis = 0
    
    x = x - np.max(x, axis = axis, keepdims=True)
    x = np.exp(x)

    softmax = x / np.sum(x, axis = axis, keepdims=True)
    return softmax