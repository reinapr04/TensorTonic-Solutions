import numpy as np

def triplet_loss(anchor: list, positive: list, negative: list, margin: float = 1.0) -> float:
    """
    Returns the loss as a float.
    """
    # Write code here
    a = np.array(anchor)
    p = np.array(positive)
    n = np.array(negative)
    
    def euclid(a,b):
        return np.sum((a - b)**2, axis = -1)
        
    loss = np.maximum(euclid(a,p) - euclid(a,n) + margin, 0)
    return np.mean(loss).item()