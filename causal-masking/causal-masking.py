import numpy as np

def apply_causal_mask(scores: list, mask_value: float = -1e9) -> np.ndarray:
    """
    Returns a causally masked NumPy array matching the shape of scores.
    """
    # Write code here
    masked_scores = np.array(scores)

    T = masked_scores.shape[-1]
    # mask = np.zeros((T,T))
    # mask_with_values = np.zeros((T,T))
    
    # for i in range(T):
    #     for j in range(T):
    #         if i >= j:
    #             mask[i][j] = 1
    #             mask_with_values[i][j] = 0
    #         else:
    #             mask[i][j] = 0
    #             mask_with_values[i][j] = mask_value
    # return masked_scores * mask + mask_with_values

    mask = np.triu(np.ones((T,T), dtype=bool), k = 1)
    masked_scores[..., mask] = mask_value
    return masked_scores
                
