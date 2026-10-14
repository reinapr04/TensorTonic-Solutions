import numpy as np

def gru_cell_forward(x: list, h_prev: list, params: dict) -> np.ndarray:
    """
    Returns the updated hidden state as a NumPy array matching the shape of h_prev.
    """
    # Write code here

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
        
    x = np.array(x)
    h_prev = np.array(h_prev)
    
    r_t = sigmoid(x @ params["Wr"] + h_prev @ params["Ur"] + params["br"])
    z_t = sigmoid(x @ params["Wz"] + h_prev @ params["Uz"] + params["bz"])

    h_t_c = np.tanh(x  @ params["Wh"] + (r_t * h_prev) @ params["Uh"] + params["bh"])

    h_t = (1 - z_t) * h_prev + (z_t * h_t_c)

    return h_t