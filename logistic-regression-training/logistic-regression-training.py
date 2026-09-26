import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    # Write code here

    def initialize_weights():
        W = np.zeros(shape=(1, X.shape[1]))
        b = 0

        return W,b

    def forward(W,b, X):
        
        z = W @ X.T + b
        out = _sigmoid(z)
        return out

    def backward(delta, W, b, X):
        #delta = dL/dz
        #y_i / out - (1 - y_i) / (1 - out)
        #(1 - out) * y_i - (1 - y_i) * out
        #y_1 - out 
        #dz/dW
        grad_w = X
        #dz/dB
        grad_b = 1 
        W = W - lr * delta @ grad_w / X.shape[0]
        b = b - lr * np.mean(delta * grad_b, axis = 1, keepdims=True)

        return W,b

    W,b = initialize_weights()
    y = np.expand_dims(y, axis=0)
    
    for step in range(steps):
        p = forward(W,b,X) # 1 x N

        delta = p - y
        
        W,b = backward(delta, W,b, X)
        
    return np.squeeze(W,axis=0),b
