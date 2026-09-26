import numpy as np

def impute_missing(X: list, strategy: str = "mean") -> np.ndarray:
    """
    Returns a NumPy array with the same shape as X.
    """
    # Write code here
    X = np.array(X)
    axis = 0
    one_d = False
    if len(X.shape) < 2:
        X = X[:,None]
        one_d = True
            
    map = np.where(np.isnan(X), 1, 0)
    num_nans = np.sum(map, axis = axis, keepdims = True)


    if strategy=="mean":
        new_X = np.where(np.isnan(X), 0, X)
        X_sum = np.sum(new_X, axis=axis)
        X_mean = X_sum / (X.shape[0] - num_nans)
        X_mean = np.where(np.isnan(X_mean), 0, X_mean)
        print(map)
        map = map * X_mean 
        print(map)
        X = np.where(np.isnan(X), map, X)
        if one_d:
            X = np.squeeze(X)
        return X

    if strategy=="median":
        new_X = np.where(np.isnan(X), 0, X)
        new_X = np.sort(new_X, axis=axis)
        X_median = [[ np.median(new_X[num_nans[0][i]:,i]) ] for i in range(new_X.shape[1])]        
        X_median = np.where(np.isnan(X_median), 0, X_median)
        X_median = np.array(X_median)
        map = map * X_median.T
        X = np.where(np.isnan(X), map, X)
        if one_d:
            X = np.squeeze(X)
        return X
        
        
            
        
    
    pass