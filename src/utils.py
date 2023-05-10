import numpy as np

def find_max_l2_norm(arrays):
    max_norm = 0.0
    
    for array in arrays:
        norms = np.linalg.norm(array, axis=1)  # Calculate L2-norm along rows
        max_norm = max(max_norm, np.max(norms))
    
    return max_norm