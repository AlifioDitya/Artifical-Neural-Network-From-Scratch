import numpy as np

# Sigmoid activation function
def sigmoid(x, derivative=False):
    if derivative:
        return sigmoid(x) * (1 - sigmoid(x))
    
    return 1 / (1 + np.exp(-x))

# ReLU activation function
def relu(x, derivative=False):
    if derivative:
        return np.where(x <= 0, 0, 1)
    
    return np.maximum(0, x)

# Softmax activation function
def softmax(x, derivative=False):
    if derivative:
        s = softmax(x)
        return s * (1 - s)
    
    exp = np.exp(x)
    return exp / np.sum(exp, axis=-1, keepdims=True)