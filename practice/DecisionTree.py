import numpy as np
import matplotlib.pyplot as plt

# Sample data
X = np.array([1,2,3,4,5,6,7,8,9,10])
Y = np.array([5,6,10,10,13,14,16,19,21,22])

# Function to compute squared error
def squared_error(y):
    return np.var(y) * len(y)

# Find best split
def best_split(X, Y):
    best_loss = float("inf")
    best_idx = None
    
    for i in range(1, len(X)):  # possible split points
        left_y, right_y = Y[:i], Y[i:]
        loss = squared_error(left_y) + squared_error(right_y)
        
        if loss < best_loss:
            best_loss = loss
            best_idx = i
            
    return best_idx

# Train simple tree (depth=2)
split = best_split(X, Y)

left_mean = np.mean(Y[:split])
right_mean = np.mean(Y[split:])

# Predictions (piecewise constant)
Y_pred = np.where(X < X[split], left_mean, right_mean)

# Plot
plt.scatter(X, Y, color="blue", label="Data")
plt.step(X, Y_pred, where="mid", color="red", label="Decision Tree Prediction")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()
