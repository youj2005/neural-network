import numpy as np

def cost(y, y_e):
    return ((y-y_e)**2).mean()

print(cost(np.array([1, 2, 3]), np.array([1, 2, 3])