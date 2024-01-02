import numpy
import math

def sigmoid(arr):
    return (1/(1+math.e**(-arr)))

print(sigmoid(100))