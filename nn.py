import numpy as np
import math

def sigmoid(arr):
    return (1/(1+np.exp(-arr)))

def cost(y, y_e):
    return ((y-y_e)**2).sum()/(2*y.size)

input = np.random.random((100, 1))*2-1
real_output = np.array([0,0,0,0,1,0,0,0,0,0])
real_output = real_output.reshape((10, 1))

#first layer

first_size = 50

first_weights = np.random.random((first_size, input.size))*2-1
first_bias = np.random.random((first_size, 1))*2-1
first_layer = sigmoid(first_weights.dot(input) + first_bias)
print("First layer: ")
print(first_layer)

#second layer

second_size = 40

second_weights = np.random.random((second_size, first_size))*2-1
second_bias = np.random.random((second_size, 1))*2-1
second_layer = sigmoid(second_weights.dot(first_layer) + second_bias)
print("Second layer: ")
print(second_layer)

#output layer

output_size = 10

output_weights = np.random.random((output_size, second_size))*2-1
output_bias = np.random.random((output_size, 1))*2-1
output_layer = sigmoid(output_weights.dot(second_layer) + output_bias)
print("Output layer: ")
print(output_layer)
print("Cost: ")
print(cost(output_layer, real_output))