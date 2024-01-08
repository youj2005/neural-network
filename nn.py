import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def sig(arr):
    return (1/(1+np.exp(-arr)))

def cost(y_pred, y_true):
    return ((y_pred - y_true)**2).mean()

def sig_deriv(arr):
    return sig(arr)*(1-sig(arr))

def accuracy(y_pred, y_true):
    acc = y_pred.argmax(axis=0) == y_true.argmax(axis=0)
    return acc.mean()

# Load dataset
data = load_iris()

# Get features and target
X=data.data
y=data.target

#Split data into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)

print(y_train)

input = X_train.T
input_size = input.shape[0]
real_output = y_train[np.newaxis, :]
output_size = real_output.shape[0]

print(input.shape, real_output.shape)

first_size = 50

first_weights = np.random.random((first_size, input_size))*2-1
first_bias = np.random.random((first_size, 1))*2-1

second_size = 50

second_weights = np.random.random((second_size, first_size))*2-1
second_bias = np.random.random((second_size, 1))*2-1

output_weights = np.random.random((output_size, second_size))*2-1
output_bias = np.random.random((output_size, 1))*2-1

iterations = 1000
iter = np.arange(iterations)
costs = np.zeros(iterations)

for i in range(iterations):
#first layer

    first_layer = sig(first_weights.dot(input) + first_bias)

    #second layer

    second_layer = sig(second_weights.dot(first_layer) + second_bias)

    #output layer

    output_layer = output_weights.dot(second_layer) + output_bias
    print(output_layer)
    costs[i] = accuracy(output_layer, real_output)

    #backpropagation

    learning_rate = 0.1

    #output layer

    error_final = (real_output-output_layer)
    out_weight_pd = error_final[:, np.newaxis, :] * second_layer[np.newaxis, :, :]
    out_bias_pd = error_final
    avg_out_weight_pd = np.mean(out_weight_pd, axis=2, keepdims=False)
    avg_out_bias_pd = np.mean(out_bias_pd, axis=1, keepdims=True)

    #second layer

    error_second = output_weights.T.dot(error_final) * sig_deriv(second_weights.dot(first_layer) + second_bias)
    second_weight_pd = error_second[:, np.newaxis, :] * first_layer[np.newaxis, :, :]
    second_bias_pd = error_second
    avg_second_weight_pd = np.mean(second_weight_pd, axis=2, keepdims=False)
    avg_second_bias_pd = np.mean(second_bias_pd, axis=1, keepdims=True)

    #first layer

    error_first = second_weights.T.dot(error_second) * sig_deriv(first_weights.dot(input) + first_bias)
    first_weight_pd = error_first[:, np.newaxis, :] * input[np.newaxis, :, :]
    first_bias_pd = error_first
    avg_first_weight_pd = np.mean(first_weight_pd, axis=2, keepdims=False)
    avg_first_bias_pd = np.mean(first_bias_pd, axis=1, keepdims=True)

    output_weights += learning_rate * avg_out_weight_pd
    output_bias += learning_rate * avg_out_bias_pd
    second_weights += learning_rate * avg_second_weight_pd
    second_bias += learning_rate * avg_second_bias_pd
    first_weights += learning_rate * avg_first_weight_pd
    first_bias += learning_rate * avg_first_bias_pd
    
plt.plot(iter, costs) 
plt.xlabel("X-axis")  # add X-axis label
plt.ylabel("Y-axis")  # add Y-axis label 
plt.title("Iter x Costs")  # add title 
plt.show() 