import pandas as pd
import numpy as np

df = pd.read_csv('mnist_784.csv')

# FOR TESTING
np.random.seed(50)

# Initialization
num_input_neurons = 784
num_hidden_neurons = 100
weights = np.random.rand(num_hidden_neurons, num_input_neurons)
bias = np.random.rand(num_hidden_neurons, 1)

first_digit = (df.iloc[0, 0:num_input_neurons] / 255).values.reshape(num_input_neurons, 1)
print(first_digit)
first_class = df.iloc[0,num_input_neurons]

dot_product = (weights @ first_digit) + bias
sigmoid_activation = 1 / (1 + np.exp(-dot_product))
relu_activation = np.maximum(0, dot_product)

print(sigmoid_activation)
print(relu_activation)