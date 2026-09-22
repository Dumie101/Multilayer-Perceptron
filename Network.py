import numpy as np

# Number of neurons in each layer
sizes = [2,3,1]

# Create the biases
biases = []

for layer_size in sizes[1:]:
    bias = np.random.rand(layer_size, 1)
    biases.append(bias)

# Create the weights
weights = []

for i in range(len(sizes) - 1):
    number_of_inputs = sizes[i]
    number_of_outputs = sizes[i + 1]

    weight = np.random.rand(number_of_outputs, number_of_inputs)
    weights.append(weight)

# Input values
a = np.array([
    [1],
    [0],
])

# Pass the input through each layer
for i in range(len(weights)):
    weight = weights[i]
    bias = biases[i]

    z = np.dot(weight, a) + bias

    # Apply sigmoid
    a = 1 / (1 + np.exp(-z))
    print(a)


print(a)