import numpy as np

class Network:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes

        self.biases = []
        for layer in sizes[1:]:
            layer_biases = np.random.rand(layer, 1)
            self.biases.append(layer_biases)

        self.weights = []
        for input_layer, output_layer in zip(sizes[:-1], sizes[1:]):
            layer_weights = np.random.rand(output_layer, input_layer)
            self.weights.append(layer_weights)

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w, a) + b)
        return a