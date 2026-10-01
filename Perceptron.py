import random

def activation_function(num):
    if num >= 0:
        return 1
    else:
        return -1

class Perceptron:
    def __init__(self):
        self.weights = [0.0] * 2

        for i in range(len(self.weights)):
            self.weights[i] = random.uniform(-1, 1)

    def guess(self, inputs):
        sum = 0
        for i in range(len(self.weights)):
            sum += inputs[i] * self.weights[i]

        activated_sum = activation_function(sum)
        return activated_sum