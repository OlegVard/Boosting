import math
import random

import numpy as np


class Network:
    def __init__(self, data, weight):
        self.weight = weight
        self.data = data
        self.data_iter = 0
        self.inputs = 9
        self.hidden = 3
        self.hidden_neurons = 5
        self.outs = 1
        self.weights = []
        self.create_network()

    @staticmethod
    def generate_weights(neuron_count):
        weights = []
        for i in range(neuron_count):
            weights.append(random.random())
        return weights

    def create_network(self):
        self.weights.append(self.generate_weights(self.inputs))
        for _ in range(self.hidden):
            self.weights.append(self.generate_weights(self.hidden_neurons))
        self.weights.append(self.generate_weights(self.outs))

    def feed_forward(self, train=True, test_data=None):
        if train:
            data = self.data[self.data_iter]
            self.data_iter += 1
        else:
            data = test_data
        results = []
        for i in range(len(self.weights)):
            prom_results = []
            for j in range(len(self.weights[i])):
                if i == 0:
                    prom_results.append(self.activate(data[j] * self.weights[i][j]))
                else:
                    neuron_mean = results[i-1][j] * self.weights[i][j]
                    prom_results.append(self.activate(neuron_mean))
            results.append(prom_results.copy())
        if train:
            self.back_propagation(results, data[-2])
        else:
            return results[-1], data[-2:]

    def back_propagation(self, results, reference):
        learning_rate = 0.0001
        error = np.square(reference - results[-1][0])/2
        for i in range(len(self.weights) - 1, - 1, -1):
            for j in range(len(self.weights[i])):
                self.weights[i][j] = self.weights[i][j] + (1 - results[i][j]) * learning_rate * error * results[i][j]

    @staticmethod
    def activate(x):
        return 1 / (1 + math.exp(-x))
