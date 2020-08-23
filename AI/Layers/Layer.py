import random
import numpy as np
from AI.NeuralNetworkPackage.NeuralNetworkConstants import NeuralNetworkConstants as Const
from AI.NeuralNetworkPackage.ActivationFunction import sigmoidFunc
from AI.NeuralNetworkPackage.ActivationFunction import dSigmoidFunc
import itertools


class layer:
    weights = list()
    deltaWeights = list()

    def weightsInit(self, inputSize, outputSize):
        weights = list()
        for i in range((inputSize + 1) * outputSize):
            weights.append(random.uniform(-2, 2))
        return weights

    def __init__(self, inputSize, outputSize):
        self.inputSize = inputSize + 1
        self.outputSize = outputSize
        self.weights = self.weightsInit(inputSize, outputSize)
        self.deltaWeights = np.zeros((inputSize + 1) * outputSize)

    def run(self, input):
        self.input = input.copy()
        self.output = []
        self.input.append(1.0)
        for i in range(self.outputSize):
            tmpOutput = self.weights[i * self.inputSize:(i + 1) * self.inputSize]
            tmpOutput = sum(np.array(tmpOutput) * np.array(self.input))
            self.output.append(sigmoidFunc(tmpOutput))
        return self.output.copy()

    def train(self, error):  # backpropagation using gradient
        nextError = [0.0] * (self.inputSize - 1)
        delta = np.array(list(map(dSigmoidFunc, self.output))) * np.array(error)    # derivative calculated by multiplying node input by error at this node
        inputList = list(self.input) * self.outputSize  # temporary lists to fasten the algorithm
        outputList = list(itertools.chain.from_iterable(itertools.repeat(x, self.inputSize) for x in delta))
        deltaWeight = np.array(inputList) * np.array(outputList) * Const.learningRate
        self.weights = np.array(self.weights) + Const.momentum * np.array(self.deltaWeights) + np.array(deltaWeight)    # delta weight calculated by multiplying derivative by learning rate, incrementing by previous difference of weight multiplied by momentum
        self.deltaWeights = deltaWeight
        for i in range(self.inputSize - 1):
            partialWeightArray = self.weights[i::self.inputSize]                # calculating error used in next layer
            nextError[i] = sum(np.array(partialWeightArray) * np.array(delta))  # multiplying previous error by weights from this node multiplied by input, nodes that don't affect output won't be changed
        return nextError

    def save(self, n):
        name = 'Layer'
        name += n
        name += '_weights.npy'
        name1 = 'Layer'
        name1 += n
        name1 += '_deltaWeights.npy'
        np.save(Const.layers_folder + "/" + name, np.array(self.weights))
        np.save(Const.layers_folder + "/" + name1, np.array(self.deltaWeights))

    def load(self, n):
        name = 'Layer'
        name += n
        name += '_weights.npy'
        name1 = 'Layer'
        name1 += n
        name1 += '_deltaWeights.npy'
        self.weights = np.load(Const.layers_folder + "/" + name)
        self.deltaWeights = np.load(Const.layers_folder + "/" + name1)
