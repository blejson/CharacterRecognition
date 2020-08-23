from AI.Layers.Layer import layer
import operator
import time

class NeuralNetwork:
    layers = []

    def __init__(self, inputSize,hiddenLayerSize, outputSize):
        self.inputSize = inputSize
        self.hiddenLayerSize = hiddenLayerSize
        self.outputSize = outputSize
        self.layers.append(layer(self.inputSize, self.hiddenLayerSize))
        self.layers.append(layer(self.hiddenLayerSize, self.outputSize))

    def run(self, data):
        for layer in self.layers:
            data = layer.run(data)
        return data

    def train(self, input, targetOutput):
        output = self.run(input)
        error = list(map(operator.sub, targetOutput, output))
        for i in range(len(self.layers)):
            error = self.layers[len(self.layers)-i-1].train(error)


    def save(self):
        self.layers[0].save('0')
        self.layers[1].save('1')

    def load(self):
        self.layers[0].load('0')
        self.layers[1].load('1')
