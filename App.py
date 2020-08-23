from AI.NeuralNetworkPackage.NeuralNetwork import NeuralNetwork
from AI.NeuralNetworkPackage.NeuralNetworkConstants import NeuralNetworkConstants as Const
from ImageConverter import ImageConverter
from os import listdir
from os.path import isfile, join
import time
import sys


class App:
    @staticmethod
    def create_tests():                           # creates tests input photos from row of numbers/sings
        my_path = Const.pre_test_folder           # folder with photos to cut out
        only_files = [f for f in listdir(my_path) if isfile(join(my_path, f))]

        for path in only_files:
            image_converter = ImageConverter(save_images=1)
            input_data = image_converter.get_ai_input_data(my_path+"/"+path)

    @staticmethod
    def print_result(tmp):
        max = 0
        for j in range(14):
            if tmp[j] > tmp[max]:
                max = j
        if max == 0:
            second = 1
        else:
            second = 0
        for j in range(14):
            if tmp[j] > tmp[second] and j != max:
                second = j
        print(round(tmp[max] / sum(tmp) * 100, 2), end='% ')
        print(Const.names[max], end=" ")
        print(round(tmp[second] / sum(tmp) * 100, 2), end='% ')
        print(Const.names[second], end=" ")

    @staticmethod
    def learn_machine(iterations):
        start = time.time()
        neural_network = NeuralNetwork(600, 400, 14)
        neural_network.load()
        trainingData = [[[] for _ in range(30)] for _ in range(len(Const.names))]
        for i in range(len(Const.training_result)):
            for j in range(30):
                file_path = Const.training_folder
                file_path += "/"
                file_path += Const.names[i]
                file_path += "/"
                file_path += str(j)
                file_path += ".jpg"
                trainingData[i][j] = ImageConverter.get_raw_data(path=file_path)
        for n in range(iterations):
            print(n)
            for file_number in range(30):
                for char_number in range(len(Const.names)):
                    neural_network.train(trainingData[char_number][file_number], Const.training_result[char_number])
            neural_network.save()
        print("Time: %.2f " % (time.time() - start))


    @staticmethod
    def run(path):
        image_converter = ImageConverter(save_images=1)
        start = time.time()
        input_data = image_converter.get_ai_input_data(path)
        print("Image conversion to data time:  %.2f s" % (time.time() - start))
        #  prepare NN to work
        x = NeuralNetwork(600, 400, 14)
        x.load()
        start = time.time()
        for data in input_data:
            result = x.run(data)
            print(Const.characters[result.index(max(result))], end='')
        print("")
        print("Neural network time:  %.2f s" % (time.time() - start))


# ------------------------------------------------
# Start
print("Please enter image that u would like to test")
image_path = input()
App.run(image_path)


