import math


def sigmoidFunc(x):
    if x < -700: # dla wartości mniejszych niż około -800 wywala błąd, wyjście poza zakres
        return 0
    return 1 / (1 + math.exp(-x))


def dSigmoidFunc(x):
    return (sigmoidFunc(x)*(1-sigmoidFunc(x)))
