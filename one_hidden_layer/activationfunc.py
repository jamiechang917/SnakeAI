'''
ActivationFunction
@Author: JamieChang
@Date: 2020/06/02
'''
import numpy as np
class ActivationFunction():

    @staticmethod
    def linear(matrix):
        if type(matrix) != np.ndarray:
            raise TypeError("The matrix for activation_function should be numpy.ndarray.")
        return matrix
    @staticmethod
    def sigmoid(matrix):
        if type(matrix) != np.ndarray:
            raise TypeError("The matrix for activation_function should be numpy.ndarray.")
        return 1/(1+np.exp(-1*matrix))
    @staticmethod
    def softmax(matrix):
        if type(matrix) != np.ndarray:
            raise TypeError("The matrix for activation_function should be numpy.ndarray.")
        return np.exp(matrix)/np.sum(np.exp(matrix))
    @staticmethod
    def ReLU(matrix):
        if type(matrix) != np.ndarray:
            raise TypeError("The matrix for activation_function should be numpy.ndarray.")
        return np.maximum(matrix,0)
    @staticmethod
    def tanh(matrix):
        if type(matrix) != np.ndarray:
            raise TypeError("The matrix for activation_function should be numpy.ndarray.")
        return np.tanh(matrix)