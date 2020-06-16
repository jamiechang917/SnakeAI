'''
NeuralNetwork
@Author: JamieChang
@Date: 2020/06/02
'''
import numpy as np

class Layer():
    def __init__(self,size,activation_func):
        self.size = size
        self.activation_function = activation_func
        self.values = np.ndarray
        self.weight = np.ndarray
        self.bias = np.ndarray

def construct_layers(layers_list=['layer1','layer2'],seed=False)-> None:
    if bool(seed)==True:
        for index,layer in enumerate(layers_list[:-1]):
            np.random.seed(seed)
            layer.weight = np.random.uniform(-1,1,size=(layer.size,layers_list[index+1].size))
            np.random.seed(seed)
            layer.bias = np.random.uniform(-1,1,size=layers_list[index+1].size)
    else:
        for index,layer in enumerate(layers_list[:-1]):
            layer.weight = np.random.uniform(-1,1,size=(layer.size,layers_list[index+1].size))
            layer.bias = np.random.uniform(-1,1,size=layers_list[index+1].size)

def forward_propogation(layer1,layer2,batch_normalize=True,gamma=1,beta=0) -> None:
    # forward propogation from layer1 to layer2: layer2 = activation_func_of_layer1(batch_norm[(weight_of_1*layer1)+bias])
    if batch_normalize == True:
        M = layer1.values.dot(layer1.weight)+layer1.bias
        layer2.values = layer1.activation_function(gamma*((M-np.mean(M))/(np.std(M)+1e-5))+beta)
    elif batch_normalize == False:
        layer2.values = layer1.activation_function(layer1.values.dot(layer1.weight)+layer1.bias)