'''
GeneticAlgorithm-Crossover
@Author: JamieChang
@Date: 2020/05/27
'''
import numpy as np
from typing import Tuple

def single_point_crossover(chormosome1:np.ndarray,chormosome2:np.ndarray) -> Tuple[np.ndarray,np.ndarray]: # chormosomes should be 1 dimension array
    if chormosome1.ndim != 1 or chormosome2.ndim !=1:
        raise Exception("single_point_crossover(): the chormosomes should be 1 dimension numpy.ndarray")
    crosspoint = np.random.randint(chormosome1.size)
    offspring1 = np.concatenate((chormosome1[:crosspoint],chormosome2[crosspoint:]))
    offspring2 = np.concatenate((chormosome2[:crosspoint],chormosome1[crosspoint:]))
    return offspring1,offspring2

def global_crossover(chormosome1:np.ndarray,chormosome2:np.ndarray) -> Tuple[np.ndarray,np.ndarray]: # chormosomes should be 1 dimension array
    if chormosome1.ndim != 1 or chormosome2.ndim !=1:
        raise Exception("global_crossover(): the chormosomes should be 1 dimension numpy.ndarray")
    offsprint1,chormosome1_copy = chormosome1.copy(),chormosome1.copy()
    offsprint2,chormosome2_copy = chormosome2.copy(),chormosome2.copy()
    rand = np.random.uniform(0,1,size=chormosome1.size)
    offsprint1[rand < 0.5] = chormosome2_copy[rand < 0.5]
    offsprint2[rand < 0.5] = chormosome1_copy[rand < 0.5]
    return offsprint1,offsprint2

def SBX(chormosome1:np.ndarray,chormosome2:np.ndarray,eta:float) -> Tuple[np.ndarray,np.ndarray]:
    if chormosome1.ndim != 1 or chormosome2.ndim !=1:
        raise Exception("SBX(): the chormosomes should be 1 dimension numpy.ndarray")
    rand = np.random.random(chormosome1.shape) # You have to use random.random instead of random.uniform(0,1) since rand cannot be equal to 1
    gamma = np.empty(shape=chormosome1.shape)
    gamma[rand <= 0.5] = (2*rand[rand <= 0.5])**(1/(eta+1))
    gamma[rand > 0.5] = (1/(2*(1-rand[rand > 0.5])))**(1/(eta+1))
    offspring1 = 0.5*((1+gamma)*chormosome1 + (1-gamma)*chormosome2)
    offspring2 = 0.5*((1-gamma)*chormosome1 + (1+gamma)*chormosome2)
    return offspring1,offspring2