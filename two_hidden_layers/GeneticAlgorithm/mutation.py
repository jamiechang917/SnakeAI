'''
GeneticAlgorithm-Mutation (for float-encoding chromosome)
@Author: JamieChang
@Date: 2020/05/27
'''
import numpy as np
from typing import Tuple

def random_uniform_mutation(chromosome:np.ndarray,mutation_rate:float,mutation_size:float) -> np.ndarray:
    if chromosome.ndim != 1:
        raise Exception("random_uniform_mutation(): the chromosome should be 1 dimension numpy.ndarray")
    chromosome_copy = chromosome.copy()
    for index in range(chromosome_copy.size):
        if np.random.rand() <= mutation_rate:
            chromosome_copy[index] = np.random.uniform(-mutation_size,mutation_size)
    return chromosome_copy

def random_gaussian_mutation(chromosome:np.ndarray,mutation_rate:float,mutation_size:float,mean:float,sigma:float) -> np.ndarray:
    if chromosome.ndim != 1:
        raise Exception("random_gaussian_mutation(): the chromosome should be 1 dimension numpy.ndarray")
    chromosome_copy = chromosome.copy()
    mutate_index = np.random.random(chromosome_copy.shape) <= mutation_rate
    mutation_array = np.random.normal(mean,sigma,size=chromosome_copy.shape)
    chromosome_copy[mutate_index] += mutation_size*(mutation_array[mutate_index])
    return chromosome_copy
