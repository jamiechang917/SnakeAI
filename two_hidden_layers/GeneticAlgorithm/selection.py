'''
GeneticAlgorithm-Selection
@Author: JamieChang
@Date: 2020/06/08
'''
import numpy as np

def roulette_wheel(fitness_list)->list: # return parents indexes in list form, the index could be same
    probability = (np.array(fitness_list)-min(np.array(fitness_list)))/np.sum(np.array(fitness_list)-np.min(fitness_list))
    try:
        selected_index = np.random.choice(np.arange(probability.size),p=probability,size=2,replace=True)
    except:
        raise Exception("probility may be negative:",probability)
    return selected_index

def tournament_selection(fitness_list,tournament_size=2)->list:
    selected_index = []
    for _ in range(2):
        candidate = np.random.choice(np.arange(np.array(fitness_list).size),size=tournament_size,replace=False)
        selected_index.append(max(candidate,key=lambda index:fitness_list[index]))
    return selected_index
