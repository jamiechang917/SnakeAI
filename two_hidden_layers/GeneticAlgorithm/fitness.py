'''
GeneticAlgorithm-Fitness
@Author: JamieChang
@Date: 2020/05/27
'''
import numpy as np

def fitness(snake) -> float:
    return snake.steps*snake.steps*(2**snake.score)
    # return snake.score*1000 + snake.steps
    #return snake.steps + (2**(snake.score)+(snake.score**2.1)*500) - (snake.score**1.2)*((0.25*snake.steps)**(1.3))