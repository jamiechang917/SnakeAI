'''
SnakeGame with GA(one_hidden_layer)
@Author: JamieChang
@Date: 2020/06/06
'''
'''
problem1 : not using crossover rate
problem2 : in selection, wheel method, there exist a bug.
'''


import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
import os

from activationfunc import ActivationFunction
import neuralnet
import snakegame
import GUI
from GeneticAlgorithm import crossover,mutation,fitness,selection

#===============================================================================================================#
# Global
MAPSIZE = 10
SNAKE_LENGTH = 4 # initial snake length
FOODS = 1 # number of foods
TOP = 0.1 # 0.2=20%

# Batch Normalization
GAMMA = 1
BETA = 0

# SBX
ETA = 100

# Seed for model evaluation
SEED = 0 # Set to 0 for random seed (2020->17scores)

#random gaussian mutation
MEAN = 0
SIGMA = 1

# GA Global
POPULATION_SIZE = 1000
MAXIMUM_GENERATION = 5001
CROSSOVER_RATE = 0.5
MUTATION_RATE = 0.02
MUTATION_SIZE = 1
GAMES_SNAKE_PLAY = 1 # How many games each snake play, the average fitness will become its fitness

# GUI Global
DISPLAY_INTERVAL = 100
NUMBER_OF_VIDEOS = 25
FPS = 15 # how many moves of snake in one second in display
FFMPEG_PATH = r'C:\Users\Jamie Chang\Desktop\FINAL_VERSION\FFMPEG\bin\ffmpeg.exe' # for recording the gameplay
VIDEO_SAVE_PATH = r'C:\Users\Jamie Chang\Desktop\FINAL_VERSION\test_training\TEST2_REPORT\DEEP_TRAIN'
#===============================================================================================================#
def initialize_snake_layers(snake:snakegame.Snake) -> None:
    inputlayer = neuralnet.Layer(16,activation_func=ActivationFunction.tanh)
    hiddenlayer1 = neuralnet.Layer(8,activation_func=ActivationFunction.tanh)
    outputlayer = neuralnet.Layer(4,activation_func=ActivationFunction.linear)
    neuralnet.construct_layers(layers_list=[inputlayer,hiddenlayer1,outputlayer],seed=SEED)
    snake.layers = {"Input":inputlayer,"Hidden1":hiddenlayer1,"Output":outputlayer}

def decide_direction(snake:snakegame.Snake) -> str: # return direction
    snake.layers["Input"].values = snake.ML_output_4_directions()
    neuralnet.forward_propogation(snake.layers["Input"],snake.layers["Hidden1"],batch_normalize=True,gamma=GAMMA,beta=BETA)
    neuralnet.forward_propogation(snake.layers["Hidden1"],snake.layers["Output"],batch_normalize=True,gamma=GAMMA,beta=BETA)
    def check_direction_validity(direction):
        nonlocal snake
        # opposites = ({"U","D"},{"D","U"},{"L","R"},{"R","L"})
        # if {direction,snake.direction} in opposites:
        #     return snake.direction
        # else:
        #     return direction
        return direction
    return check_direction_validity(["U","D","L","R"][np.where(snake.layers["Output"].values == np.max(snake.layers["Output"].values))[0][0]])

def create_snakes_population(population_size=POPULATION_SIZE)->list:
    snake_list = []
    for ID in range(population_size):
        snake = snakegame.Snake(ID=ID,length=SNAKE_LENGTH,mapsize=MAPSIZE,foods=FOODS)
        initialize_snake_layers(snake)
        snake_list.append(snake)
    return snake_list

def gameplay(snakelist) -> tuple:
    fitness_list = []
    scores_list = []
    steps_list = []
    for snake in snakelist:
        sum_fitness = 0
        sum_scores = 0
        sum_steps = 0
        for _ in range(GAMES_SNAKE_PLAY):
            snake.init_parameters()
            while True:
                snake.direction = decide_direction(snake)
                actions = snake.perform_actions()

                if bool(actions) == True:
                    sum_fitness += fitness.fitness(snake)
                    sum_scores += snake.score
                    sum_steps += snake.steps
                    break
        fitness_list.append(sum_fitness//GAMES_SNAKE_PLAY)
        scores_list.append(sum_scores/GAMES_SNAKE_PLAY)
        steps_list.append(sum_steps/GAMES_SNAKE_PLAY)
    return (fitness_list,scores_list,steps_list)


def reproduce(snakelist,gameplay_result):
    snakelist_copy = snakelist.copy()
    for ID in range(POPULATION_SIZE//2):
        parent1_index,parent2_index = selection.tournament_selection(gameplay_result[0],tournament_size=5)
        parent1,parent2 = snakelist[parent1_index],snakelist[parent2_index]
        for layer_name in ["Input","Hidden1"]:
            chormosome1_W = parent1.layers[layer_name].weight
            chormosome1_b = parent1.layers[layer_name].bias
            chormosome2_W = parent2.layers[layer_name].weight
            chormosome2_b = parent2.layers[layer_name].bias
            weight_shape = chormosome1_W.shape
            bias_shape = chormosome1_b.shape
            child1_W,child2_W = crossover.SBX(chormosome1_W.flatten(),chormosome2_W.flatten(),eta=ETA)
            child1_b,child2_b = crossover.SBX(chormosome1_b.flatten(),chormosome2_b.flatten(),eta=ETA)
            child1_W = mutation.random_gaussian_mutation(child1_W,mutation_rate=MUTATION_RATE,mutation_size=MUTATION_SIZE,mean=MEAN,sigma=SIGMA)
            child1_b = mutation.random_gaussian_mutation(child1_b,mutation_rate=MUTATION_RATE,mutation_size=MUTATION_SIZE,mean=MEAN,sigma=SIGMA)
            child2_W = mutation.random_gaussian_mutation(child2_W,mutation_rate=MUTATION_RATE,mutation_size=MUTATION_SIZE,mean=MEAN,sigma=SIGMA)
            child2_b = mutation.random_gaussian_mutation(child2_b,mutation_rate=MUTATION_RATE,mutation_size=MUTATION_SIZE,mean=MEAN,sigma=SIGMA)
            # child1_W = mutation.random_uniform_mutation(chromosome=child1_W,mutation_rate=MUTATION_RATE,mutation_size=MUTATION_SIZE)
            # child1_b = mutation.random_uniform_mutation(chromosome=child1_b,mutation_rate=MUTATION_RATE,mutation_size=MUTATION_SIZE)
            # child2_W = mutation.random_uniform_mutation(chromosome=child2_W,mutation_rate=MUTATION_RATE,mutation_size=MUTATION_SIZE)
            # child2_b = mutation.random_uniform_mutation(chromosome=child1_b,mutation_rate=MUTATION_RATE,mutation_size=MUTATION_SIZE)

            snakelist_copy[ID].layers[layer_name].weight,snakelist_copy[POPULATION_SIZE-ID-1].layers[layer_name].weight = child1_W.reshape(weight_shape),child2_W.reshape(weight_shape)
            snakelist_copy[ID].layers[layer_name].bias,snakelist_copy[POPULATION_SIZE-ID-1].layers[layer_name].bias = child1_b.reshape(bias_shape),child2_b.reshape(bias_shape)
    #Top
    top_index = np.argsort(gameplay_result[0])[int(POPULATION_SIZE*(1-TOP)):]
    for index in top_index:
        for layer_name in ["Input","Hidden1"]:
            snakelist_copy[index].layers[layer_name].weight = snakelist[index].layers[layer_name].weight 
            snakelist_copy[index].layers[layer_name].bias = snakelist[index].layers[layer_name].bias

    
    
    return snakelist_copy            
        


if __name__ == '__main__':
    fitness_every_gen = []
    scores_every_gen = []
    best_scores_every_gen = []
    snake_list = create_snakes_population()
    start_time = datetime.datetime.now()
    record_video_time = datetime.datetime.now()-start_time

    for gen in range(MAXIMUM_GENERATION):
        result = gameplay(snake_list)
        top_index = np.argsort(result[0])[int(POPULATION_SIZE*(1-TOP)):]
        fitness_every_gen.append(np.average(np.array(result[0]).astype(np.float)))
        scores_every_gen.append(np.average(result[1]))
        best_scores_every_gen.append(np.max(result[1]))

        if (gen!=0 and (gen%DISPLAY_INTERVAL == 0 or gen%(MAXIMUM_GENERATION-1)==0))  or gen==25 or gen==50 :
            snake_list_copy = snake_list.copy()
            fig = plt.figure(figsize=(16,9),dpi=80)
            plt.subplot(1,2,1)
            plt.plot(list(range(gen+1)),fitness_every_gen)
            plt.title("Fitness",fontsize=16)
            plt.xlabel("Generations",fontsize=14)
            plt.ylabel("Fitness",fontsize=14)
            plt.subplot(1,2,2)
            plt.plot(list(range(gen+1)),scores_every_gen)
            plt.plot(list(range(gen+1)),best_scores_every_gen,color='r')
            plt.title("Scores",fontsize=16)
            plt.xlabel("Generations",fontsize=14)
            plt.ylabel("Scores",fontsize=14)
            plt.savefig(os.path.join(VIDEO_SAVE_PATH,f"gen_{gen}_result.png"))
            training_time = (datetime.datetime.now()-start_time-record_video_time).total_seconds()
            this_gen_record_video_start_time = datetime.datetime.now()
            for ID in range(POPULATION_SIZE):
                if ID in top_index[-NUMBER_OF_VIDEOS:]:
                    plot_parameters={
                        "generation": gen,
                        "ID": ID,
                        "bestscore": np.max(result[1]),
                        "fitness": format(result[0][ID],'.3E'),
                        "fitnessfunction": "steps^2 * 2^apples",
                        "populationsize": POPULATION_SIZE,
                        "crossoverrate": "",
                        "mutationrate": MUTATION_RATE,
                        "trainingtime": f"{int(training_time//86400)}d,{int((training_time%86400)//3600)}h,{int(((training_time%86400)%3600)//60)}m,{int((((training_time%86400)%3600)%60)//1)}s",
                        "layerstructure": "[16-tanh-8-tanh-4]",
                        "FFMPEG_PATH":FFMPEG_PATH,
                        "VIDEO_SAVE_PATH":VIDEO_SAVE_PATH,
                        "scores_list":scores_every_gen,
                        "best_scores_list":best_scores_every_gen
                        }

                    GUI.display(snake=snake_list_copy[ID],mapsize=MAPSIZE,gamespeed=FPS,parameters=plot_parameters)
            record_video_time = record_video_time + datetime.datetime.now() - this_gen_record_video_start_time
        snake_list = reproduce(snake_list,result)
        print(f"GEN: {gen}, BestScores: {np.max(result[1])}, BestFitness: {np.max(result[0]):.3E}, AvgScores: {np.average(result[1]):.3f}, AvgSteps: {np.average(result[2]):.2f}, AvgFitness: {np.average(np.array(result[0]).astype(np.float)):.3E}")
