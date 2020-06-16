'''
GUI for SnakeGame
@Author: JamieChang
@Date: 2020/06/09
'''
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib import colors
import os
from snakegame import Snake


def draw_map(snake:Snake,mapsize:int) -> np.ndarray:
    game_map = np.zeros(shape=(mapsize,mapsize))
    for food_position in snake.food_position:
        game_map[food_position[0]][food_position[1]] = 1
    for body_position in snake.snake_position:
        game_map[body_position[0]][body_position[1]] = -1
    game_map[snake.snake_position[0][0]][snake.snake_position[0][1]] = 2
    return game_map

def display(snake:Snake,mapsize:int,gamespeed:int,parameters={}):
    #parameters:{generation,bestscore,fitness,finess,fitnessfunction,layerstructure,populationsize,crossoverrate,mutationrate,trainingtime}
    if type(snake)!=Snake: return TypeError("type(snake) should be class 'Snake'")
    snake.init_parameters()
    #======Initialize=================================================================================#
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    from matplotlib import colors
    import matplotlib as mpl
    import sys

    import warnings
    import matplotlib.cbook
    warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
    mpl.rcParams['agg.path.chunksize'] = 100000
    plt.rcParams['animation.ffmpeg_path'] = parameters['FFMPEG_PATH']

    cmap = colors.ListedColormap(['black','white', 'red', 'blue']) 
    bounds = [-1.5,-0.5,0.5,1.5,2.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    fig = plt.figure(figsize=(16,9),dpi=80)

    # initialize the subplots
    plt.subplot(1,2,1)
    plt.imshow(draw_map(snake=snake,mapsize=mapsize))
    grid_ticks = np.arange(0.5,mapsize+0.5,1.0)
    for i in grid_ticks:
        plt.gca().axvline(i,color='grey',alpha=0.5)
        plt.gca().axhline(i,color='grey',alpha=0.5)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2,2,4) # Parameter's text box
    plt.xlim((0,16))
    plt.ylim((0,16))
    plt.xticks([])
    plt.yticks([])
    plt.title("Parameters")
    t1=plt.text(x=0.5,y=14.0,s=f"Generations: Gen{parameters['generation']}_ID{parameters['ID']}",fontsize=11)
    t2=plt.text(x=0.5,y=12.0,s=f"Best Scores: {parameters['bestscore']}",fontsize=12)
    t3=plt.text(x=0.5,y=10.0,s="Scores: ",fontsize=12)
    t4=plt.text(x=0.5,y=8.0,s="HP: ",fontsize=12)
    t5=plt.text(x=0.5,y=6.0,s="Steps: ",fontsize=12)
    t6=plt.text(x=0.5,y=4.0,s=f"LayerStructure: {parameters['layerstructure']}",fontsize=12)
    t7=plt.text(x=0.5,y=2.0,s=f"Fitness Function: {parameters['fitnessfunction']}",fontsize=12)
    t8=plt.text(x=7.0,y=14.0,s=f"Fitness: {parameters['fitness']}",fontsize=12)
    t9=plt.text(x=7.0,y=12.0,s=f"Population Size: {parameters['populationsize']}",fontsize=12)
    t10=plt.text(x=7.0,y=10.0,s=f"Direction: {snake.direction}",fontsize=12)
    t11=plt.text(x=7.0,y=8.0,s=f"Mutation Rate: {parameters['mutationrate']}",fontsize=12)
    t12=plt.text(x=7.0,y=6.0,s=f"Training Time: {parameters['trainingtime']}",fontsize=12)

    plt.subplot(2,2,2)
    plt.plot(list(range(parameters["generation"]+1)),parameters["scores_list"])
    plt.plot(list(range(parameters["generation"]+1)),parameters["best_scores_list"],color='r')
    plt.legend(["Avg. Scores","Best Scores"])
    plt.title("Scores")
    plt.xlabel("Generations")
    plt.ylabel("Scores")

    #===========================================================================================#
    
    
    from main import decide_direction
    global_map = plt.subplot(1,2,1).imshow(draw_map(snake=snake,mapsize=mapsize),cmap=cmap,norm=norm)
    from matplotlib.animation import FuncAnimation
    best_scores = 0

    def frames_generator():
        nonlocal snake
        while True:
            yield draw_map(snake=snake,mapsize=mapsize)
            snake.direction = decide_direction(snake=snake)
            actions = snake.perform_actions()
            if bool(actions) is True: # Collisions or HP
                yield (actions,snake.direction,snake.score)
                ani.event_source.stop()
                break
        ani.event_source.stop()

    def update(frame):
        nonlocal global_map,best_scores
        if type(frame) == tuple:
            t10.set_text(f"Death:{frame[0]}, {frame[1]}")
            best_scores = frame[2]
        else:
            global_map.set_data(frame)
            plt.subplot(1,2,1).set_xlabel(f"Generations:{parameters['generation']}, Scores: {snake.score}, HP: {snake.HP}",fontsize=12)
            
            if snake.score>int(parameters['bestscore']):
                t2.set_text(f"Best Scores: {snake.score}")
            t3.set_text(f"Scores: {snake.score}")
            t4.set_text(f"HP: {snake.HP}")
            t5.set_text(f"Steps: {snake.steps}")
            t10.set_text(f"Direction: {snake.direction}")


        
    ani = FuncAnimation(fig=fig,func=update,frames=frames_generator,interval=1000/gamespeed,repeat=False,save_count=sys.maxsize)
    save_directory = parameters["VIDEO_SAVE_PATH"]
    save_name_with_path = os.path.join(save_directory,f"Gen{parameters['generation']}_ID{parameters['ID']}.mp4")
    ani.save(save_name_with_path,fps=gamespeed)
    plt.close(fig)