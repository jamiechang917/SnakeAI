'''
SnakeGame
@Author: JamieChang
@Date: 2020/06/02
'''
import numpy as np
import random

class Snake():
    def __init__(self,ID,length,mapsize,foods):
        self.ID = ID
        self.direction = "DIRECTION"
        self.snake_position = [("y_postion","x_position")]
        self.food_position = set() # "set" form, there could be multiple foods
        self.HP = mapsize**2-length
        self.score = 0
        self.steps = 0
        self.mapsize = mapsize
        self.foods = foods
        self.length = length
        self.layers = {} #{"Input":,"Hidden1":,"Hidden2":,"Output":}
        
        self.generate_snake_position()
        for _ in range(foods):
            self.generate_food_position()

    def init_parameters(self): # initialize some parameters of snake
        self.HP = self.mapsize**2-self.length
        self.score = 0
        self.steps = 0
        self.direction = "DIRECTION"
        self.snake_position = [("y_postion","x_position")]
        self.food_position = set() # "set" form, there could be multiple foods
        self.generate_snake_position()
        for _ in range(self.foods):
            self.generate_food_position()
    
    def generate_snake_position(self):
        head = [(random.randint(1,self.mapsize-2),random.randint(1,self.mapsize-2))] #(y_position, x_position)
        if head[0][0] <= self.mapsize//2:
            if head[0][1] <= self.mapsize//2:
                if random.randint(0,1) == 0:
                    head.extend([(head[0][0],head[0][1]+i) for i in range(1,self.length)])
                    self.snake_position = head
                    self.direction = "L"
                else: 
                    head.extend([(head[0][0]+i,head[0][1]) for i in range(1,self.length)])
                    self.snake_position = head
                    self.direction = "U"
            elif head[0][1] > self.mapsize//2:
                if random.randint(0,1) == 0:
                    head.extend([(head[0][0],head[0][1]-i) for i in range(1,self.length)])
                    self.snake_position = head
                    self.direction = "R"
                else:
                    head.extend([(head[0][0]+i,head[0][1]) for i in range(1,self.length)])
                    self.snake_position = head
                    self.direction = "U"
        elif head[0][0] > self.mapsize//2:
            if head[0][1] <= self.mapsize//2:
                if random.randint(0,1) == 0:
                    head.extend([(head[0][0],head[0][1]+i) for i in range(1,self.length)])
                    self.snake_position = head
                    self.direction = "L"
                else: 
                    head.extend([(head[0][0]-i,head[0][1]) for i in range(1,self.length)])
                    self.snake_position = head
                    self.direction = "D"
            elif head[0][1] > self.mapsize//2:
                if random.randint(0,1) == 0:
                    head.extend([(head[0][0],head[0][1]-i) for i in range(1,self.length)])
                    self.snake_position = head
                    self.direction = "R"
                else:
                    head.extend([(head[0][0]-i,head[0][1]) for i in range(1,self.length)])
                    self.snake_position = head
                    self.direction = "D"

    def generate_food_position(self):
        while True:
            food = (random.randint(0,self.mapsize-1),random.randint(0,self.mapsize-1))
            if food not in set(self.snake_position)|self.food_position:
                self.food_position.add(food)
                break
            if set(self.snake_position)|self.food_position == set([(i,j) for i in range(self.mapsize) for j in range(self.mapsize)]):
                break
            

    def move(self,keep_length=True):
        if self.direction == "U":
            new_head = [(self.snake_position[0][0]-1,self.snake_position[0][1])]
        elif self.direction == "D":
            new_head = [(self.snake_position[0][0]+1,self.snake_position[0][1])]
        elif self.direction == "L":
            new_head = [(self.snake_position[0][0],self.snake_position[0][1]-1)]
        elif self.direction == "R":
            new_head = [(self.snake_position[0][0],self.snake_position[0][1]+1)]
        if keep_length==True:
            new_head.extend(self.snake_position[:-1]) # chop off the last part of snake
            self.snake_position = new_head
            self.HP -= 1
            self.steps += 1
        else:
            new_head.extend(self.snake_position)
            self.snake_position = new_head
            self.HP -= 1
            self.steps += 1

    def check_food_collisions(self):
        if self.snake_position[0] in self.food_position:
            return True
        return False
    
    def check_collisions(self):
        if self.snake_position[0][0] in {-1,self.mapsize} or self.snake_position[0][1] in {-1,self.mapsize}:
            return "Border"
        elif self.snake_position[0] in self.snake_position[1:]:
            return "Body"

    #==================GameLogic=================#
    def perform_actions(self):
        if bool(self.check_food_collisions()) == True:
            self.food_position.remove(self.snake_position[0])
            self.generate_food_position()
            self.move(keep_length=False)
            self.score += 1
            self.HP += self.mapsize**2- (len(self.snake_position)+len(self.food_position))      
        else:
            self.move()

        collision = self.check_collisions()
        if bool(collision) is True:
            return collision
        if self.HP == 0:
            return "HP"
    #==============================================#

    def ML_output_4_directions(self): # binary output (0 or 1)
        # initialize
        head_location = self.snake_position[0]
        food_direction = np.array([0,0,0,0]) #[Up,Down,Left,Right]
        head_direction = np.array([0,0,0,0]) #[Up,Down,Left,Right]
        border_detection = np.array([0,0,0,0]) #[Up,Down,Left,Right]
        body_detection = np.array([0,0,0,0]) #[Up,Down,Left,Right]
        # food direction
        for food in self.food_position:
            if head_location[0]-food[0] < 0:
                if head_location[1] - food[1] < 0:
                    food_direction[1],food_direction[3] = 1,1
                elif head_location[1] - food[1] > 0:
                    food_direction[1],food_direction[2] = 1,1
                elif head_location[1] == food[1]:
                    food_direction[1] = 1
            elif head_location[0]-food[0] > 0:
                if head_location[1] - food[1] < 0:
                    food_direction[0],food_direction[3] = 1,1
                elif head_location[1] - food[1] > 0:
                    food_direction[0],food_direction[2] = 1,1
                elif head_location[1] == food[1]:
                    food_direction[0] = 1
            elif head_location[0] == food[0]:
                if head_location[1] - food[1] < 0:
                    food_direction[3] = 1
                elif head_location[1] - food[1] > 0:
                    food_direction[2] = 1
                elif head_location[1] == food[1]:
                    food_direction = np.zeros(shape=(4))
        # distance to border of four directions
        if head_location[0]+1 == self.mapsize : border_detection[1] =1
        elif head_location[0]-1 == -1 : border_detection[0] =1
        elif head_location[1]+1 == self.mapsize : border_detection[3] =1
        elif head_location[1]-1 == -1 : border_detection[2] =1
        # head direction
        if self.direction == "U": head_direction[0] = 1
        elif self.direction == "D": head_direction[1] = 1
        elif self.direction == "L": head_direction[2] = 1
        elif self.direction == "R": head_direction[3] = 1
        # body detection
        for body in self.snake_position[1:]:
            if body[0] == head_location[0]:
                if body[1] > head_location[1]: body_detection[3] = 1
                elif body[1] < head_location[1]: body_detection[2] = 1
            if body[1] == head_location[1]:
                if body[0] > head_location[0]: body_detection[1] = 1
                elif body[0] < head_location[0]: body_detection[0] = 1
        return np.concatenate([head_direction,food_direction,body_detection,border_detection])
    
    def ML_output_8_directions(self): # binary output (0 or 1), 28neurons
        # initialize
        head_location = self.snake_position[0]
        food_direction = np.array([0,0,0,0,0,0,0,0]) #[Up,Down,Left,Right,LU,RU,LD,RD]
        head_direction = np.array([0,0,0,0]) #[Up,Down,Left,Right]
        border_detection = np.array([0,0,0,0,0,0,0,0]) #[Up,Down,Left,Right,LU,RU,LD,RD]
        body_detection = np.array([0,0,0,0,0,0,0,0]) #[Up,Down,Left,Right,LU,RU,LD,RD]
        # food direction
        for food in self.food_position:
            if head_location[0] < food[0]:
                if head_location[1]  < food[1]:
                    if np.abs(head_location[0]-food[0])==np.abs(head_location[1]-food[1]):
                        food_direction[4] = 1
                elif head_location[1] > food[1]:
                    if np.abs(head_location[0]-food[0])==np.abs(head_location[1]-food[1]):
                        food_direction[5] = 1
                elif head_location[1] == food[1]:
                    food_direction[1] = 1
            elif head_location[0] > food[0]:
                if head_location[1]  < food[1]:
                    if np.abs(head_location[0]-food[0])==np.abs(head_location[1]-food[1]):
                        food_direction[6] = 1
                elif head_location[1] > food[1]:
                    if np.abs(head_location[0]-food[0])==np.abs(head_location[1]-food[1]):
                        food_direction[7] = 1
                elif head_location[1] == food[1]:
                    food_direction[0] = 1
            elif head_location[0] == food[0]:
                if head_location[1] < food[1]:
                    food_direction[3] = 1
                elif head_location[1] > food[1]:
                    food_direction[2] = 1

        # distance to border of four directions
        if head_location[0]+1 == self.mapsize : border_detection[1] =1
        elif head_location[0]-1 == -1 : border_detection[0] =1
        elif head_location[1]+1 == self.mapsize : border_detection[3] =1
        elif head_location[1]-1 == -1 : border_detection[2] =1
        elif head_location == (0,0): border_detection[4] = 1
        elif head_location == (self.mapsize-1,0): border_detection[6] = 1
        elif head_location == (0,self.mapsize-1): border_detection[5] = 1
        elif head_location == (self.mapsize-1,self.mapsize-1): border_detection[7] = 1


        # head direction
        if self.direction == "U": head_direction[0] = 1
        elif self.direction == "D": head_direction[1] = 1
        elif self.direction == "L": head_direction[2] = 1
        elif self.direction == "R": head_direction[3] = 1
        # body detection
        for body in self.snake_position[1:]:
            if body[0] == head_location[0]:
                if body[1] > head_location[1]: body_detection[3] = 1
                elif body[1] < head_location[1]: body_detection[2] = 1
            elif body[1] == head_location[1]:
                if body[0] > head_location[0]: body_detection[1] = 1
                elif body[0] < head_location[0]: body_detection[0] = 1
            elif body[0] < head_location[0]:
                if body[1] < head_location[1]:
                    if np.abs(head_location[0]-food[0])==np.abs(head_location[1]-food[1]):
                        body_detection[4] = 1
                elif body[1] > head_location[1]:
                    if np.abs(head_location[0]-food[0])==np.abs(head_location[1]-food[1]):
                        body_detection[5] = 1
            elif body[0] > head_location[0]:
                if body[1] < head_location[1]:
                    if np.abs(head_location[0]-food[0])==np.abs(head_location[1]-food[1]):
                        body_detection[6] = 1
                elif body[1] > head_location[1]:
                    if np.abs(head_location[0]-food[0])==np.abs(head_location[1]-food[1]):
                        body_detection[7] = 1
        
        return np.concatenate([head_direction,food_direction,body_detection,border_detection])

    def ML_output_global(self):
        import GUI
        global_map = GUI.draw_map(snake=self,mapsize=self.mapsize)
        return global_map.flatten()
    
    def ML_output_simple(self):
        head_location = self.snake_position[0]
        collision_detection = np.array([0,0,0,0])
        for body in self.snake_position[1:]:
            if body[0] == head_location[0]:
                if body[1] > head_location[1]: collision_detection[3] = 1
                elif body[1] < head_location[1]: collision_detection[2] = 1
            if body[1] == head_location[1]:
                if body[0] > head_location[0]: collision_detection[1] = 1
                elif body[0] < head_location[0]: collision_detection[0] = 1
        if head_location[0]+1 == self.mapsize : collision_detection[1] =1
        elif head_location[0]-1 == -1 : collision_detection[0] =1
        elif head_location[1]+1 == self.mapsize : collision_detection[3] =1
        elif head_location[1]-1 == -1 : collision_detection[2] =1
        return collision_detection