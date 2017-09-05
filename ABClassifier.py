import numpy as np
from termcolor import colored
import time
import math

class clf():
    def __init__(self, p_count, curr_type_count):
        # Algorithm
        self.curr_heur_mat = np.ones([1, curr_type_count], dtype=float)
        self.obj_heur_mat = np.ones([1, p_count], dtype=float)
        pass
    
    def correct(self, error_mat, obj_learning_rate, curr_learning_rate, p_mat):
        curr_cont_mat = p_mat/p_mat.sum(axis=0)
    
        print(curr_cont_mat)
    
        obj_cont_mat = p_mat/p_mat.sum(axis=1).reshape(curr_type_count,1)
        
        cont_mat = ((obj_cont_mat + curr_cont_mat) / 
                (obj_cont_mat.sum() + curr_cont_mat.sum())) #(curr_cont_mat + obj_cont_mat)/2
        
        x = cont_mat.sum(axis=1)
        y = cont_mat.sum(axis=0)
     
        z = x.sum() + y.sum()
        
        x /= z
        y /= z
       
        print('x and y')
        print(x)
        print(y)
        print()
    
        self.obj_heur_mat += (error_mat * obj_learning_rate)* y
       
        # TODO: DOESN'T SCALE WITH DIFFERENT NUMBERS
        # TODO: DOESN'T SCALE WITH SMALL MATRICES
        
        # j = error_mat.sum()
        # if abs(j) > 1:
        #     j = 1/j # * curr_learning_rate
       
        #print('actual curr learning rate')
        #print(j)
        #print()
    
        self.curr_heur_mat += (error_mat * curr_learning_rate).sum() * x # x * j
        
        res_mat = np.dot(self.curr_heur_mat, self.obj_heur_mat * p_mat)
        
        rank_mat = res_mat.argsort()
        
        return res_mat

    def train(self, x, y, correcting_range, obj_learning_rate=0.1, curr_learning_rate=0.1):
        for i in range(len(x)):
            p_mat = x[i]
            exp_mat = y[i]
        
            res_mat = np.dot(self.curr_heur_mat, self.obj_heur_mat * p_mat)

            for j in range(correcting_range):
                print(colored('i:', 'red'), i)
                print()
                
            
                error_mat = (exp_mat - res_mat)
                print(colored('error mat', 'yellow'))
                print(error_mat)
                print()
                
                print(colored('object learning rate', 'yellow'))
                print(obj_learning_rate)
                print()
            
                print(colored('curr learning rate', 'yellow'))
                print(curr_learning_rate)
                print()
            
                res_mat = self.correct(error_mat, obj_learning_rate , curr_learning_rate, p_mat)
            
                print(colored('participant matrix:', 'cyan'))
                print(p_mat)
                print()
                
                print(colored('expected resultant matrix:', 'cyan'))
                print(exp_mat)
                print()
            
                print(colored('currency heuristic matrix:', 'green'))
                print(self.curr_heur_mat)
                print()
                
                print(colored('object heuristic matrix:', 'green'))
                print(self.obj_heur_mat)
                print()
                
                print(colored('resultant matrix:', 'green'))
                print(res_mat)
                print()

    def predict(self, x):
        res_mat = np.dot(self.curr_heur_mat, self.obj_heur_mat * x)
        return res_mat


p_count = 30
curr_type_count = 5

_clf = clf(p_count, curr_type_count)

# Training Data
p_mat = np.random.rand(curr_type_count, p_count)#
#p_mat = np.arange(p_count * curr_type_count).reshape(curr_type_count, p_count)
exp_mat = np.arange(p_count)#np.random.rand(1,p_count)#np.arange(p_count)
rank_exp_mat = exp_mat.argsort()

input_x = [p_mat]
input_y = [exp_mat]

# Learning
# Optimizing for expected matrix
obj_learning_rate = .1#1/(p_count * curr_type_count)#.02 #1/(p_count + curr_type_count) # Max learning rate
curr_learning_rate = .1#obj_learning_rate

_clf.train(input_x, input_y, 1000, obj_learning_rate, curr_learning_rate)


