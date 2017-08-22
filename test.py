import numpy as np
from termcolor import colored
import time
import math

def correct(error_mat, p_mat, obj_heur_mat, curr_heur_mat):
    print(colored('error matrix', 'yellow'))
    print(error_mat)
    print()
    
    curr_cont_mat = p_mat/p_mat.sum(axis=0)
    
    obj_cont_mat = p_mat/p_mat.sum(axis=1).reshape(curr_type_count,1)
    
    cont_mat = ((obj_cont_mat + curr_cont_mat) / 
            (obj_cont_mat.sum() + curr_cont_mat.sum())) #(curr_cont_mat + obj_cont_mat)/2
    
    x = cont_mat.sum(axis=1)
    y = cont_mat.sum(axis=0)
    
    z = x.sum() + y.sum()
    
    x /= z
    y /= z
    
    obj_heur_mat += error_mat * y
    
    z = error_mat.sum()
    
    curr_heur_mat += x*z
    
    res_mat = np.dot(curr_heur_mat, obj_heur_mat * p_mat)
    
    rank_mat = res_mat.argsort()
    
    return res_mat, obj_heur_mat, curr_heur_mat, learning_rate

p_count = 10
curr_type_count = 5

# Training Data
#p_mat = np.random.rand(curr_type_count, p_count)#
p_mat = np.arange(p_count * curr_type_count).reshape(curr_type_count, p_count)
exp_mat = np.arange(p_count)
rank_exp_mat = exp_mat.argsort()

print(colored('participant matrix:', 'cyan'))
print(p_mat)
print()

print(colored('expected resultant matrix:', 'cyan'))
print(exp_mat)
print()

# Algorithm
curr_heur_mat = np.ones([1, curr_type_count], dtype=float)
obj_heur_mat = np.ones([1, p_count], dtype=float)

res_mat = np.dot(curr_heur_mat, obj_heur_mat * p_mat)

rank_mat = res_mat.argsort()

print(colored('currency heuristic matrix:', 'green'))
print(curr_heur_mat)
print()

print(colored('object heuristic matrix:', 'green'))
print(obj_heur_mat)
print()

print(colored('resultant matrix:', 'green'))
print(res_mat)
print()

# Learning
# Optimizing for expected matrix
learning_rate = .02 #1/(p_count + curr_type_count) # Max learning rate
min_learning_rate = .001
for i in range(500):
    print(colored('i:', 'red'), i)
    print()
    
    error_mat = exp_mat - res_mat
    print('error mat::')
    print(error_mat)

    #print('calculated learning rate')
    #cal_lr = (learning_rate * (np.absolute(error_mat).sum()/len(error_mat))) / initial_err 
   
    #if cal_lr > learning_rate or math.isnan(cal_lr):
    #    cal_lr = learning_rate

    #print(cal_lr)
    
    #error_mat *= cal_lr
    error_mat *= learning_rate
    learning_rate *= .8
    res_mat, obj_heur_mat, curr_heut_mat, learning_rate = correct(error_mat, p_mat, obj_heur_mat, curr_heur_mat)

    print(colored('currency heuristic matrix:', 'green'))
    print(curr_heur_mat)
    print()
    
    print(colored('object heuristic matrix:', 'green'))
    print(obj_heur_mat)
    print()
    
    print(colored('resultant matrix:', 'green'))
    print(res_mat)
    print()
    #time.sleep(.2)

print(colored('participant matrix:', 'cyan'))
print(p_mat)
print()

print(colored('expected resultant matrix:', 'cyan'))
print(exp_mat)
print()

print(colored('currency heuristic matrix:', 'green'))
print(curr_heur_mat)
print()

print(colored('object heuristic matrix:', 'green'))
print(obj_heur_mat)
print()

print(colored('resultant matrix:', 'green'))
print(res_mat)
print()

error_mat = exp_mat - res_mat 

print(colored('error matrix', 'yellow'))
print(error_mat)
print()
