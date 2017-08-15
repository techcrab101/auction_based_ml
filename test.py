import numpy as np
from termcolor import colored

def show_init_cond():
    '''prints the initial conditions'''
    print(colored('participant matrix:', 'cyan'))
    print(p_mat)
    print()

    print(colored('expected resultant matrix:', 'cyan'))
    print(exp_mat)
    print()
    
    #print(colored('expected resultant matrix ranked least to greatest:', 'cyan'))
    #print(rank_exp_mat)
    #print()

    print(colored('currency heuristic matrix:', 'green'))
    print(curr_heur_mat)
    print()
    
    print(colored('object heuristic matrix:', 'green'))
    print(obj_heur_mat)
    print()

    print(colored('resultant matrix:', 'green'))
    print(res_mat)
    print()
    
    #print(colored('resultant matrix ranked least to greatest:', 'green'))
    #print(rank_mat)
    #print()
    pass
    
p_count = 3
curr_type_count = 2

# Training Data
p_mat = np.arange(6).reshape(curr_type_count, p_count)
exp_mat = np.array([4,3,1], dtype=float)
rank_exp_mat = exp_mat.argsort()

# Algorithm
curr_heur_mat = np.ones([1, curr_type_count], dtype=float)
obj_heur_mat = np.ones([1, p_count], dtype=float)

res_mat = np.dot(curr_heur_mat, obj_heur_mat * p_mat)

rank_mat = res_mat.argsort()

show_init_cond()

# Learning
# Optimizing for expected matrix

error_mat = exp_mat - res_mat 

learning_rate = 1
error_mat *= learning_rate

print(colored('error matrix', 'yellow'))
print(error_mat)
print()

cont_mat = p_mat/p_mat.sum(axis=1).reshape(curr_type_count,1)#p_mat/p_mat.sum(axis=0)

print('cont mat')
print(cont_mat)
print()
