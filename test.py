import numpy as np
from termcolor import colored

def correct(exp_mat, res_mat, p_mat, obj_heur_mat, curr_heur_mat):
    error_mat = exp_mat - res_mat 
    
    learning_rate = .2
    error_mat *= learning_rate
    
    print(colored('error matrix', 'yellow'))
    print(error_mat)
    print()
    
    curr_cont_mat = p_mat/p_mat.sum(axis=0)
    
    print(colored('currency cont mat', 'yellow'))
    print(curr_cont_mat)
    print()
    
    obj_cont_mat = p_mat/p_mat.sum(axis=1).reshape(curr_type_count,1)
    
    print(colored('obj cont mat', 'yellow'))
    print(obj_cont_mat)
    print()
    
    cont_mat = ((obj_cont_mat + curr_cont_mat) / 
            (obj_cont_mat.sum() + curr_cont_mat.sum())) #(curr_cont_mat + obj_cont_mat)/2
    
    print(colored('overall contribution matrix', 'yellow'))
    print(cont_mat)
    print()
    
    x = cont_mat.sum(axis=1)
    y = cont_mat.sum(axis=0)
    
    z = x.sum() + y.sum()
    
    x /= z
    y /= z
    print()
    print (x)
    print (y)
    print ()
    
    
    for i in range(len(error_mat[0])):
        error = error_mat[0][i]
    
        print()
        print('error:', error)
        print()
        print('Before error')
        print('curr heur')
        print(curr_heur_mat)
        print('obj heur')
        print(obj_heur_mat)
    
        curr_heur_mat[0][0] += error*x[0]
        curr_heur_mat[0][1] += error*x[1]
        obj_heur_mat[0][i] += error*y[i]
    
        print()
        print('after error')
        print('curr heur')
        print(curr_heur_mat)
        print('obj heur')
        print(obj_heur_mat)
        print()
    
    res_mat = np.dot(curr_heur_mat, obj_heur_mat * p_mat)
    
    rank_mat = res_mat.argsort()
    
    return res_mat, obj_heur_mat, curr_heur_mat

    pass

p_count = 3
curr_type_count = 2

# Training Data
p_mat = np.arange(6).reshape(curr_type_count, p_count)
exp_mat = np.array([4,3,1], dtype=float)
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

# Learning
# Optimizing for expected matrix
for i in range(100):
    res_mat, obj_heur_mat, curr_heut_mat = correct(exp_mat, res_mat, p_mat, obj_heur_mat, curr_heur_mat)

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
