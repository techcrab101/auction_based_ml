import numpy as np
from termcolor import colored
import time
import math
import pickle
import matplotlib.pyplot as plt

class clf():
    def __init__(self, p_count, curr_type_count):
        """Initializes the classifier with the right sizes

        Keyword Arguments: 
        p_count -- The number of individuals of a population. The number of columns of the input matrix
        curr_type_count -- The number of features of an individual. The number of rows of the input matrix
        """
        # TODO: Make the p_count scalable, I think?
        self.curr_heur_mat = np.ones([1, curr_type_count], dtype=float)
        self.obj_heur_mat = np.ones([1, p_count], dtype=float)
        self.x = None
        self.y = None
        pass
    
    def correct(self, error_mat, obj_learning_rate, curr_learning_rate, p_mat):
        """Corrects for the error matrix, modifying both heuristic vectors, and returns the resultant matrix
        
        Keyword Arguments:
        error_mat -- The difference between the expected and resultant vectors
        obj_learning_rate -- The multiplier of the object error correction, to prevent over correcting
        curr_learning_rate -- The multiplier of the currency error correction, to prevent over correcting
        p_mat -- The input matrix, used to find contribution of error
        """
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
        """Adjusts the curr_heur and obj_heur based on x,y training data

        Keyword Arguments:
        x -- The input data. Should be a list of 2D matrices (generations) or a single 2D matrix (a population)
        y -- The output data. Should be a list of vectors or a single vector
        correcting_range -- The number of times the algorithm runs to correct for a new error
        obj_learning_rate -- The multiplier for the obj error correction (default=0.1)
        curr_learning_rate -- The multiplier for the curr error correction (default=0.1)
        """
        
        self.x = x
        self.y = y

        for i in range(len(self.x)):
            p_mat = self.x[i]
            exp_mat = self.y[i]
        
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
        """Predicts the output based on current heuristics

        input x can either be a 2d matrix or a vector
        obj_heur_mat is applied only to a 2d matrix
        """
        try:
            x = self.obj_heur_mat * x
        except:
            pass
        res_mat = np.dot(self.curr_heur_mat, x)
        
        return res_mat
    
    def plot_price_diff(self, category):
        indices = [i for i, x in enumerate(self.y[0]) if x == category]
        print(indices)

        plt_x = []
        plt_y = []

        j = 0;
        for i in indices:
            plt_x.append(j)
            j += 1
            plt_y.append(self.obj_heur_mat[0][i])
            
        plt.plot(plt_x, plt_y)
        plt.ylabel('The object heur. The price of the individual categories')
        plt.xlabel('The object amount. Or time')
        plt.show()


p_count = 150 
curr_type_count = 4

_clf = clf(p_count, curr_type_count)

# Training Data
p_mat = np.random.rand(curr_type_count, p_count)#
p_mat = np.arange(p_count * curr_type_count).reshape(curr_type_count, p_count)
exp_mat = np.arange(p_count)#np.random.rand(1,p_count)#np.arange(p_count)
rank_exp_mat = exp_mat.argsort()

data_file = open('training_data/data.pkl', 'rb')

raw_data = pickle.load(data_file)
raw_data = np.array(raw_data)

x = raw_data[:,:len(raw_data[0])-1]
y = raw_data[:,-1:]

input_x = [p_mat]
input_y = [exp_mat]


# print(input_x)
# print(input_y)
# print()
# print()


new_y = []
for i in y:
    new_y.append(i[0])
y = [np.array(new_y)]

x = [np.rot90(x)]

print(x)
print(y)

print()
print()

print(len(x))
print(len(x[0]))

# Learning
# Optimizing for expected matrix
obj_learning_rate = .001#1/(p_count * curr_type_count)#.02 #1/(p_count + curr_type_count) # Max learning rate
curr_learning_rate = .001#obj_learning_rate

_clf.train(x, y, 100, obj_learning_rate, curr_learning_rate)

_clf.plot_price_diff(0)
_clf.plot_price_diff(1)
_clf.plot_price_diff(2)
