import numpy as np
from termcolor import colored
import time
import math
import pickle
import matplotlib.pyplot as plt
import argparse

def load_data(data_file):
    data_file = open(data_file, 'rb')
    
    raw_data = pickle.load(data_file)
    raw_data = np.array(raw_data)
    
    x = raw_data[:,:len(raw_data[0])-1]
    y = raw_data[:,-1:]
    
    new_y = []
    for i in y:
        new_y.append(i[0])
    y = np.array(new_y)
    
    x = np.rot90(x)
    return x, y

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
    
    def correct(self, error_mat, obj_learning_rate, curr_learning_rate, p_mat, display=True):
        """Corrects for the error matrix, modifying both heuristic vectors, and returns the resultant matrix
        
        Keyword Arguments:
        error_mat -- The difference between the expected and resultant vectors
        obj_learning_rate -- The multiplier of the object error correction, to prevent over correcting
        curr_learning_rate -- The multiplier of the currency error correction, to prevent over correcting
        p_mat -- The input matrix, used to find contribution of error
        """
        curr_cont_mat = p_mat/p_mat.sum(axis=0)
    
        obj_cont_mat = p_mat/p_mat.sum(axis=1).reshape(curr_type_count,1)

        cont_mat = ((obj_cont_mat + curr_cont_mat) / (obj_cont_mat.sum() + curr_cont_mat.sum()))

        if display:
            print('currency cont matrix')
            print(curr_cont_mat)
            print()
            print('obj cont matrix')
            print(obj_cont_mat)
            print()
            print('overall contribution matrix')
            print(cont_mat)
            print()
            
        x = cont_mat.sum(axis=1)
        y = cont_mat.sum(axis=0)
     
        z = x.sum() + y.sum()
        
        x /= z
        y /= z
       
        if display:
            print('x and y')
            print(x)
            print(y)
            print()
    
        self.obj_heur_mat += (error_mat * obj_learning_rate)* y
       
        self.curr_heur_mat += (error_mat * curr_learning_rate).sum() * x # x * j
        
        res_mat = np.dot(self.curr_heur_mat, self.obj_heur_mat * p_mat)
        
        rank_mat = res_mat.argsort()
        
        return res_mat

    def train(self, x, y, correcting_range, obj_learning_rate=0.1, curr_learning_rate=0.1, display=True):
        """Adjusts the curr_heur and obj_heur based on x,y training data

        Keyword Arguments:
        x -- The input data. Should be a single 2D matrix (a population)
        y -- The output data. Should be a single vector
        correcting_range -- The number of times the algorithm runs to correct for a new error
        obj_learning_rate -- The multiplier for the obj error correction (default=0.1)
        curr_learning_rate -- The multiplier for the curr error correction (default=0.1)
        display -- Boolean on whether or not the training process should be displayed. (default=True)
        """
        
        self.x = x
        self.y = y

        #for i in range(len(self.x)):
        p_mat = self.x
        exp_mat = self.y
        
        res_mat = np.dot(self.curr_heur_mat, self.obj_heur_mat * p_mat)

        for j in range(correcting_range):
            if display:
                print(colored('j:', 'red'), j)
                print()
        
            error_mat = (exp_mat - res_mat)
            if display:
                print(colored('error mat', 'yellow'))
                print(error_mat)
                print()
                print(colored('object learning rate', 'yellow'))
                print(obj_learning_rate)
                print()
                print(colored('curr learning rate', 'yellow'))
                print(curr_learning_rate)
                print()
        
            res_mat = self.correct(error_mat, obj_learning_rate , curr_learning_rate, p_mat, display)
        
            if display:
                print(colored('participant matrix:', 'cyan'))
                print(p_mat)
                print()
        
                print(colored('currency heuristic matrix:', 'green'))
                print(self.curr_heur_mat)
                print()
                
                print(colored('object heuristic matrix:', 'green'))
                print(self.obj_heur_mat)
                print()
                
                print(colored('expected resultant matrix:', 'cyan'))
                print(exp_mat)
                print()

                print(colored('resultant matrix:', 'green'))
                print(res_mat)
                print()

    def find_nearest(self, value):
        idx = (np.abs(self.y-value)).argmin()
        return self.y[idx]

    def get_obj_multiplier(self, category):
        indices = [i for i, x in enumerate(self.y) if x == category]

        rates = []

        for i in indices:
            rates.append(self.obj_heur_mat[0][i])
        # Average Method
        return sum(rates) / len(rates)

    def predict(self, x):
        """Predicts the output based on current heuristics

        input x is a vector of features
        obj_heur_mat is applied only to a 2d matrix
        """

        curr_confidence = 0
        curr_res_mat = []

        categories = list(np.unique(self.y))

        for i in categories:
            rate = self.get_obj_multiplier(i)
            res_mat = np.dot(self.curr_heur_mat, rate * x)
            
            label = self.find_nearest(res_mat[0])
    
            projected_error = abs(label - res_mat[0])
    
            if label == 0:
                projected_confidence = 1 - projected_error
            else:
                projected_confidence = (label-projected_error) / label

            res_mat = [res_mat[0], label, projected_error, projected_confidence]
            if projected_confidence > curr_confidence:
                curr_error = projected_confidence
                curr_res_mat = res_mat
    
        return curr_res_mat
    
    def plot_price_diff(self, category):
        indices = [i for i, x in enumerate(self.y) if x == category]
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

parser = argparse.ArgumentParser(description='TODO: Add this here')

parser.add_argument('-d', '--data_set', required=True,
        help='Path to training data')

parser.add_argument('-t', '--test_data', required=False, 
        help='Path to test data set')

args = vars(parser.parse_args())

x, y = load_data(args['data_set'])


p_count = x.shape[1] # Number of data samples 
curr_type_count = x.shape[0] # Number of features per data sample

_clf = clf(p_count, curr_type_count)

# Learning
# Optimizing for expected matrix
obj_learning_rate = 1
curr_learning_rate = .001

_clf.train(x, y, 100, obj_learning_rate, curr_learning_rate, False)

_clf.plot_price_diff(0)
_clf.plot_price_diff(1)
_clf.plot_price_diff(2)

if (args['test_data'] is not None):
    test_x, test_y = load_data(args['test_data'])
else:
    quit()

total = 0
correct = 0
incorrect = 0
correct_confidence = []
incorrect_confidence = []
incorrect_actual_confidence = []

for i, val in enumerate(test_x.T):
    print(val)
    expected = test_y[i]
    print(expected)
    prediction = _clf.predict(val)
    print(colored(prediction,'cyan'))

    if expected == prediction[1]:
        correct += 1
        correct_confidence.append(prediction[3])
    else:
        incorrect += 1

        actual_error = abs(expected - prediction[0])

        actual_confidence = (expected-actual_error) / expected

        incorrect_confidence.append(prediction[3])
        incorrect_actual_confidence.append(actual_confidence)
        print(colored('ERROR', 'red'))
    print()

    total += 1

percent_correct = correct/total * 100
percent_incorrect = incorrect/total * 100
avg_correct_confidence = sum(correct_confidence)/len(correct_confidence)
avg_incorrect_confidence = sum(incorrect_confidence)/len(incorrect_confidence)
avg_incorrect_percent_error = sum(incorrect_actual_confidence)/len(incorrect_actual_confidence)
avg_confidence = sum(incorrect_confidence + correct_confidence)/len(incorrect_confidence + correct_confidence)


print('Total tested:', total)
print('Total correct:', correct)
print('Total incorrect:', incorrect)
print()

print('Percent correct:', percent_correct)
print('Percent incorrect:', percent_incorrect)
print('Average confidence:', avg_confidence)
print('Avg correct confidence:', avg_correct_confidence)
print('Avg incorrect confidence:', avg_incorrect_confidence)
print('Avg actual incorrect percent error:', avg_incorrect_percent_error)

