import auction_alg as au
import argparse
import pickle
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--training_data', required=True, help='Path to the training data')

args = vars(parser.parse_args())

data_file = open(args['training_data'], 'rb')


raw_data = pickle.load(data_file)

raw_data = np.array(raw_data)

# Input data
x = raw_data[:,0:len(raw_data[0])-1]

# Labels / output
y = raw_data[:,-1:]

weights = np.random.rand(1, len(x[0]))

test = au.auction_alg(x, y)

test.perform_auction_process()
