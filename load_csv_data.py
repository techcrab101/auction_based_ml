import csv
import argparse
import pickle

parser = argparse.ArgumentParser()

parser.add_argument('-p', '--path', required=True, help='Path to the data set as a csv')
parser.add_argument('-s', '--save', required=True, help='Path to save the newly formated dataset')

args = vars(parser.parse_args())

data_path = args['path']
save_path = args['save']

data = []

label_dict = {}

with open(data_path, 'r', newline='') as csvfile:
    i = 0
    reader = csv.reader(csvfile)
    for row in reader:
        if len(row) == 0:
            continue

        try:
            label_dict[row[-1]]
        except KeyError:
            label_dict[row[-1]] = i
            i += 1
        
        label = label_dict[row[-1]]

        row = [float(i) for i in row[:-1]]
        row.append(label)
        print(row)
        data.append(row)

File = open(args['save'] + 'data.pkl', 'wb')

pickle.dump(data, File)

File.close()
