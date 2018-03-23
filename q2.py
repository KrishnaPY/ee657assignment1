import numpy as np
from matplotlib import pyplot as plt 
import seaborn as sns 
import csv

with open('/home/krispy/assignment1/P2_data/P2_train.csv') as csvfile:
	data = csv.reader(csvfile, delimiter=',')
	train = np.array(list(data)).astype(float)


with open('/home/krispy/assignment1/P2_data/P2_test.csv') as csvfile:
	data = csv.reader(csvfile, delimiter=',')
	test = np.array(list(data)).astype(float)

print train.shape, test.shape