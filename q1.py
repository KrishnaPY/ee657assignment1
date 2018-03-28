import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt 
import seaborn as sns 
import csv


################################## PARAMETER CALCULATION ############################################

with open('/home/krispy/assignment1/P1_data/P1_data_train.csv', 'rb') as csvfile:
	data  = csv.reader(csvfile, delimiter=',')
	xtrain = list(data)
	xtrain = np.array(xtrain).astype(int)
#(777,64)

with open('/home/krispy/assignment1/P1_data/P1_labels_train.csv', 'rb') as csvfile:
	data  = csv.reader(csvfile, delimiter=',')
	ytrain = list(data)
	ytrain = np.array(ytrain).astype(int)
#(777,1)

'''
for i in range(10):
	number = xtrain[i,:].reshape(8,8)
	ax=sns.heatmap(number,cmap="Blues")
	ax.invert_yaxis()
	plt.show() 
'''
train5 = np.array([xtrain[row,:] for row in range(np.size(xtrain,0)) if ytrain[row]==5]) #396
train6 = np.array([xtrain[row,:] for row in range(np.size(xtrain,0)) if ytrain[row]==6]) #381																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																	n5 = np.size(train5,axis=0)

n5 = np.size(train5,axis=0)
n6 = np.size(train6,axis=0)

p5 = n5/float(n5+n6) #0.509652509653

u = np.average(xtrain,axis=0)
u5 = np.average(train5,axis=0)
u6 = np.average(train6,axis=0)

##### class wise covariance
cov5 = 1/float(n5)*np.dot(train5.T,train5) - np.outer(u5,u5) 
cov6 = 1/float(n6)*np.dot(train6.T,train6) - np.outer(u6,u6)
cov5.tofile('cov5.csv',sep=',',format='%.5f')
cov6.tofile('cov6.csv',sep=',',format='%.5f')
#### overall covariance
cov = 1/float(n5+n6)*(np.dot(xtrain.T,xtrain) - np.outer(u,u))

del xtrain,train5,train6,ytrain

################################## TESTING #################################################

with open('/home/krispy/assignment1/P1_data/P1_data_test.csv', 'rb') as csvfile:
	data  = csv.reader(csvfile, delimiter=',')
	xtest = list(data)
	xtest = np.array(xtest).astype(int)

with open('/home/krispy/assignment1/P1_data/P1_labels_test.csv', 'rb') as csvfile:
	data  = csv.reader(csvfile, delimiter=',')
	ytest = list(data)
	ytest = np.array(ytest).astype(int)

# negative log liklehood

def DisrSepCov(i):
	x = xtest[i,:]
	d6 = np.log(1-p5) - np.dot(np.dot((x-u6).T,np.linalg.inv(cov6)),(x-u6)) - 0.5*np.log(np.abs(np.linalg.det(cov6)))
	d5 = np.log(p5) - np.dot(np.dot((x-u5).T,np.linalg.inv(cov5)),(x-u5)) - 0.5*np.log(np.abs(np.linalg.det(cov5)))
	return int(0.5+np.sign(d5-d6))

def DisrSameCov(i):
	x = xtest[i,:]
	d6 = np.log(1-p5) - np.dot(np.dot((x-u6).T,np.linalg.inv(cov)),(x-u6))
	d5 = np.log(p5) - np.dot(np.dot((x-u5).T,np.linalg.inv(cov)),(x-u5)) 
	return int(0.5+np.sign(d5-d6))


d = lambda x : x*5 + (1-x)*6

yhat = np.array([ d(DisrSepCov(i)) for i in range(np.size(xtest,axis=0)) ])
conf = np.zeros((2,2))
# confusion matrix
for i in range(yhat.size):
	if ytest[i]==5:
		if yhat[i]==5:
			conf[0,0] += 1;
		else:
			conf[0,1] += 1;	
	if ytest[i]==6:
		if yhat[i]==6:
			conf[1,1] += 1;
		else:
			conf[1,0] += 1;
print conf.astype(int)

yhat = np.array([ d(DisrSameCov(i)) for i in range(np.size(xtest,axis=0)) ])
conf = np.zeros((2,2))
# confusion matrix
for i in range(yhat.size):
	if ytest[i]==5:
		if yhat[i]==5:
			conf[0,0] += 1;
		else:
			conf[0,1] += 1;	
	if ytest[i]==6:
		if yhat[i]==6:
			conf[1,1] += 1;
		else:
			conf[1,0] += 1;
print conf.astype(int)