import numpy as np
from matplotlib import pyplot as plt 
import csv

with open('/home/krispy/assignment1/P2_data/P2_train.csv') as csvfile:
	data = csv.reader(csvfile, delimiter=',')
	train = np.array(list(data)).astype(float)
#310,3
with open('/home/krispy/assignment1/P2_data/P2_test.csv') as csvfile:
	data = csv.reader(csvfile, delimiter=',')
	test = np.array(list(data)).astype(float)
#90,3

train0 = np.array([train[row,:-1] for row in range(np.size(train,axis=0)) if train[row,-1]==0]).astype(float)
train1 = np.array([train[row,:-1] for row in range(np.size(train,axis=0)) if train[row,-1]==1]).astype(float)
xtrain = np.array(train[:,:-1]).astype(float)

xtest = test[:,:-1]
ytest = test[:,-1]
####################################### Parameters ####################################################

n1 = np.size(train1,axis=0)
n0 = np.size(train0,axis=0)
p0 = n0/float(n0+n1)
p1 = n1/float(n1+n0)

u = np.average(xtrain,axis=0)
u0 = np.average(train0,axis=0)
u1 = np.average(train1,axis=0)


cov = np.abs(1/float(n0+n1)*np.dot(xtrain.T,xtrain) - np.outer(u,u))
eigval,eigvec = np.linalg.eig(cov)
covB = np.diag(np.abs(np.dot(eigval,np.abs(eigvec))))
covA = np.mean(np.abs(np.dot(eigval,np.abs(eigvec))))*np.diag([1,1])
print covA


cov1 = 1/float(n1)*np.dot(train1.T,train1) - np.outer(u1,u1)
cov0 = 1/float(n0)*np.dot(train0.T,train0) - np.outer(u0,u0)

cov1 = covA
cov0 = covA
print cov, cov0, cov1
'''
eigval0,eigvec0 = np.linalg.eig(cov0)
cov0 = np.dot(np.dot(eigvec0.T,cov0),eigvec0)
cov0 = np.dot(cov0,cov0)
cov0[cov0<=1e-5] = 0
cov1 = cov0
'''
##################################### Classifier and Plotting #########################################

def disc0(x):
	return p0/float(np.sqrt((2*np.pi)**2*np.abs(np.linalg.det(cov0))))*np.exp(-0.5*(np.dot(np.dot((x-u0).T,np.linalg.inv(cov0)),(x-u0))))
def disc1(x):
	return p1/float(np.sqrt((2*np.pi)**2*np.abs(np.linalg.det(cov1))))*np.exp(-0.5*(np.dot(np.dot((x-u1).T,np.linalg.inv(cov1)),(x-u1))))

def disc(x):
	temp = 0.5	 + 0.5*np.sign(disc0(x)-disc1(x))
	return 0*temp + 1*(1-temp)  


yhat = np.array([ disc(xtest[i,:]) for i in range(np.size(xtest,axis=0)) ])
conf = np.zeros((2,2))
# confusion matrix
for i in range(yhat.size):
	if ytest[i]==0:
		if yhat[i]==0:
			conf[0,0] += 1;
		else:
			conf[0,1] += 1;	
	if ytest[i]==1:
		if yhat[i]==1:
			conf[1,1] += 1;
		else:
			conf[1,0] += 1;
print conf.astype(int)



#plt.scatter(xtest[:,0],xtest[:,1],c=ytest)
#plt.show()
num = 200
a = np.linspace(-4,4,num)
b = np.linspace(-6,6,num)
X,Y = np.meshgrid(a,b)

def decColor(x,y):
	temp = np.zeros((num,num))
	print x.shape, np.size(x,axis=0)
	for l in range(num):
		for m in range(num):
			p = np.array([x[l,m],y[l,m]])
			#print p
			temp[l,m] = disc(p)
	#print temp
	return temp

boundColorMap = decColor(X,Y)


group = 0
boundary = []
for x in range(num):
	group = boundColorMap[x,0]
	for y in range(num):
		if boundColorMap[x,y]!=group:
			boundary.append([X[x,y],Y[x,y]])
			group = boundColorMap[x,y]	
boundary = np.array(boundary)
plt.figure()



def f(x,y,u,cov):
	temp = np.zeros((num,num))
	print x.shape, np.size(x,axis=0)
	for l in range(num):
		for m in range(num):
			p = np.array([x[l,m],y[l,m]])
			#print p
			temp[l,m] = (1/float(np.sqrt((2*np.pi)**2*np.abs(np.linalg.det(cov))))*np.exp(-0.5*(np.dot(np.dot((p-u).T,np.linalg.inv(cov)),(p-u)))))
		#print tmep[l,m]
	#print temp
	return temp




F1 = f(X,Y,u1,cov1)

plt.contourf(X, Y, F1, 10, alpha=0.5, cmap='Blues')
C = plt.contour(X, Y, F1, 10, colors='black', linewidths=0.1)
plt.clabel(C, inline=1, fontsize=5)


F0 = f(X,Y,u0,cov0)
plt.contourf(X, Y, F0, 10, alpha=0.5, cmap='Reds')
C = plt.contour(X, Y, F0, 10, colors='black',linewidths=0.1)
plt.clabel(C, inline=1, fontsize=5)
clr = []
for x in train[:,2]:
	if x == 0:
		clr.append('red')
	else:
		clr.append('blue')


plt.scatter(train[:,0],train[:,1],alpha=0.5,color=clr,s=1)
#plt.title('Bivariate Gaussian fit using unequal scaled identity matrix covariance')

plt.scatter(boundary[:,0],boundary[:,1],color='black',s=2) 
plt.savefig('boundary.png',dpi=700)
#print X,Y
plt.show()
