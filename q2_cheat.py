import numpy as np
from matplotlib import pyplot as plt 
import csv
import seaborn as sns
import pandas as pd

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


#################################### Parameters #################################################

n1 = np.size(train1,axis=0)
n0 = np.size(train0,axis=0)

p1 = n1/float(n1+n0) #0.509652509653

u = np.average(xtrain,axis=0)
u0 = np.average(train0,axis=0)
u1 = np.average(train1,axis=0)


cov = 1/float(n0+n1)*np.dot(xtrain.T,xtrain) - np.outer(u,u)
eigval,eigvec = np.linalg.eig(cov)
cov = np.dot(np.dot(eigvec.T,cov),eigvec)


cov1 = 1/float(n1)*np.dot(train1.T,train1) - np.outer(u1,u1)
'''
eigval0,eigvec0 = np.linalg.eig(cov0)
cov0 = np.dot(np.dot(eigvec0.T,cov0),eigvec0)
cov0 = np.dot(cov0,cov0)
cov0[cov0<=1e-5] = 0
cov1 = cov0
'''
####################################### Plotting ##################################################

'''
num = 200
def f(x,y):
	temp = np.zeros((num,num))
	print x.shape, np.size(x,axis=0)
	for l in range(num):
		for m in range(num):
			p = np.array([x[l,m],y[l,m]])
			#print p
			temp[l,m] = (1/float(np.sqrt((2*np.pi)**2*np.abs(np.linalg.det(cov1))))*np.exp(-0.5*(np.dot(np.dot((p-u1).T,np.linalg.inv(cov1)),(p-u1)))))
		#print tmep[l,m]
	#print temp
	return temp

a = np.linspace(-10,10,num)
b = np.linspace(-5,5,num)
X,Y = np.meshgrid(a,b)
F = f(X,Y)

plt.figure()
plt.contourf(X, Y, F, 8, alpha=.25, cmap='jet')
C = plt.contour(X, Y, F, 8, colors='black')
plt.scatter(train[:,0],train[:,1],c=train[:,2])
plt.clabel(C, inline=1, fontsize=10)
#print X,Y
'''

sns.set(style="darkgrid")
data = pd.DataFrame(train, columns=['att1','att2','group'])
#print data.head(n=3)

# Subset the iris dataset by species
group0 = data.query('group == 0.0')
group1 = data.query('group == 1.0')

# Set up the figure
f, ax = plt.subplots(figsize=(16, 16))
ax.set_aspect("equal")

# Draw the two density plots
ax = sns.kdeplot(group0.att1, group0.att2,
                 cmap="Reds", shade=True, shade_lowest=False)
ax = sns.kdeplot(group1.att1, group1.att2,
                 cmap="Blues", shade=True, shade_lowest=False)

# Add labels to the plot
red = sns.color_palette("Reds")[-2]
blue = sns.color_palette("Blues")[-2]
ax.text(2.5, 8.2, "Class 1", size=16, color=blue)
ax.text(3.8, 4.5, "Class 0", size=16, color=red)

plt.savefig('test.png',dpi=500)

plt.show()