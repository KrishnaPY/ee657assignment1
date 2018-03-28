import numpy as np 
import matplotlib.pyplot as plt  
import csv
import seaborn as sns
import pandas as pd
import matplotlib

with open('/home/krispy/assignment1/Problem_3/Wage_dataset.csv') as csvfile:
	data = np.array(list(csv.reader(csvfile, delimiter = ','))).astype(float)

with open('/home/krispy/assignment1/Problem_3/Wage_original.csv') as csvfile:
	header = csv.reader(csvfile,delimiter=',')
	for row in header:
		header = row
		break
### NO FRICKIN GENDER IN THE DATASET--- REMOVED####
n = np.size(data,axis=0)

wage = len(header)-3
logwage = len(header)-4
year = 1
age = 2
education = 5

xYear = data[:,year]
xAge = data[:,age]
xEducation = data[:,education]
xWage = data[:,wage]
xLogwage = data[:,logwage]

def predict(x,params):
	return np.matmul(x,params)


cmap = matplotlib.cm.get_cmap('Reds')

nmaxAge = 3
plt.figure()
for nAge in range(nmaxAge):
	nAge +=1
	polyAge = np.ndarray((n,nAge))
	for i in range(nAge):
		polyAge[:,i] = xAge**(i+1)


	cAge = np.matmul(np.linalg.inv(np.dot(polyAge.T,polyAge)),np.dot(polyAge.T,xWage))
	print cAge.shape

	header = ['age','year','education','wage','logwage']
	df = pd.DataFrame(np.array([xAge,xYear,xEducation,xWage,xLogwage]).T, columns=header)
	plt.scatter(xAge,predict(polyAge,cAge),c=cmap(1/float(nmaxAge)*nAge))


plt.figure()
sns.set_style('whitegrid')
sns.violinplot(x='age', y='wage', data=df)

plt.show()



#plt.plot(range(1000),)