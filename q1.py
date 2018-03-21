import pandas as pd 
from matplotlib import pyplot as plt 
import seaborn as sns 

df = pd.read_csv('Pokemon.csv', index_col=0)
print(df.head())

sns.lmplot(x="Attack", y="Defense",data = df, fit_reg = False, hue='Stage')
plt.xlim(0,None)
plt.ylim(0,None)


plt.show()