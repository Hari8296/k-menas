Name :- Hari singh r
batch id :- DSWDMCOD 25082022 B
1
Business problems 
maximize the security with camers
minimize the crime rate in city

2
buissness constrain
giving awerness of crime 

import pandas as pd
import matplotlib.pylab as plt
from sklearn.cluster import KMeans

df=pd.read_csv("D:/assignments of data science/08 k menas/crime_data.csv")
df

df.head()
df.describe()
df.duplicated().sum()
df.isnull().sum()

def norm(i):
    x=(i-i.min())/(i.max()-i.min())
    return (x)

df_norm=norm(df.iloc[:,1:])
df_norm.describe()

df.mean()
df.median()
list(df.mode())
df.skew()
df.kurt()
df.var()

for i in df.columns:
    plt.hist(df[i])
    plt.xlabel(i)
    plt.show()


for i in ['Murder','Assault','Rape']:
            plt.scatter(df1["UrbanPop"],df[i])
            plt.xlabel('UrbanPop')
            plt.ylabel(i)
            plt.show()

TWSS=[]

k=list(range(2,9))

for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
    
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

model=KMeans(n_clusters=3)
model.fit(df_norm)    

model.labels_
mb=pd.Series(model.labels_)    
df['clust']=mb    

df.head()    
df_norm.head()    

df=df.iloc[:,:]    
df.head()    

df.groupby('clust').mean()   

in the citizen of clust 0:-it has moderate crime rate in everthing which less safer then compare to others 
in the citizen of clust 1:-it has high crime rate when compare to clust 0 citizen which make not safe to stay 
in the citizen pf clust 2:-it has less crime rate when compare to clust 0 and 1 which make more safer to stay in clust 2 city for better life 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    