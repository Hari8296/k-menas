Name:- Hari singh r
Batch id :-DSWDMCOD 25082022 B

#Business objectives
maximize the profit
minimize the claims 

#buissness constrain
better competetion on offer given by other companies

import pandas as pd
import matplotlib.pylab as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv("D:/assignments of data science/08 k menas/AutoInsurance.csv")

df.info()
df.head()
df.duplicated()
df.describe()
df.isnull().sum()
df.dtypes

df1=df.drop(["Customer","State"],axis=1)
df1.dtypes

labelencoder=LabelEncoder()

x = df1.iloc[:,[1,2,3,5,4,6,8,9,15,16,17,18,20,21]]  #columns which are having objects
y = df1.iloc[:,[7,10,11,12,13,14,19]]     #columns which are having the numerical data

x = df1.select_dtypes(include="object").columns
print(x)

z=df1[x]=df1[x].apply(labelencoder.fit_transform)

for i in df1.columns:
    plt.hist(df1[i])
    plt.xlabel(i)
    plt.show()

plt.scatter(df['Customer Lifetime Value'],df['Total Claim Amount'])
plt.xlabel('Customer Lifetime Value')
plt.ylabel('Total Claim Amount')   

df2=pd.concat([z,y],axis=1)

df1.isna().sum()
df1.info()

def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

df_norm= norm_func(df2.iloc[:,:])
df_norm.describe()
df_norm.isnull().any()

TWSS=[]
k = list(range(2,10))

for i in k:
    kmeans=KMeans(n_clusters=1)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)

plt.plot(k,TWSS,'ro-');plt.xlabel("no of cluster");plt.ylabel("total within ss")

model=KMeans(n_clusters=3)
model.fit(df_norm)
model.labels_

mb=pd.Series(model.labels_)
df['clust']=mb

df.head()
df_norm.head()

df=df.iloc[:,:]
df.head()

final = df.iloc[:,:].groupby(df.clust).mean()
final


cluster 0 = having more CLV and more claimed amount people
cluster 1 = having less CLV ,claimed amount to people 
cluster 2 = having in between CLV then the claimed amount to people compared to cluster 0 and 1







































