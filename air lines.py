Name :- Hari singh r
Batch Id :- DSWDMCOD 25082022 B

1
Business problems 
1.1 what is the Business odjectives 
Ans:- Maximize the profit
      minimize the cost of airlines to compare other airlines 
      
1.2 What are the constraints      
Ans:- provide offers on air tickicts to compare other airlines 


import pandas as pd
import matplotlib.pylab as plb
from sklearn.cluster import KMeans

df=pd.read_excel("D:/assignments of data science/08 k menas/EastWestAirlines.xlsx",sheet_name=1)
df
df.info()
df.duplicated().sum()
df.isna().sum()
df.describe()

df1=df.drop(["ID#"],axis=1)
df1

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
    
lis=[]
for i in df1.columns:
    for j in df1.columns:
        if(i!=j):
            plt.scatter(df1[i],df1[j])
            plt.xlabel(i)
            plt.ylabel(j)
            plt.show()


def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

df_norm = norm_func(df1.iloc[:, :])

TWSS=[]

k=list(range(2,11))

for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
       
plb.plot(k,TWSS,'ro-');plb.xlabel("NO of cluster");plb.ylabel("total with in ss") 

model = KMeans(n_clusters = 3) 
model.fit(df_norm) 
model.labels_ 
mb = pd.Series(model.labels_) 

df['clust']=mb

df1.head()  
df_norm.head()

df1=df1.iloc[:,:]
df1.head()
df.iloc[:,:].groupby(df.clust).mean()















































