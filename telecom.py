Name:- Hari singh r
batch id :- DSWDMCOD 25082022 B

#buissness objectives
Maximize-revenue
Minimize-customer churn

#buissness constrain
to decrease customer churn will need efficent hardware service

import pandas as pd
import matplotlib.pylab as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

df=pd.read_excel("D:/assignments of data science/08 k menas/Telco_customer_churn.xlsx")

df.info()
df.head()
df.duplicated()
df.describe()
df.isnull().sum()

labelencoder=LabelEncoder()


x = df.iloc[:,[0,3,2,6,7,9,10,11,13,14,15,16,17,18,19,20,21,22,23]]  #columns which are having objects
y = df.iloc[:,[1,4,5,8,12,24,25,26,27,28,29]]  #columns which are having the numerical data


x = df.select_dtypes(include="object").columns
print(x)

z=df[x]=df[x].apply(labelencoder.fit_transform)

for i in df.columns: #histogram(Univariate)
    plt.hist(df[i])
    plt.xlabel(i)
    plt.show()
    
plt.scatter(df["Tenure in Months"], df["Total Revenue"]) #scatterplot(Bivariate)
plt.xlabel("Tenure in months")
plt.ylabel("Total revenue")

df1=pd.concat([z,y],axis=1)

df1.isnull().any()

def norm(i):
    x=(i-i.min())/(i.max()-i.min())
    return(x)

df_norm=norm(df1.iloc[:,:])
df_norm.describe()

df_norm.isnull().sum()

df_norm=df_norm.drop(["Quarter","Count"],axis=1)

TWSS=[]
k=list(range(1,16))

for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)

TWSS

plt.plot(k,TWSS,'ro-');plt.xlabel("No of Cluster");plt.ylabel("Total within ss")

model=KMeans(n_clusters=3)
model.fit(df_norm)

model.labels_
mb=pd.Series(model.labels_)
df['clust']=mb
mb

df.head()
df_norm.head()

df = df.iloc[:,:]
df.head()


final = df.iloc[:, :].groupby(df.clust).mean()
final

Cluster 0 = These are the customers that are frequent users The revenue earned through these is also the best. Hence, these are the customers that are least likely to churn.
Cluster 2 = These are the customers may or may not churn that frequently.
cluster 1 = These are the customers that are least frequent.these are the ones that churn chances is more
















































