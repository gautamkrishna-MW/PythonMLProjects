
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot

df = pd.read_csv("income.csv")

# Plotting the data
pyplot.scatter(df['Age'],df['Income($)'])
pyplot.show(block=False)
pyplot.pause(3)
pyplot.close()

# K-means clustering
kmObj = KMeans(n_clusters=3)
predOut = kmObj.fit_predict(df[['Age','Income($)']])
df['cluster'] = predOut

df0 = df[df['cluster']==0]
df1 = df[df['cluster']==1]
df2 = df[df['cluster']==2]

'''
Clusters change and are not classified correctly since 
Y-axis steps are huge compared to x-axis resulting in 
scaling issues.
'''
pyplot.scatter(df0['Age'],df0['Income($)'],color='green')
pyplot.scatter(df1['Age'],df1['Income($)'],color='red')
pyplot.scatter(df2['Age'],df2['Income($)'],color='blue')
pyplot.show(block=False)
pyplot.pause(3)
pyplot.close()

# Using min-max scaler to scale the inputs before clustering.
scalerObj = MinMaxScaler()
scalerObj.fit(df[['Income($)']])
df['Income($)'] = scalerObj.transform(df[['Income($)']])
scalerObj.fit(df[['Age']])
df['Age'] = scalerObj.transform(df[['Age']])

# K-means clustering after scaling
kmObj = KMeans(n_clusters=3)
predOut = kmObj.fit_predict(df[['Age','Income($)']])
df['cluster'] = predOut

df0 = df[df['cluster']==0]
df1 = df[df['cluster']==1]
df2 = df[df['cluster']==2]

pyplot.scatter(df0['Age'],df0['Income($)'],color='green')
pyplot.scatter(df1['Age'],df1['Income($)'],color='red')
pyplot.scatter(df2['Age'],df2['Income($)'],color='blue')
pyplot.scatter(kmObj.cluster_centers_[:,0],kmObj.cluster_centers_[:,1],marker='*',color='black')
pyplot.show(block=False)
pyplot.pause(3)
pyplot.close()