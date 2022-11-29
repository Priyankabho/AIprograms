# Perform the necessary imports
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

milk = pd.read_csv("milk.csv", index_col=0)

scaler = StandardScaler()
milkscaled=scaler.fit_transform(milk)

wss = []
for i in np.arange(2,10):
    km = KMeans(n_clusters=i,random_state=2022)
    km.fit(milkscaled)
    wss.append(km.inertia_)
 
plt.plot(np.arange(2,10),wss)
plt.xlabel("No. of Clusters")
plt.ylabel("WSS")
plt.show()
    
km = KMeans(n_clusters=4,random_state=2022)
km.fit(milkscaled)    
    
labels = km.predict(milkscaled)

print(labels)
lbl_df = pd.Series(labels,name="label",index=milk.index).to_frame()

clustered = pd.concat([milk,lbl_df],axis=1)
clustered.groupby("label").mean()
