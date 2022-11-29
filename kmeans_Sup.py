import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score,roc_auc_score
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("C:\Training\Academy\Statistics (Python)\Cases\Bankruptcy\Bankruptcy.csv")
dum_df = pd.get_dummies(df, drop_first=True)

X = df.drop(['D','NO'],axis=1)
y = df['D']

# Import the necessary modules
from sklearn.model_selection import train_test_split 

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                    random_state=2022,
                                                    stratify=y)
scaler = StandardScaler()
X_trn_scl = scaler.fit_transform(X_train)
X_tst_scl = scaler.transform(X_test)

clustNos = [2,3,4,5,6,7,8,9,10]
silhouettes = []

for i in clustNos :
    model = KMeans(n_clusters=i,random_state=2022)
    model.fit(X_trn_scl)
    labels = model.predict(X_trn_scl)
    sil_score = silhouette_score(X_trn_scl,labels)
    silhouettes.append(sil_score)
    
# Import pyplot
import matplotlib.pyplot as plt

plt.plot(clustNos, silhouettes, '-o')
plt.title("Scree Plot")
plt.xlabel('Number of clusters, k')
plt.ylabel('Silhouette Score')
plt.xticks(clustNos)
plt.show()

model = KMeans(n_clusters=6,random_state=2022)
model.fit(X_trn_scl)
labels = model.predict(X_trn_scl)

lbl_df = pd.Series(labels,name="cluster",dtype='object',index=X_train.index).to_frame()
X_trn_clust = pd.concat([X_train,lbl_df],axis=1)

clf1 = RandomForestClassifier(random_state=2022)
clf1.fit(X_train,y_train)
y_pred_prob = clf1.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test,y_pred_prob))
###################################################
dum_X_trn_clust = pd.get_dummies(X_trn_clust)
dum_X_trn_clust.drop('cluster_3',axis=1,inplace=True)
clf2 = RandomForestClassifier(random_state=2022)
clf2.fit(dum_X_trn_clust,y_train)

labels = model.predict(X_tst_scl)

lbl_df = pd.Series(labels,name="cluster",dtype='object',index=X_test.index).to_frame()
X_tst_clust = pd.concat([X_test,lbl_df],axis=1)
dum_X_tst_clust = pd.get_dummies(X_tst_clust)

y_pred_prob = clf2.predict_proba(dum_X_tst_clust)[:,1]
print(roc_auc_score(y_test,y_pred_prob))
