import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV,cross_val_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv("Bankruptcy.csv")
df.drop('NO',axis=1,inplace=True)
X=df.iloc[:,1:]
y=df.iloc[:,0]

#----------------scaler=MinMaxScaler()-------------------

knn=KNeighborsClassifier()
params={'n_neighbors':[1,3,5,7,9,11,13]}
kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
gcv=GridSearchCV(knn,param_grid=params,cv=kfold,scoring='roc_auc')

knn_pipe=Pipeline([('scaler',scaler)  ,('grid_search',gcv)])
knn_pipe.fit(X,y) 
print(gcv.best_params_)
print(gcv.best_score_)

#----------------StandardScaler-----------------------

scaler1=StandardScaler()
knn=KNeighborsClassifier()
params={'n_neighbors':[1,3,5,7,9,11,13]}
kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
gcv=GridSearchCV(knn,param_grid=params,cv=kfold,scoring='roc_auc')

knn_pipe1=Pipeline([('scaler',scaler1),('grid_search',gcv)])
knn_pipe1.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)
#--------------------SGDClassifier---------------
scaler2=MinMaxScaler()
sgd=SGDClassifier()
params={'eta0':np.linspace(0.0001,0.7,10),
       'learning_rate':['constant','optimal','invscaling','adaptive']}
kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
gcv2=GridSearchCV(sgd,param_grid=params,cv=kfold,scoring='roc_auc')

knn_pipe=Pipeline([('scaler',scaler2)  ,('grid_search',gcv2)])
knn_pipe.fit(X,y)

print(gcv2.best_params_)
print(gcv2.best_score_)

#--------------LogisticRegression-----------------

lr=LogisticRegression()
scaler_lr=MinMaxScaler()
lr_pipe=Pipeline([('scaler',scaler_lr),('lr',lr)])
kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
results=cross_val_score(lr_pipe,X,y,cv=kfold,scoring='roc_auc')
print(results.mean())





