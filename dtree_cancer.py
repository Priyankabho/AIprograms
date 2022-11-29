import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn import tree
import os
os.chdir("C:\Training\Academy\Statistics (Python)\Cases\Wisconsin")

df = pd.read_csv("BreastCancer.csv")
dum_df = pd.get_dummies(df,drop_first=True)

X = dum_df.iloc[:,1:-1]
y = dum_df.iloc[:,-1]

X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                 random_state=2022,
                                                 test_size=0.3,
                                                 stratify=y)
clf = DecisionTreeClassifier(max_depth=3,random_state=2022)
clf.fit(X_train,y_train)

plt.figure(figsize=(15,10))
tree.plot_tree(clf,feature_names=X_train.columns,
               class_names=['Benign','Malignant'],
               filled=True,fontsize=13) 
plt.show()

y_pred = clf.predict(X_test)
y_pred_prob = clf.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test,y_pred_prob))
########################## Grid Search CV ####################
kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
clf = DecisionTreeClassifier(random_state=2022)
params = {'max_depth':[3,4,None],
          'min_samples_split':[2,10,20],
          'min_samples_leaf':[1,5,10]}
gcv = GridSearchCV(clf, param_grid=params,
                   scoring='roc_auc',cv=kfold)
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)
pd_cv = pd.DataFrame(gcv.cv_results_)

best_model = gcv.best_estimator_
plt.figure(figsize=(25,10))
tree.plot_tree(best_model,feature_names=X.columns,
               class_names=['Benign','Malignant'],
               filled=True,fontsize=13) 
plt.show()