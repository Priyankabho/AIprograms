import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
import os
os.chdir("C:\Training\Academy\Statistics (Python)\Cases\Wisconsin")

df = pd.read_csv("BreastCancer.csv")
dum_df = pd.get_dummies(df,drop_first=True)
X = df.iloc[:,1:-1]
y = dum_df.iloc[:,-1]

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=2022,
                                                 test_size=0.3,
                                                 stratify=y)
lr = LogisticRegression()
model_bg = BaggingClassifier(base_estimator=lr,
                             random_state=2022,n_estimators=15,
                             max_samples=X_train.shape[0],
                             max_features=X_train.shape[1])
model_bg.fit(X_train,y_train)
y_pred_prob = model_bg.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test,y_pred_prob))
##############################################################
lr = SVC(probability= True)
model_bg = BaggingClassifier(base_estimator=lr,
                             random_state=2022,n_estimators=15,
                             max_samples=X_train.shape[0],
                             max_features=X_train.shape[1])
model_bg.fit(X_train,y_train)
y_pred_prob = model_bg.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test,y_pred_prob))

##############################################################
model_bg = BaggingClassifier(random_state=2022,n_estimators=15,
                             max_samples=X_train.shape[0],
                             max_features=X_train.shape[1])
model_bg.fit(X_train,y_train)
y_pred_prob = model_bg.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test,y_pred_prob))
