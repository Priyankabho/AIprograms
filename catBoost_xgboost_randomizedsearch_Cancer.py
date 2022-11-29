import pandas as pd
# Import the necessary modules
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
import numpy as np

df = pd.read_csv("BreastCancer.csv")

dum_df = pd.get_dummies(df, drop_first=True)

X = dum_df.iloc[:,1:10]
y = dum_df.iloc[:,10]

################# sklearn API ######################

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                    random_state=2022,
                                                    stratify=y)

clf = CatBoostClassifier(random_state=2022)
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test,y_pred))

##################Tunning XGB using Grid Search CV#####################

lr_range = [0.001,0.01,0.2,0.5,0.6,0.89]
n_est_range = [30,70,100,120,150]
depth_range = [3,4,5,6,7,8,9]

parameters = dict(learning_rate=lr_range,
                  n_estimators=n_est_range,
                  max_depth=depth_range)

from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=5, random_state=2022,shuffle=True)

from sklearn.model_selection import GridSearchCV
clf = CatBoostClassifier(random_state=2022)
cv = GridSearchCV(clf, param_grid=parameters,cv=kfold,scoring='roc_auc')

cv.fit(X,y,verbose=False)
df_cv = pd.DataFrame(cv.cv_results_)
print(cv.best_params_)

print(cv.best_score_)


##################Tunning using Randomized Search CV ############
lr_range = [0.001,0.01,0.2,0.5,0.6,0.89]
n_est_range = [30,70,100,120,150]
depth_range = [3,4,5,6,7,8,9]

parameters = dict(learning_rate=lr_range,
                  n_estimators=n_est_range,
                  max_depth=depth_range)


from sklearn.model_selection import RandomizedSearchCV
clf = XGBClassifier(random_state=2021,use_label_encoder=False)
rcv = RandomizedSearchCV(clf, param_distributions=parameters,
                  cv=kfold,scoring='roc_auc',n_iter=15,random_state=2021)

rcv.fit(X,y)
df_rcv = pd.DataFrame(rcv.cv_results_)
print(rcv.best_params_)

print(rcv.best_score_)




