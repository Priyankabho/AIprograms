import pandas as pd
# Import the necessary modules
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
import xgboost as xgb
from xgboost import XGBClassifier
import numpy as np

df = pd.read_csv("C:/Training/Academy/Statistics (Python)/Cases/Wisconsin/BreastCancer.csv")

dum_df = pd.get_dummies(df, drop_first=True)

X = dum_df.iloc[:,1:10]
y = dum_df.iloc[:,10]

##################################################################################

############### X B Boost Library ###################
# Create the DMatrix from X and y: cancer_dmatrix
cancer_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary: params
params = {"objective":"reg:logistic", "max_depth":3}

# Perform cross-validation: cv_results
cv_results = xgb.cv(dtrain=cancer_dmatrix, params=params, 
                  nfold=3, num_boost_round=5, 
                  metrics="error", as_pandas=True, seed=2020)

# Print cv_results
print(cv_results)

# Print the accuracy
print(((1-cv_results["test-error-mean"]).iloc[-1]))

### AUC
cv_results = xgb.cv(dtrain=cancer_dmatrix, params=params, 
                  nfold=3, num_boost_round=5, 
                  metrics="auc", as_pandas=True, seed=2020)

# Print cv_results
print(cv_results)

# Print the accuracy
print(((cv_results["test-auc-mean"]).iloc[-1]))

################# sklearn API ######################

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                    random_state=2018,
                                                    stratify=y)

clf = XGBClassifier(random_state=2020)
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
clf = XGBClassifier(random_state=2022,use_label_encoder=False)
cv = GridSearchCV(clf, param_grid=parameters,cv=kfold,scoring='roc_auc')

cv.fit(X,y)
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




