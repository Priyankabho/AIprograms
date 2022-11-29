import pandas as pd
import numpy as np
df = pd.read_csv("C:\Training\Academy\Statistics (Python)\Cases\Real Estate\Housing.csv")
dum_df = pd.get_dummies(df.iloc[:,1:11], drop_first=True)

from sklearn.model_selection import train_test_split 

X = dum_df
y = df.iloc[:,0]

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                    random_state=2022)

from xgboost import XGBRegressor
clf = XGBRegressor(random_state=2022)
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

#def mean_absolute_percentage_error(y_true, y_pred): 
#    y_true, y_pred = np.array(y_true), np.array(y_pred)
#    return np.mean(np.abs((y_true - y_pred) / y_true))
#
#print(mean_absolute_percentage_error(y_test,y_pred))

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
print(mean_squared_error(y_test, y_pred))
print(mean_absolute_error(y_test, y_pred))
print(r2_score(y_test, y_pred))

################ Tunning XG Boost ##################################

lr_range = [0.001, 0.01, 0.1, 0.2,0.25, 0.3]
n_est_range = [10,20,30,50,100]
md_range = [2,4,6,8,10]

parameters = dict(learning_rate=lr_range,
                  n_estimators=n_est_range,
                  max_depth=md_range)

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
kfold = KFold(n_splits=5, random_state=2022,shuffle=True)

clf = XGBRegressor(random_state=1211)
cv = GridSearchCV(clf, param_grid=parameters,
                  cv=kfold,scoring='r2')

cv.fit(X,y)

print(cv.best_params_)

print(cv.best_score_)
