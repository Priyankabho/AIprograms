

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, KFold

df = pd.read_csv("train.csv")
x = df.iloc[:,1:-1]
y = df.iloc[:,-1]
test = pd.read_csv("test.csv")
x_test = test.iloc[:,1:]

### svc linear
svc_l = SVC(kernel='linear',probability=True,random_state=2022)
params = {'C':np.linspace(0.001,6,5)}
kfold = KFold(n_splits=5,shuffle=True,random_state=2022)
gcv = GridSearchCV(svc_l,param_grid=params,scoring='roc_auc',cv=kfold)
gcv.fit(x,y)

print(gcv.best_params_)
print(gcv.best_score_)

pd_cv = pd.DataFrame(gcv.cv_results_)
best_model = gcv.best_estimator_
y_pred_l = gcv.predict(x_test)
submit = pd.DataFrame({'ID':test.ID,'TARGET':y_pred_l})
submit.to_csv("submit_gc.csv",index=False)


svc_r = SVC(kernel='rbf',probability=True,random_state=2022)
params = {'C':np.linspace(0.001,6,5),'gamma':np.linspace(1,7,5)}
kfold = KFold(n_splits=5,shuffle=True,random_state=2022)

gcv = GridSearchCV(svc_r,param_grid=params,scoring='roc_auc',cv=kfold)
gcv.fit(x,y)

print(gcv.best_params_)
print(gcv.best_score_)

y_pred_r = svc_r.predict(x_test)
submit = pd.DataFrame({'ID':test.ID,'TARGET':y_pred_r})
submit.to_csv("submit_svc.csv",index=False)

dt = DecisionTreeClassifier()
dt.fit(x, y)
y_pred_dt = dt.predict(x_test)
submit = pd.DataFrame({'ID':test.ID,'TARGET':y_pred_dt})
submit.to_csv("submit_dt.csv",index=False)