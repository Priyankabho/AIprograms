import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree

Housing = pd.read_csv(r"C:\Training\Academy\Statistics (Python)\Cases\Real Estate\Housing.csv")
dum_Housing = pd.get_dummies(Housing.iloc[:,1:11], drop_first=True)

X = dum_Housing
y = Housing.iloc[:,0]

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, 
                                                    random_state=42)

clf = DecisionTreeRegressor(max_depth=3,random_state=2021)
clf2 = clf.fit(X_train, y_train)

###################################################################
import graphviz 
# =============================================================================
# dot_data = tree.export_graphviz(clf2, out_file=None) 
# graph = graphviz.Source(dot_data) 
# graph.render("Housing") 
# =============================================================================
dot_data = tree.export_graphviz(clf2, out_file=None, 
                         feature_names=list(X_train),  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 

###################################################################

y_pred = clf2.predict(X_test)

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
print(mean_squared_error(y_test, y_pred) )
print(mean_absolute_error(y_test, y_pred))
print(r2_score(y_test, y_pred))



import matplotlib.pyplot as plt
plt.figure(figsize=(25,10))
tree.plot_tree(clf2,feature_names=X.columns,
               filled=True,fontsize=13,rounded=True)
plt.show()
#######################Grid Search CV#############################
depth_range = [None,3,4,5,6,7,8,9]
minsplit_range = [2,5,10,20,25,30]
minleaf_range = [1,5,10,15]

parameters = dict(max_depth=depth_range,
                  min_samples_split=minsplit_range, 
                  min_samples_leaf=minleaf_range)

from sklearn.model_selection import KFold
kfold = KFold(n_splits=5, random_state=2021,shuffle=True)
from sklearn.model_selection import GridSearchCV
clf = DecisionTreeRegressor(random_state=2021)
cv = GridSearchCV(clf, param_grid=parameters,
                  cv=kfold,scoring='r2')

cv.fit(X,y)
# Best Parameters
print(cv.best_params_)

print(cv.best_score_)

best_model = cv.best_estimator_

import matplotlib.pyplot as plt
plt.figure(figsize=(25,10))
tree.plot_tree(best_model,feature_names=X.columns,
               filled=True,fontsize=13,rounded=True)
plt.show()
######################################################################
best_model = cv.best_estimator_
import matplotlib.pyplot as plt

best_model.feature_importances_

ind = np.arange(X.shape[1])
plt.bar(ind,best_model.feature_importances_)
plt.xticks(ind,(X.columns),rotation=45)
plt.title('Feature Importance')
plt.xlabel("Variables")
plt.show()
########################################################################