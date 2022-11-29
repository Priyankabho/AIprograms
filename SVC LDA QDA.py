import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

df = pd.read_csv("Sonar.csv")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

le = LabelEncoder()
le_y = le.fit_transform(y)
print(le.classes_)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, le_y, random_state = 2022, test_size = 0.3, stratify = y)

## SVM - linear (tuning)

################ SVM Linear using Grid Search ##############
scaler = StandardScaler()
model = SVC(kernel = 'linear')

pipe = Pipeline([('scaler', scaler), ('SVC', model)])
kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 2022)
params = {'SVC__C': np.linspace(0.001, 6, 20)}

gcv = GridSearchCV(pipe, scoring = 'roc_auc', cv = kfold, param_grid = params)
gcv.fit(X_scaled, le_y)
print("SVM Linear Best Params using GCV: ", gcv.best_params_)
print("SVM Linear Best Score using GCV: ", gcv.best_score_)

################# SVM - RBF (tuning)  ###################
scaler = StandardScaler()
model = SVC(kernel = 'rbf')
pipe = Pipeline([('scaler', scaler),('SVC', model)])
kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 2022)
params = {'SVC__C': np.linspace(0.001, 6, 20),'SVC__gamma': np.linspace(0.001, 6, 20)}
gcv = GridSearchCV(pipe, scoring = 'roc_auc', cv = kfold, param_grid = params)
gcv.fit(X_scaled, le_y)
print("SVM Radial Best Params using GCV: ", gcv.best_params_)
print("SVM Radial Best Score using GCV: ", gcv.best_score_)

lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

#### Stratified K-Fold for LDA ####
kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 2022)
results = cross_val_score(lda, X_scaled, le_y, scoring = 'roc_auc', cv = kfold)
print("Stratified K_Fold ROC AUC Score for Linear Discriminant Analysis: ", results.mean())

## Quadratic Discriminant Analysis

qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)

#### Stratified K-Fold for QDA ####
kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 2022)
results = cross_val_score(qda, X_scaled, le_y, scoring = 'roc_auc', cv = kfold)
print("Stratified K_Fold ROC AUC Score for Quadratic Discriminant Analysis: ", results.mean())

