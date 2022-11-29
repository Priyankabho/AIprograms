import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score

pizza = pd.read_csv("pizza.csv")

lr = LinearRegression()
X = pizza['Promote'].values
y = pizza['Sales']

lr.fit(X.reshape(-1,1),y)

print(lr.intercept_)
print(lr.coef_)

insure = pd.read_csv("Insure_auto.csv")

X = insure.iloc[:,1:3]
y = insure.iloc[:,3]

lr.fit(X,y)
print(lr.intercept_)
print(lr.coef_)
####################################
boston = pd.read_csv("Boston.csv")
X = boston.iloc[:,:-1]
y = boston.iloc[:,-1]
X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                 test_size=0.3,
                                                 random_state=2022)

lr.fit(X_train,y_train)
print(lr.intercept_)
print(lr.coef_)

y_pred = lr.predict(X_test)
print(r2_score(y_test,y_pred))
############################################
kfold = KFold(n_splits=5,shuffle=True,random_state=2022)
results = cross_val_score(lr, X,y,scoring='r2',cv=kfold)
print(results.mean())


