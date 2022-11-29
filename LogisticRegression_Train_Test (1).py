import pandas as pd
import numpy as np

Default = pd.read_csv("G:/Statistics (Python)/Datasets/Default.csv")
dum_Default = pd.get_dummies(Default, drop_first=True)

## Considering only Student variable
#X=pd.DataFrame(dum_Default['student_Yes'])

X = dum_Default.iloc[:,[0,1,3]]
y = dum_Default.iloc[:,2]

# Import the necessary modules
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score


# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, 
                                                    random_state=42, stratify=y)

# Create the classifier: logreg
logreg = LogisticRegression()
# Fit the classifier to the training data
logreg.fit(X_train,y_train)
# Predict the labels of the test set: y_pred
y_pred = logreg.predict(X_test)

y_probs = logreg.predict_proba(X_test)
# Compute and print the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test,y_pred))

################ROC#############################

# Import necessary modules
from sklearn.metrics import roc_curve, roc_auc_score

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
import matplotlib.pyplot as plt
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
roc_auc_score(y_test, y_pred_prob)

############# k FOLD CV #############

from sklearn.model_selection import StratifiedKFold
#from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

kfold = StratifiedKFold(n_splits=5, random_state=42)

logreg = LogisticRegression()

results = cross_val_score(logreg, X, y, cv=kfold, 
                          scoring='roc_auc')
print(results)
print("ROC AUC: %.4f (%.4f)" % (results.mean(), results.std()))

results = cross_val_score(logreg, X, y, cv=kfold)
print(results)
print("Accuracy Score: %.4f (%.4f)" % (results.mean(), results.std()))






