import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import  roc_curve, auc, roc_auc_score

X_train = pd.read_csv('X_train.csv' )
y_train = pd.read_csv('y_train.csv')
X_test2 = pd.read_csv('X_test2.csv')
y_test2 = pd.read_csv('y_test2.csv')

print(X_train.shape)
print(y_train.shape)
print(X_test2.shape)
print(y_test2.shape)
y_train = np.ravel(y_train)

model = SGDClassifier()
model.fit(X_train, y_train)
y_test2_predictions = model.predict(X_test2)
y_test2_predictions = pd.DataFrame(y_test2_predictions)
y_test2_predictions.to_csv('y_test2_predictions.csv', index=False)