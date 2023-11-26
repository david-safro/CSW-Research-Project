from data import data_preprocessing
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score
import time
X_train, X_test, y_train, y_test = data_preprocessing('Heart_Disease_Prediction.csv')
svm = svm.SVC()
svm.fit(X_train, y_train)
start = time.time()
y_pred = svm.predict(X_test)
end = time.time()
accuracy = accuracy_score(y_pred, y_test)
print(accuracy)
print("time:" + str(end-start))
print("f1  = " + str(f1_score(y_test, y_pred)))