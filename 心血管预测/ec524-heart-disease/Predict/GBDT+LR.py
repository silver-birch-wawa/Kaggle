import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

data = load_iris()
# 划分训练集和测试集
X_train, x_test, Y_train, y_test = train_test_split(data['data'],data['target'],random_state=0)

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import GradientBoostingClassifier
gbm = GradientBoostingClassifier(n_estimators=50, random_state=10, subsample=0.6)
gbm.fit(X_train, Y_train)
weak_classifier_output=gbm.apply(x_test)
weak_classifier_output=weak_classifier_output.reshape(-1,50)

print(weak_classifier_output)