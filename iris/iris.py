import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
data = load_iris()
# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(data['data'],data['target'],random_state=0)

# 训练集合，选定参数k
def train(n_neighbor,x_train,y_train):
    knn=KNeighborsClassifier(n_neighbors=n_neighbor)
    knn.fit(x_train,y_train)
    return knn
# 计算准确率
def test(x_test,y_test,knn):
    prediction=knn.predict(x_test)
    num=0
    for i in range(len(prediction)):
        if(prediction[i]==y_test[i]):
            num+=1
    return num/len(prediction)
choosed_k=1;choosed_accurate=0
# 通过1到20遍历得到最佳的k
for k in range(1,20):
    knn=train(k,x_train,y_train)
    accurate=test(x_test,y_test,knn)
    print(accurate)
    if(accurate>choosed_accurate):
        choosed_accurate=accurate
        choosed_k=k
print("Final:",choosed_accurate)
# Final: 0.9736842105263158