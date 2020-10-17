from EM import em
from AE import ae
from MatrixFactorization import mf
from MICE import mice
from MissForest import rf
from get_data import get_data
from GAIN import GAIN as gain
from RMSE import rmse
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

"""
每个LR/MLP负责一列的预测
填补只针对缺失值,但是训练可以训练随机缺失

其实缺失值填充有点NLP的味道,能不能上个注意力机制？
early-stop能不能自己实现一个？

GAN可不可以用于生成数据用于训练AE？？
"""
def lr_emsemble_train(data,compare_data):
    EM=em(data)
    # GAIN=gain(data)
    MF=mf(data)
    AE=ae(10,5,data).values
    RF=rf(data)
    MICE=mice(data)

    ttt=[]
    for i in range(len(MICE)):
        tt=[]
        for j in range(len(MICE[i])):
            t=[]
            t.append(MICE[i][j])
            t.append(EM[i][j])
            t.append(MF[i][j])
            t.append(AE[i][j])
            t.append(RF[i][j])
            tt.append(t)
        ttt.append(tt)

    from sklearn.linear_model import LogisticRegression as LR
    lr = [LR() for i in range(len(MICE[0]))]

    for i in range(len(data.values)):

    lr.fit(ttt,compare_data)
    return lr

def lr_emsemble_predict(data,lr):
    EM=em(data)
    # GAIN=gain(data)
    MF=mf(data)
    AE=ae(10,5,data).values
    RF=rf(data)
    MICE=mice(data)
    ttt=[]
    for i in range(len(MICE)):
        tt=[]
        for j in range(len(MICE[i])):
            t=[]
            t.append(MICE[i][j])
            t.append(EM[i][j])
            t.append(MF[i][j])
            t.append(AE[i][j])
            t.append(RF[i][j])
            tt.append(t)
        ttt.append(tt)
    return lr.predict(ttt)

def run():
    data,compare_data=get_data()
    from params import j,term
    for i in range(term):
        data,compare_data=get_data()   
        X_train, x_test, Y_train, y_test = train_test_split(data,compare_data,random_state=0)
        lr=lr_emsemble_train(X_train,Y_train)
        j+=rmse(lr_emsemble_predict(x_test),y_test)
    print(j/term)

run()