import pandas as pd 
import numpy as np 
import joblib
import random
import copy
from get_data import get_data

data,compare_data=get_data()
# miss_data=data[data.isnull().values==True]
# # miss_data=data[data.isnull().values==True].drop("id",axis=1)

# unmiss_data=data.drop(data[data.isnull().values==True].index)
# # unmiss_data=data.drop(data[data.isnull().values==True].index).drop("id",axis=1)

# medians={}
# # 提取众数，平均数，中位数
# for col in data:
#     temp=data[col]
#     unmiss=temp.drop(temp[temp.isnull()==True].index)
#     print(col," ",np.mean(unmiss),np.median(unmiss))
#     means=np.mean(unmiss)
#     medians[col]=np.median(unmiss)

# print(medians)

# 每个缺失变量的缺失值数量
# for index, row in miss_data.iterrows():
#     print(np.sum(row.isnull()))
# 一般会先聚类处理一下（缺失值怎么聚类？）
# 两种方法：一种利用缺失值预填充(可以利用更多信息，适应性更强)，一种利用无缺失值的干净数据
# 填充方法：填充0，填充中位数，填充众数，填充平均数
# 一般认为AutoEncoder是利用置0法然后对输出的对应位置置0然后不反向传播
# 交叉熵 VS MSE

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

AE_TYPE='ae'

# Creating the architecture of the Neural Network
class AE(nn.Module):
    def __init__(self, column_len, len1, len2,AE_TYPE):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(column_len, len1),
            nn.ReLU(inplace=True),
            nn.Linear(len1,len2),
            nn.ReLU(inplace=True)
            )
        self.decoder = nn.Sequential(
            nn.Linear(len2,len1),
            nn.ReLU(inplace=True),
            nn.Linear(len1,column_len),
            )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def best_train(len1,len2,data,compare_data,op="RMS"):
    #global data,compare_data
    criterion = nn.MSELoss()
    
    # criterion=nn.L1Loss()
    ae = AE(len(data.columns),len1,len2,AE_TYPE)
    if op == "RMS":
        optimizer = optim.RMSprop(ae.parameters(), lr = 0.01, weight_decay = 0.5)
    elif op == "Adam":
        optimizer = optim.Adam(ae.parameters(),lr=0.1, weight_decay=0.5)
    bestterm = 0
    best_loss=99999999
    for terms in range(1,18):
        # input()
        LOSS=0
        for epoch in data.values:
            # mark the null position
            mark_null=[]
            for index,v in enumerate(epoch):
                if(np.isnan(v)):
                    # print(index,v)
                    mark_null.append(index)
                    epoch[index]=0.0
                    # print(epoch)
            # print(mark_null)
            # train
            from torch.autograd import Variable
            epoch=Variable(torch.from_numpy(epoch.astype(np.double)).double())
            ae.double()
            outputs=ae(epoch.double())
            for index in mark_null:
                outputs[index]=0
            loss = criterion(outputs, epoch)
            loss=torch.sqrt(loss)
            # print(loss)
            # rmse
            LOSS+=loss
            # loss = torch.sqrt(criterion(outputs,epoch))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if(LOSS<best_loss):
            best_loss=LOSS
            bestterm=terms
    return bestterm+1

def train(ae,criterion,optimizer,len1,len2,op="RMS"):
    # global data,compare_data
    # bestterm=best_train(len1,len2,data.copy(),compare_data.copy(),op)
    bestterm=10
    print(bestterm)
    for terms in range(bestterm):
        # input()
        LOSS=0
        for epoch in data.values:
            # mark the null position
            mark_null=[]
            for index,v in enumerate(epoch):
                if(np.isnan(v)):
                    # print(index,v)
                    mark_null.append(index)
                    epoch[index]=0.0
                    # print(epoch)
            # print(mark_null)
            # train
            from torch.autograd import Variable
            epoch=Variable(torch.from_numpy(epoch.astype(np.double)).double())
            ae.double()
            outputs=ae(epoch.double())
            for index in mark_null:
                outputs[index]=0
            loss = criterion(outputs, epoch)
            loss=torch.sqrt(loss)
            # print(loss)
            # rmse
            LOSS+=loss
            # loss = torch.sqrt(criterion(outputs,epoch))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return ae
def ae(len1,len2,data,op="RMS"):
    criterion = nn.MSELoss()
    # criterion=nn.L1Loss()
    # criterion=nn.CrossEntropyLoss()
    ae = AE(len(data.columns),len1,len2,AE_TYPE)
    if op == "RMS":
        optimizer = optim.RMSprop(ae.parameters(), lr = 0.01, weight_decay = 0.5)
    elif op == "Adam":
        optimizer = optim.Adam(ae.parameters(),lr=0.1, weight_decay=0.5)  
    ae=train(ae,criterion,optimizer,len1,len2,op)

    # 查看参数
    # for name, param in ae.named_parameters():
    #     print(name,param)
    # input()

    d=data.copy()
    for index,epoch in enumerate(data.copy().values):
        # 标记缺失位置
        mark_null=[]
        for i,v in enumerate(epoch):
            if(np.isnan(v)):
                mark_null.append(i)
                epoch[i]=0
        
        epoch=Variable(torch.from_numpy(epoch.astype(np.double)).double())
        ae.double()
        #epoch=torch.from_numpy(epoch.astype(np.double)).double()
        pre=ae(epoch.double())
        d.values[index]=pre.detach().numpy()
    from pandas import DataFrame
    d=DataFrame(d,index=data.index,columns=data.columns)
    for index,col in enumerate(d):
        # print(data[col].dtype)
        d[col]=d[col].astype(str(data[col].dtype))
    return d
def test(len1,len2,data,op="RMS"):
    criterion = nn.MSELoss()
    # criterion=nn.L1Loss()
    # criterion=nn.CrossEntropyLoss()
    ae = AE(len(data.columns),len1,len2,AE_TYPE)
    if op == "RMS":
        optimizer = optim.RMSprop(ae.parameters(), lr = 0.01, weight_decay = 0.5)
    elif op == "Adam":
        optimizer = optim.Adam(ae.parameters(),lr=0.1, weight_decay=0.5)  
    ae=train(ae,criterion,optimizer,len1,len2,op)

    # 查看参数
    # for name, param in ae.named_parameters():
    #     print(name,param)
    # input()

    d=data.copy()
    for index,epoch in enumerate(data.copy().values):
        # 标记缺失位置
        mark_null=[]
        for i,v in enumerate(epoch):
            if(np.isnan(v)):
                mark_null.append(i)
                epoch[i]=0
        
        epoch=Variable(torch.from_numpy(epoch.astype(np.double)).double())
        ae.double()
        #epoch=torch.from_numpy(epoch.astype(np.double)).double()
        pre=ae(epoch.double())
        d.values[index]=pre.detach().numpy()
    return d

from RMSE import rmse

def best_test(op="RMS"):   
    best_len1=5
    best_len2=10
    best_loss=999999999999.
    for len2 in range(5,10):
        for len1 in range(len2+1,len(data.columns)):
            d=test(len1,len2,data,op)
            # print(d)
            for col in compare_data.columns:
                d[col]=d[col].astype(str(compare_data[col].dtype))
            #print(d)
            # input()
            t_loss=rmse(d,compare_data)
            if(float(t_loss)<best_loss):
                best_len1=len1
                best_len2=len2
                best_loss=t_loss
    return best_len1,best_len2,best_loss

def run():
    from params import j,term
    res=0
    for i in range(term):
        data,compare_data,_=get_data()
        len1,len2,best_loss=best_test("Adam")
        res+=best_loss
        print("len:",len1,len2)
        # len1=12
        # len2=9
        # import os
        # os.system("cls")
        # test(len1,len2,True)
    print("avgloss:",res/term)
    # print("best:",len1,len2)
if __name__ == "__main__":
  run()
# AE
# 12 9 1.3161  MAE RMS
# 8 7 4.1190/3.9806 RMSE RMS

# 8 6  RMSE Adam 3.456758494016127
# 9 5  RMSE Adam 3.8411474624303423
# 7 6  RMSE RMS  6.7134

# 5% 3.8899614968041343
# 20% 4.085027844100423



def missing_method(raw_data, mechanism='mcar', method='uniform') :
    data = raw_data.copy()
    rows, cols = data.shape
    # missingness threshold
    t = 0.2
    if mechanism == 'mcar' :    
        if method == 'uniform' :
            # uniform random vector
            v = np.random.uniform(size=(rows, cols))
            # missing values where v<=t
            mask = (v<=t)
            data[mask] = 0
        elif method == 'random' :
            # only half of the attributes to have missing value
            missing_cols = np.random.choice(cols, cols//2)
            c = np.zeros(cols, dtype=bool)
            c[missing_cols] = True
            # uniform random vector
            v = np.random.uniform(size=(rows, cols))
            # missing values where v<=t
            mask = (v<=t)*c
            data[mask] = 0
        else :
            print("Error : There are no such method")
            raise
    elif mechanism == 'mnar' :        
        if method == 'uniform' :
            # randomly sample two attributes
            sample_cols = np.random.choice(cols, 2)
            # calculate ther median m1, m2
            m1, m2 = np.median(data[:,sample_cols], axis=0)
            # uniform random vector
            v = np.random.uniform(size=(rows, cols))
            # missing values where (v<=t) and (x1 <= m1 or x2 >= m2)
            m1 = data[:,sample_cols[0]] <= m1
            m2 = data[:,sample_cols[1]] >= m2
            m = (m1*m2)[:, np.newaxis]
            mask = m*(v<=t)
            data[mask] = 0
        elif method == 'random' :
            # only half of the attributes to have missing value
            missing_cols = np.random.choice(cols, cols//2)
            c = np.zeros(cols, dtype=bool)
            c[missing_cols] = True
            # randomly sample two attributes
            sample_cols = np.random.choice(cols, 2)
            # calculate ther median m1, m2
            m1, m2 = np.median(data[:,sample_cols], axis=0)
            # uniform random vector
            v = np.random.uniform(size=(rows, cols))
            # missing values where (v<=t) and (x1 <= m1 or x2 >= m2)
            m1 = data[:,sample_cols[0]] <= m1
            m2 = data[:,sample_cols[1]] >= m2
            m = (m1*m2)[:, np.newaxis]
            mask = m*(v<=t)*c
            data[mask] = 0
        else :
            print("Error : There is no such method")
            raise
    else :
        print("Error : There is no such mechanism")
        raise
    return data, mask