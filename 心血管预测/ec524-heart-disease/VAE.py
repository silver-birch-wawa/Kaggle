import pandas as pd 
import numpy as np 
import joblib
import random
import copy
def generate_null(data,percents=5):
    # 生成NA数据(5%)
    for index, row in data.iterrows():
        for feature_name in data.iteritems():
            if(random.randint(0,100)<percents):
                data.at[index,feature_name[0]]=None
    return data

data=pd.read_csv("./heart.csv")
data=data.drop('target',axis=1)
compare_data=copy.deepcopy(data)
data=generate_null(data)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data,compare_data,random_state=0)
len_train=len(x_train)
len_test=len(x_test)
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
import torch.nn.functional as F

reconstruction_function = nn.BCELoss(size_average=False)  # mse loss

AE_TYPE='vae'
def loss_function(recon_x, x, mu, logvar):
    """
    recon_x: generating vec
    x: origin vec
    mu: latent mean
    logvar: latent log variance
    """
    BCE = reconstruction_function(recon_x, x)
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return BCE + KLD

# Creating the architecture of the Neural Network
# class AE(nn.Module):
    # def __init__(self, column_len, len1, len2,AE_TYPE):
    #     super(AE, self).__init__()

    #     self.fc1=nn.Linear(column_len,len1)
    #     self.fc21=nn.Linear(len1,len2)
    #     self.fc22=nn.Linear(len1,len2)
    #     self.fc3=nn.Linear(len2,len1)
    #     self.fc4=nn.Linear(len1,column_len)
#     def encode(self,x):
#         h1=F.relu(self.fc1(x),inplace=False)
#         return self.fc21(h1),self.fc22(h1)
#     def decode(self,x):
#         h3=F.relu(self.fc3(x),inplace=False)
#         return F.sigmoid(self.fc4(h3))
#     def reparametrize(self,u,logvar):
#         std = logvar.mul(0.5).exp_()
#         if torch.cuda.is_available():
#             eps = torch.cuda.FloatTensor(std.size()).normal_()
#         else:
#             eps = torch.FloatTensor(std.size()).normal_()
#         eps = Variable(eps)
#         return eps.mul(std).add_(u)
#     def forward(self, x):
#         u,logvar = self.encode(x)
#         z=self.reparametrize(u,logvar)
#         return self.decode(z)
class AE(nn.Module):
    def __init__(self, column_len, len1, len2,AE_TYPE):
        super(AE, self).__init__()

        self.fc1=nn.Linear(column_len,len1)
        self.fc21=nn.Linear(len1,len2)
        self.fc22=nn.Linear(len1,len2)
        self.fc3=nn.Linear(len2,len1)
        self.fc4=nn.Linear(len1,column_len)

    def encode(self, x):
        h1 = F.relu(self.fc1(x),inplace=False)
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z),inplace=False)
        return F.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z)

def best_train(len1,len2,x_train, x_test, y_train, y_test,op="RMS"):
    global data
    criterion = nn.MSELoss()
    
    # criterion=nn.L1Loss()
    ae = AE(len(data.columns),len1,len2,AE_TYPE).cuda()
    if op == "RMS":
        optimizer = optim.RMSprop(ae.parameters(), lr = 0.01, weight_decay = 0.5)
    elif op == "Adam":
        optimizer = optim.Adam(ae.parameters(),lr=0.1, weight_decay=0.5)
    bestterm = 0
    best_loss=99999999
    for terms in range(1,18):
        # input()
        LOSS=0
        for epoch in x_train.values:
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
            epoch=Variable(torch.from_numpy(epoch.astype(np.double)).double()).cuda()
            ae.double()
            outputs=ae(epoch.double())
            # outputs=list(outputs)
            if(len(mark_null)!=0):
                # print(outputs)
                outputs=torch.tensor(list(outputs),requires_grad=True).cuda()              
                # print(outputs)
                # print(mark_null)
                for index in mark_null:
                    outputs[index]=0
                # print(type(outputs),type(epoch))
                # print("----------------\n")
                loss = criterion(outputs, epoch)
                loss=torch.sqrt(loss)
                # print(loss)
                # rmse
                LOSS=LOSS+loss
                # loss = torch.sqrt(criterion(outputs,epoch))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        if(LOSS<best_loss):
            best_loss=LOSS
            bestterm=terms
    return bestterm+1

def train(ae,criterion,optimizer,len1,len2,op="RMS"):
    global data,x_train, x_test, y_train, y_test
    bestterm=best_train(len1,len2,x_train.copy(), x_test.copy(), y_train.copy(), y_test.copy(),op)
    print(bestterm)
    for terms in range(bestterm):
        # input()
        LOSS=0
        for epoch in x_train.values:
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
            epoch=Variable(torch.from_numpy(epoch.astype(np.double)).double()).cuda()
            ae.double()
            outputs=ae(epoch.double())
            for index in mark_null:
                outputs[index]=0
            print(outputs,epoch)
            loss = criterion(outputs, epoch)
            loss=torch.sqrt(loss)
            # print(loss)
            # rmse
            LOSS=LOSS+loss
            # loss = torch.sqrt(criterion(outputs,epoch))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
def test(len1,len2,printf,op="RMS"):
    criterion = nn.MSELoss()
    # criterion=nn.L1Loss()
    # criterion=nn.CrossEntropyLoss()
    ae = AE(len(data.columns),len1,len2,AE_TYPE).cuda()
    if op == "RMS":
        optimizer = optim.RMSprop(ae.parameters(), lr = 0.01, weight_decay = 0.5)
    elif op == "Adam":
        optimizer = optim.Adam(ae.parameters(),lr=0.1, weight_decay=0.5)    
    train(ae,criterion,optimizer,len1,len2,op)
    

    # 查看参数
    # for name, param in ae.named_parameters():
    #     print(name,param)
    # input()

    # 计算损失
    cal=0
    # 统一填充0
    # x_test=x_test.fillna(0)
    mark_loss_num=0
    for index,epoch in enumerate(x_test.copy().values):
        # 标记缺失位置
        mark_null=[]
        for i,v in enumerate(epoch):
            if(np.isnan(v)):
                mark_null.append(i)
                epoch[i]=0
        
        epoch=Variable(torch.from_numpy(epoch.astype(np.double)).double()).cuda()
        ae.double()
        pre=ae(epoch.double())

        newpre=copy.deepcopy(epoch)
        for i in mark_null:
            newpre[i]=int(pre[i]) if pre[i]>0 else 0
            
            if(x_train.columns[i]=='sex'):
                if newpre[i]>1:
                    newpre[i]= 1 
            
        loss=criterion(newpre,torch.tensor(y_test.values[index]).cuda())
        loss=torch.sqrt(loss)
        # print(epoch)
        # print(pre)
        # print(torch.tensor(y_test.values[index]))
        # print("loss:",loss)
        if(len(mark_null)!=0 and printf):
            # print(torch.tensor(x_test.values[index]))
            for i in mark_null:
                print(x_train.columns[i],": ",float(newpre[i]),y_test.values[index][i])
            # print(mark_null)
            # print(newpre)
            # print(torch.tensor(y_test.values[index]))
            print("#######################")
        cal=cal+loss
        mark_loss_num=mark_loss_num+len(mark_null)
    # print(mark_loss_num)
    print(len1,len2,cal/mark_loss_num)
    # 10 8 193
    print("----------")
    # input()
    return float(cal/mark_loss_num)
def best_test(op="RMS"):
    best_len1=5
    best_len2=10
    best_loss=999999999999.
    for len2 in range(5,10):
        for len1 in range(len2+1,len(data.columns)):
            t_loss=test(len1,len2,False,op)
            if(float(t_loss)<best_loss):
                best_len1=len1
                best_len2=len2
                best_loss=t_loss
    return best_len1,best_len2,best_loss
res=0
def run():
    global res
    for i in range(20):
        len1,len2,best_loss=best_test("Adam")
        res=res+best_loss
        # len1=12
        # len2=9
        # import os
        # os.system("cls")
        test(len1,len2,True)
    print("avgloss:",res/10)
    # print("best:",len1,len2)
run()