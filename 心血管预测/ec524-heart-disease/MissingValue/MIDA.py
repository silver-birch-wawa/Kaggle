import pandas as pd 
import numpy as np 
import joblib
import random
import copy
from get_data import get_data
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler


num_epochs = 60
theta=7
miss_rato=0.05


drop_out_ratio=0.1
test_size = 0.3
mechanism = 'mcar'
method = 'random'
use_cuda=False
device = torch.device("cuda" if use_cuda else "cpu")
# Creating the architecture of the Neural Network
class AE(nn.Module):
    def __init__(self, column_len):
        super(AE, self).__init__()
        global drop_out_ratio
        self.dim=column_len
        self.drop_out = nn.Dropout(p=drop_out_ratio)
        self.encoder = nn.Sequential(
            nn.Linear(column_len,column_len+theta),
            nn.Tanh(),
            nn.Linear(column_len+theta,column_len+2*theta)
            )
        self.decoder = nn.Sequential(
            nn.Linear(column_len+2*theta,column_len+theta),
            nn.Tanh(),
            nn.Linear(column_len+theta,column_len),
            )

    def forward(self, x):
        x = x.view(-1, self.dim)
        x_missed = self.drop_out(x)
        z = self.encoder(x_missed)
        out = self.decoder(z)
        
        out = out.view(-1, self.dim)
        
        return out

def missing_method(raw_data, mechanism='mcar', method='uniform',p=0.05) :
    data = raw_data.copy()
    rows, cols = data.shape
    # missingness threshold
    t = p
    # uniform 自定义缺失率
    # random 随机丢一半
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

def train(train_data,mask,batch_size):
    rows, cols = train_data.shape
    model = AE(cols).to(device)

    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                            batch_size=batch_size,
                                            shuffle=True)

    loss = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), momentum=0.99, lr=0.01, nesterov=True)

    cost_list = []
    model_list=[]
    early_stop = False
    for epoch in range(num_epochs):
        total_batch = len(train_data) // batch_size
        sum_cost=0
        for i, batch_data in enumerate(train_loader):
            # 缺失值已经被置为0，所以输出
            batch_data = batch_data.to(device)
            reconst_data = model(batch_data.float())
            # for j in range(len(reconst_data[0])):
            #     if(mask.values[i][j]==True):
            #         reconst_data[0][j]=0
            # print(batch_data)
            # print(reconst_data)
            # print("-------")
            cost = loss(reconst_data.float(), batch_data.float())
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
                    
            if (i+1) % (total_batch//2) == 0:
                print('Epoch [%d/%d], lter [%d/%d], Loss: %.6f'
                    %(epoch+1, num_epochs, i+1, total_batch, cost.item()))
                
            sum_cost+=cost.item()
            # early stopping rule 1 : MSE < 1e-06
            if cost.item() < 1e-06 :
                early_stop = True
                break
    #         early stopping rule 2 : simple moving average of length 5
    #         sometimes it doesn't work well.
    #         if len(cost_list) > 5 :
    #            if cost.item() > np.mean(cost_list[-5:]):
    #                early_stop = True
    #                break
        model_list.append(model)
        cost_list.append(sum_cost)
        if epoch-cost_list.index(min(cost_list))>10:
            break
    print(min(cost_list))
    return model_list[cost_list.index(min(cost_list))]

def mida(model,test_data):
    model.eval()
    test_data = torch.from_numpy(test_data).float()
    filled_data = model(test_data.to(device))
    filled_data = filled_data.cpu().detach().numpy()
    return filled_data

def run():
    data,compare_data=get_data()
    data=compare_data
    rows, cols = data.shape
    shuffled_index = np.random.permutation(rows)
    train_index = shuffled_index[:int(rows*(1-test_size))]
    test_index = shuffled_index[int(rows*(1-test_size)):]

    train_data = data.values[train_index, :]
    test_data = data.values[test_index, :]

    scaler = MinMaxScaler()
    scaler.fit(train_data)

    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)

    data, mask=missing_method(test_data,mechanism,method,miss_rato)
    missed_data = torch.from_numpy(data).double()
    train_data = torch.from_numpy(train_data).double()

    model=train(train_data)
    filled_data=mida(model,test_data)

    from RMSE import rmse
    from pandas import DataFrame
    filled_data=DataFrame(filled_data,index=compare_data.index,columns=compare_data.columns)
    for index,col in enumerate(compare_data):
        filled_data[col]=filled_data[col].astype(str(compare_data[col].dtype))

    err=rmse(filled_data,test_data)
    print(filled_data,test_data)
    print("err:",err)

if __name__ == "__main__":
    run()   

"""
只能对data进行处理，最后跟compare比较计算RMSE，考虑两种情况：
(1) 直接训练
(2) 挖空训练

如何选择最佳参数并early stop？
"""

# 5% 4.948188508223597