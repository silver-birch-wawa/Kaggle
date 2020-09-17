import pandas as pd 
import numpy as np 
import joblib
import random
import copy
data=pd.read_csv("./heart.csv")
compare_data=copy.deepcopy(data)
# 生成NA数据(2%)
for index, row in data.iterrows():
    for feature_name in data.iteritems():
        if(random.randint(0,100)<2):
            data.at[index,feature_name[0]]=None

miss_data=data[data.isnull().values==True]
# miss_data=data[data.isnull().values==True].drop("id",axis=1)

unmiss_data=data.drop(data[data.isnull().values==True].index)
# unmiss_data=data.drop(data[data.isnull().values==True].index).drop("id",axis=1)

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

# Creating the architecture of the Neural Network
class SAE(nn.Module):
    def __init__(self, column_len,len1,len2):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(column_len, len1)
        self.fc2 = nn.Linear(len1, len2)
        self.fc3 = nn.Linear(len2, len1)
        self.fc4 = nn.Linear(len1, column_len)
        self.activation = nn.Sigmoid()
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x
def train(sae,criterion,optimizer,data,len1,len2):
        sae = SAE(len(data.columns),len1,len2)

for len1 in range(5,10):
    for len2 in range(len1,len(data.columns)):
        criterion = nn.MSELoss()
        # criterion=nn.CrossEntropyLoss()
        optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)
        # optimizer= optim.Adam(sae.parameters(),lr=0.1, weight_decay=0.5)
        train(sae,criterion,optimizer,copy.deepcopy(data))
