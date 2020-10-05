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
# x都是缺失，y无缺失
len_train=len(x_train)
len_test=len(x_test)

def rmse(data,compare_data):
    return np.sqrt(np.sum(np.sum(data-compare_data)**2))/np.sum(np.sum(data-compare_data!=0.0))

def rf(data,compare_data):
    from missingpy import MissForest
    rf = MissForest()
    d=rf.fit_transform(data)
    cd=compare_data
    # print(d-cd)
    for i,j in zip(d,cd):
        print(i)
        print(j)
    print(d)
    print(cd)
    print(rmse(d,cd))

rf(data,compare_data)

# 1.754
# 0.3799