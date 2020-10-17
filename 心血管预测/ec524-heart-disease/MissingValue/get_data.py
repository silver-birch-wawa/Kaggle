import pandas as pd 
import numpy as np 
import random
import copy
def generate_null(data,percents=20):
    # 生成NA数据(5%)
    for index, row in data.iterrows():
        for feature_name in data.iteritems():
            if(random.randint(0,100)<percents and not np.isnan(data.at[index,feature_name[0]])):
                data.at[index,feature_name[0]]=None
    return data
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
def get_data():
    data=pd.read_csv("./heart.csv")
    res=data['target']
    data=data.drop('target',axis=1)
    compare_data=copy.deepcopy(data)
    from params import percents,mechanism,method
    data=generate_null(data,percents)

    # data,mask=missing_method(data,mechanism=mechanism, method=method,p=percents/100)
    # compare_data=data.copy()
    # data[mask]=None
    return data,compare_data,res