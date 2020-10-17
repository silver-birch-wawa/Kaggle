import pandas as pd 
import numpy as np 
import joblib
import random
import copy

from RMSE import rmse

def rf(data):
    from missingpy import MissForest
    rf = MissForest()
    return rf.fit_transform(data)
    # print(d-cd)
    # for i,j in zip(d,cd):
    #     print(i)
    #     print(j)
    # print(d)
    # print(cd)
    # print(rmse(d,cd))
def run():
    from params import j,term
    for i in range(term):       
        from get_data import get_data
        data,compare_data,_=get_data()
        d=rf(data)

        from pandas import DataFrame
        d=DataFrame(d,index=compare_data.index,columns=compare_data.columns)
        for index,col in enumerate(compare_data):
            d[col]=d[col].astype(str(compare_data[col].dtype))
        j+=rmse(d,compare_data)
    print(j/term)
if __name__ == "__main__":
  run()
# print(d-compare_data)
# print(d,compare_data)
# print(rmse(d,compare_data))
# 5% 1.6029478196466083
# 20% 1.0558371037895664