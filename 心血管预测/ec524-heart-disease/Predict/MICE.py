import pandas as pd 
import numpy as np 

from RMSE import rmse

def mice(data):
    from fancyimpute import IterativeImputer as MICE
    data=MICE().fit_transform(data)  
    return data
# import sys

# sys.setrecursionlimit(1000000)

def run():
    from params import j,term
    for i in range(term):    
        from get_data import get_data
        data,compare_data,_=get_data()

        d=mice(data)


        from pandas import DataFrame
        d=DataFrame(d,index=compare_data.index,columns=compare_data.columns)
        for index,col in enumerate(compare_data):
            d[col]=d[col].astype(str(compare_data[col].dtype))

        # print(d-compare_data)
        # print(d,compare_data)
        # print(rmse(d,compare_data))
        j+=rmse(d,compare_data)
    print(j/term)
if __name__ == "__main__":
  run()
# 5% 1.252403709808228
# 20% 0.8354852520954757