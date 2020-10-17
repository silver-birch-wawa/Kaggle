from MIDA import mida,train
from get_data import get_data
import numpy as np
from RMSE import rmse
def run():
    mark=[]
    from params import j,term
    for i in range(term):
        data,compare_data,_=get_data()
        d=data.copy()
        mask=data.isnull()
        model=train(data.fillna(0).values,mask,30)
        filled_data=mida(model,data.fillna(0).values)

        from pandas import DataFrame
        filled_data=DataFrame(filled_data,index=compare_data.index,columns=compare_data.columns)
        for index,col in enumerate(compare_data):
            filled_data[col]=filled_data[col].astype(str(compare_data[col].dtype))

        j+=rmse(filled_data,compare_data)
        mark.append(rmse(filled_data,compare_data))
    # for i,j in enumerate(mark):
    #     print(i+2," err:",j)
    print(mark)
    print(np.sum(mark)/len(mark))
run()