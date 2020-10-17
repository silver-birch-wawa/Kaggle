import impyute
def em(data):
    return impyute.em(data.values)

from RMSE import rmse
def run():
    from params import j,term
    for i in range(term):       
        from get_data import get_data
        data,compare_data,_=get_data()
        d=em(data)

        from pandas import DataFrame
        d=DataFrame(d,index=compare_data.index,columns=compare_data.columns)
        for index,col in enumerate(compare_data):
            d[col]=d[col].astype(str(compare_data[col].dtype))
        j+=rmse(d,compare_data)
    print(j/term)
if __name__ == "__main__":
  run()

# 20% 3.0560029251173497
# 5% 3.039140347790666