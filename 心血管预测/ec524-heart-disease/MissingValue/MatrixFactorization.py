from fancyimpute import MatrixFactorization

def mf(data):
    mf=MatrixFactorization()
    return mf.fit_transform(data)
from RMSE import rmse
def run():
    from params import j,term
    for i in range(term):       
        from get_data import get_data
        data,compare_data,_=get_data()
        d=mf(data)

        from pandas import DataFrame
        d=DataFrame(d,index=compare_data.index,columns=compare_data.columns)
        for index,col in enumerate(compare_data):
            d[col]=d[col].astype(str(compare_data[col].dtype))
        j+=rmse(d,compare_data)
    print(j/term)
if __name__ == "__main__":
  run()
# 5% 1.5786461908116887   1.7635036552650845
# 20% 1.0819814977872937   0.8684875659305165