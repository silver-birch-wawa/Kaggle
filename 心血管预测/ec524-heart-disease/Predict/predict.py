import pandas as pd 
import numpy as np 
import joblib

data=pd.read_csv("./train.csv")

data=data.drop(data[data.isnull().values==True].index)

train=data.drop(["heart_disease",'id'],axis=1)
test=data['heart_disease']

# 预处理NA

from sklearn.model_selection import train_test_split
X_train, x_test, Y_train, y_test = train_test_split(train,test,random_state=0)

#print(X_train,x_test,Y_train,y_test)
# 227 76

def Trees_Leaf(str,X_train, x_test, Y_train, y_test,n_estimators=2000):
    if str == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(bootstrap = True, oob_score = True, criterion = 'gini',n_estimators=n_estimators)
        clf.fit(X_train, Y_train)
        lv_train=clf.apply(X_train).reshape(-1,n_estimators)
        lv_test=clf.apply(x_test).reshape(-1,n_estimators)
        return lv_train,lv_test
    elif str == 'gbdt':
        from sklearn.ensemble import GradientBoostingClassifier
        gbm = GradientBoostingClassifier(n_estimators=n_estimators, random_state=10, subsample=0.6, max_depth=7)
        gbm.fit(X_train, Y_train)
        lv_train=gbm.apply(X_train).reshape(-1,n_estimators)
        lv_test=gbm.apply(x_test).reshape(-1,n_estimators)
        return lv_train,lv_test
    elif str == 'xgb':
        import xgboost as xgb
        from xgboost import XGBClassifier
        xgbm= XGBClassifier(max_depth=15,
                            learning_rate=0.1,
                            n_estimators=n_estimators,
                            min_child_weight=5,
                            max_delta_step=0,
                            subsample=0.8,
                            colsample_bytree=0.7,
                            reg_alpha=0,
                            reg_lambda=0.4,
                            scale_pos_weight=0.8,
                            silent=True,
                            objective='binary:logistic',
                            missing=None,
                            eval_metric='auc',
                            seed=1440,
                            gamma=0)
        xgbm.fit(X_train, Y_train)
        lv_train=xgbm.apply(X_train).reshape(-1,n_estimators)
        lv_test=xgbm.apply(x_test).reshape(-1,n_estimators)
        return lv_train,lv_test
    elif str == 'lgb':
        import lightgbm as lgb
        ### 数据转换
        data_train = lgb.Dataset(X_train,Y_train, free_raw_data=False,silent=True)
        data_test = lgb.Dataset(x_test, y_test, reference=data_train,free_raw_data=False,silent=True)
        params = {
                    
                    'n_estimators':n_estimators,
                    'boosting_type': 'gbdt',
                    'boosting': 'dart',
                    'objective': 'binary',
                    'metric': 'binary_logloss',

                    'learning_rate': 0.01,
                    'num_leaves':25,
                    'max_depth':3,

                    'max_bin':10,
                    'min_data_in_leaf':8,

                    'feature_fraction': 0.6,
                    'bagging_fraction': 1,
                    'bagging_freq':0,

                    'lambda_l1': 0,
                    'lambda_l2': 0,
                    'min_split_gain': 0
        }
        gbm = lgb.train(params,                     # 参数字典
                        data_train,                  # 训练集
                        num_boost_round=n_estimators,       # 迭代次数
                        verbose_eval=2000,           # 每运行多少次打印一次(尽量设置大一点)
                        valid_sets=data_test,        # 验证集
                        early_stopping_rounds=30)   # 早停系数                      
        lv_train=gbm.predict(X_train,pred_leaf=True).reshape(-1,n_estimators)
        lv_test=gbm.predict(x_test,pred_leaf=True).reshape(-1,n_estimators)
        return lv_train,lv_test

def onehotencoder(train_new_feature):
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder()
    enc.fit(train_new_feature)
    #train_new_feature2 = np.array(enc.transform(train_new_feature).toarray()) 
    #return train_new_feature2
    return enc

def accuracy(y_pred,y_true):
    from sklearn.metrics import accuracy_score
    return accuracy_score(y_pred,y_true)

def precision(y_true, y_pred):
    from sklearn.metrics import precision_score
    return precision_score(y_true, y_pred, average='micro')

def LR(lv_train,lv_test):
    global X_train, x_test, Y_train, y_test

    lv_data=np.concatenate((lv_train,lv_test),axis=0)
    enc=onehotencoder(lv_data)
    # print(len(lv_train))
    lv_train=np.array(enc.transform(lv_train).toarray()) 
    lv_test=np.array(enc.transform(lv_test).toarray())

    from sklearn.linear_model import LogisticRegression
    lr=LogisticRegression()

    lr.fit(lv_train,Y_train)

    y_=lr.predict(lv_test)

    print("accuracy_score:",accuracy(y_,y_test))
    print("precision:",precision(y_,y_test))    

train=np.array([[] for i in range(len(Y_train))])
test=np.array([[] for i in range(len(y_test))])

def LRs(str,X_train, x_test, Y_train, y_test,n_estimators):
    global train,test
    lv_train,lv_test=Trees_Leaf(str,X_train, x_test, Y_train, y_test,n_estimators)
    train=np.concatenate((train,lv_train),axis=1)
    test=np.concatenate((test,lv_test),axis=1)
    print("\n"+str+":")
    LR(lv_train,lv_test)

# n_estimators=400
# lv_train,lv_test=Trees_Leaf("rf",X_train, x_test, Y_train, y_test,n_estimators)

# train=np.concatenate((train,lv_train),axis=1)
# test=np.concatenate((test,lv_test),axis=1)
# LR(lv_train,lv_test)
LRs("rf",X_train, x_test, Y_train, y_test,400)
# 84.21%
# n_estimators=2100
# lv_train,lv_test=Trees_Leaf("lgb",X_train, x_test, Y_train, y_test,n_estimators)
# train=np.concatenate((train,lv_train),axis=1)
# test=np.concatenate((test,lv_test),axis=1)
# LR(lv_train,lv_test)
LRs("lgb",X_train, x_test, Y_train, y_test,2100)
# 84.21%
# n_estimators=60
# lv_train,lv_test=Trees_Leaf("xgb",X_train, x_test, Y_train, y_test,n_estimators)
# train=np.concatenate((train,lv_train),axis=1)
# test=np.concatenate((test,lv_test),axis=1)
# LR(lv_train,lv_test)
LRs("xgb",X_train, x_test, Y_train, y_test,60)
# 84.21%
n_estimators=100
# lv_train,lv_test=Trees_Leaf("gbdt",X_train, x_test, Y_train, y_test,n_estimators)
# train=np.concatenate((train,lv_train),axis=1)
# test=np.concatenate((test,lv_test),axis=1)
# LR(lv_train,lv_test)
LRs("gbdt",X_train, x_test, Y_train, y_test,100)
# 85.52%
print(len(train[0]),len(test[0]))
LR(train,test)