import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split  

# def rand_train_test(mb):
#     bases = mb.dataset.unique()
#     train = pd.DataFrame()
#     val = pd.DataFrame()
    
#     for dataset in bases:
#         now=mb[mb.dataset == dataset]
#         n=now.shape[0]
#         perm = torch.as_tensor(np.random.permutation(n))
        
#         train_num=int(n*.75)
#         train_indices = perm[:train_num]
        
#         val_indices = perm[train_num:]

#         if train.empty:
#             train=now.iloc[train_indices]
#         else:
#             train=pd.concat([train,now.iloc[train_indices]])

#         if val.empty:
#             val=now.iloc[val_indices]
#         else:
#             val=pd.concat([val,now.iloc[val_indices]])
#     return train,val

def load_dataset(data_dir):
    mb = pd.read_csv(data_dir)
    
    # #META_TARGETS = ['DistComp', 'Recall', 'QueryTime']
    # META_TARGETS = ['Recall']
    #划分train,test
    #返回划分后的csv格式数据
    train,val=train_test_split(mb, test_size=0.25)

    # get_xy = lambda x: (x.drop(META_TARGETS, axis=1).copy(), x.loc[:, META_TARGETS].copy())
    # #将元数据库中图配置数据和元目标分离
    # train_x,train_y=get_xy(train)
    # test_x,test_y=get_xy(test)
    
    return train,val

def rand_train(train,batch_size):
    length=train.shape[0]
    
    ans=[]
    l=0
    r=min(batch_size,length)
    
    while l<length:
        ans.append(train.iloc[l:r])
        l=r
        r=min(r+batch_size,length)
    
    return ans

