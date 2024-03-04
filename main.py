import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast, GradScaler
from sklearn.decomposition import PCA
from torch.nn.parallel import DataParallel
#from logger import Logger
from dataset import load_dataset,rand_train
from eval import evaluate
from parse import parse_method, parser_add_main_args
from struct import pack
from myformer import *
from DifFormer import *
import warnings
warnings.filterwarnings('ignore')

from torch.utils.tensorboard import SummaryWriter

def read_fvecs(fname):
    #print(fname)
    x = np.memmap(fname, dtype='int32', mode='r')
    d = x[0]
    return x.view('float32').reshape(-1, d + 1)[:, 1:]

def write_fvecs(fname, vecs):
    print(fname)
    with open(fname, "wb") as f:
        for vec in vecs:
            dim = len(vec)
            f.write(pack('<i', dim))
            f.write(pack('f' * dim, *list(vec)))


# NOTE: for consistent data splits, see data_utils.rand_train_test_idx
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
### Parse args ###
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
print(args)

writer = SummaryWriter(log_dir=f"/home/sfy/study/myformer/model/summary_pic{args.id}")  
# 第二步，确定保存的路径，会保存一个文件夹而非文件
# tensorboard --logdir=summary_pic

fix_seed(args.seed)

device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
# device_list = list(map(int,(0,1)))
# print("Using gpu"," ".join([str(v) for v in device_list]))
# device = device_list[0]


mb_train,mb_val= load_dataset(args.data_dir)
mb_train.fillna(0, inplace=True)
mb_val.fillna(0, inplace=True)
print('train:',mb_train.shape)
print('val:',mb_val.shape)

pca = PCA(n_components=128)
d=128 #输入向量维度 输入层神经元个数
c=1 #输出层神经元个数  
model = parse_method(args, d, 1, device)
model.reset_parameters()
# model = DataParallel(model, device_ids=[0, 1])

l1_loss = nn.L1Loss()
smooth_l1_loss = nn.SmoothL1Loss()
alpha = 0.5
def combined_loss(pred, target):
    return alpha * l1_loss(pred, target) + (1 - alpha) * smooth_l1_loss(pred, target)

#回归任务对应的损失函数
#criterion=nn.MSELoss()
criterion=combined_loss
print('MODEL:', model)

# model.my_reset_parameters()
#SGD loss下降缓慢
optimizer = torch.optim.Adam(model.parameters(),weight_decay=args.weight_decay, lr=args.lr)
optimizer.zero_grad()
best_val = float('inf')

for epoch in range(args.epochs):
    model.train()
    LOSS=0
    cnt=0
    
    train=mb_train[['dataset','size','dim','graph_type','NN','ef_construction','k','recall']]
    train.graph_type = train.graph_type.apply(lambda x: 0 if x == 'nsw' else x)
    train.size = np.vectorize(lambda x: np.log10(x))(train.size)

    for iter in rand_train(train,args.batch_size):
        bases = iter.dataset.unique()
        L_OSS=0
        # print(bases)
        for dataset in bases:
            if int(dataset.split('_')[-1])>=50000:
                continue
            
            now_mb=iter[iter.dataset == dataset]
            # print(len(now_mb))
            
            now=now_mb[['size','dim','graph_type','NN','ef_construction','k']]
            target=now_mb[[args.target]]
            # print(target)
            now = torch.from_numpy(now.values).float()
            now=now.to(device)
            target = torch.from_numpy(target.values)
            # print(target)
            target=target.to(device)

            if os.path.exists('/home/sfy/study/myformer/data/'+dataset+'/'+dataset+'_base_128.fvecs'):
                node_feat=read_fvecs('/home/sfy/study/myformer/data/'+dataset+'/'+dataset+'_base_128.fvecs')
            else:
                node_feat=read_fvecs('/home/sfy/study/myformer/data/'+dataset+'/'+dataset+'_base.fvecs')
                if node_feat.shape[1] > 128:
                    node_feat = pca.fit_transform(node_feat)
                write_fvecs('/home/sfy/study/myformer/data/'+dataset+'/'+dataset+'_base_128.fvecs',node_feat)
            
            # print(node_feat.shape)
            node_feat=torch.Tensor(node_feat)
            node_feat=node_feat.to(device)

            adjs=None
            
            # print(edge_list)
            # print(edge_list.shape)
              
            # print(edge_list)
            # print(edge_list.shape)
            out = model(node_feat, now)
            
            out=out.squeeze(1)
            target=target.squeeze(1)
            target=target.to(torch.float)
            
            loss=criterion(out,target)
            loss.backward()
            
            L_OSS+=loss.detach().item()
        
            #是否使用梯度累积
        # loss.backward()
        optimizer.step()
        optimizer.zero_grad()    
        # print('梯度更新')
        
        L_OSS/=len(bases)
        LOSS+=L_OSS
        cnt+=1
    
    LOSS/=cnt
    print('epoch:',epoch,' train_loss:',LOSS)
    writer.add_scalar("train_loss", LOSS, epoch)
    
    LOSS = evaluate(model,mb_val,criterion,args,device)
    print('epoch:',epoch,' val_loss:',LOSS)
    writer.add_scalar("val_loss", LOSS, epoch)
    
    #在某个epoch结束后，更新保存的模型
    if (epoch+1)%10==0:
        print('全局平均val loss值为：',LOSS)
        if LOSS<best_val:
            best_val=LOSS
            print('模型更新')
            if args.save_model:
                torch.save(model.state_dict(), args.model_dir + f'summary_pic{args.id}/{args.method}_{args.id}_{args.target}.pkl')
writer.close()