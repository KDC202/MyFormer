import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from struct import pack

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

pca = PCA(n_components=128)

@torch.no_grad()
def evaluate(model,val,criterion,args,device):
    model.eval()
    
    L_OSS=0
    
    val=val[['dataset','size','dim','graph_type','NN','ef_construction','k','recall']]
    val.graph_type = val.graph_type.apply(lambda x: 0 if x == 'nsw' else x)
    val.size = np.vectorize(lambda x: np.log10(x))(val.size)

    bases = val.dataset.unique()
    for dataset in bases:
        if int(dataset.split('_')[-1])>=50000:
            continue
        
        now_mb=val[val.dataset == dataset]
        
        now=now_mb[['size','dim','graph_type','NN','ef_construction','k']]
        target=now_mb[[args.target]]
        
        now = torch.from_numpy(now.values).float()
        now=now.to(device)
        target = torch.from_numpy(target.values)
        target=target.to(device)

        if os.path.exists('/home/sfy/study/pgformer/data/'+dataset+'/'+dataset+'_base_128.fvecs'):
            node_feat=read_fvecs('/home/sfy/study/pgformer/data/'+dataset+'/'+dataset+'_base_128.fvecs')
        else:
            node_feat=read_fvecs('/home/sfy/study/pgformer/data/'+dataset+'/'+dataset+'_base.fvecs')
            if node_feat.shape[1] > 128:
                node_feat = pca.fit_transform(node_feat)
            write_fvecs('/home/sfy/study/pgformer/data/'+dataset+'/'+dataset+'_base_128.fvecs',node_feat)
        
        print(node_feat.shape)
        node_feat=torch.Tensor(node_feat)
        node_feat=node_feat.to(device)

        adjs=None
        out = model(node_feat, now,)
        
        out=out.squeeze(1)
        target=target.squeeze(1)
        target=target.to(torch.float)
        
        loss=criterion(out,target)
        
        L_OSS+=loss.detach().item()
    L_OSS/=len(bases)
    return L_OSS



# @torch.no_grad()
# def evaluate(model,val,criterion,args,device):
#     model.eval()
    
#     LOSS=0

#     val=val[['dataset','size','dim','graph_type','NN','ef_construction','k','recall']]
#     val.graph_type = val.graph_type.apply(lambda x: 0 if x == 'nsw' else x)
#     val=val.values
    
#     cnt=0
#     for row in val:
#         dataset=row[0]
#         condition=row[1:7]
#         target=row[7]
        
#         condition = torch.from_numpy(condition.astype(np.float32))
#         condition=condition.reshape(1,-1)
#         condition=condition.to(device)
        
#         if condition[0][0] >= 100000:
#             continue
        
#         if os.path.exists('/home/zjlab/tyk/pgformer/data/'+dataset+'/'+dataset+'_base_128.fvecs'):
#             node_feat=read_fvecs('/home/zjlab/tyk/pgformer/data/'+dataset+'/'+dataset+'_base_128.fvecs')
#         else:
#             node_feat=read_fvecs('/home/zjlab/tyk/pgformer/data/'+dataset+'/'+dataset+'_base.fvecs')
#             if node_feat.shape[1] > 128:
#                 node_feat = pca.fit_transform(node_feat)
#             write_fvecs('/home/zjlab/tyk/pgformer/data/'+dataset+'/'+dataset+'_base_128.fvecs',node_feat)
        
#         print(node_feat.shape)
#         node_feat=torch.Tensor(node_feat)
#         node_feat=node_feat.to(device)

#         adjs=None
#         out = model(node_feat, adjs, condition, args.tau)
        
#         out=out.squeeze(1)
#         target = torch.tensor([target])
        
#         #target=target.to(torch.float).to(device)
#         target=target.to(device)
#         loss=criterion(out,target)
        
#         cnt+=1
#         LOSS+=loss.detach().item()
#     LOSS/=cnt
#     return LOSS


# @torch.no_grad()
# def evaluate(model,val,bases,epoch,criterion,args,device):
#     model.eval()
#     #device=torch.device("cpu")
#     #model.to(device)
#     LOSS=0

#     for dataset in bases:
#         #使用每个数据集对应的test数据评估模型损失
#         now_mb=val[val.dataset == dataset]

#         now=now_mb[['dim','graph_type','NN','ef_construction','k']]
#         #now=now_mb[['size','graph_type','NN','k','R']]
#         now.graph_type = now.graph_type.apply(lambda x: 0 if x == 'nsw' else x)
#         #target=now_mb[['recall','querytime','distcomp']]
#         #target=now_mb[['recall']]
#         target=now_mb[[args.target]]

#         # now=now_mb[['M','efC','efS','K']]
#         # target=now_mb[['recall']]
        
#         array = now.values
#         now = torch.from_numpy(array)
#         now=now.to(device)

#         array = target.values
#         target = torch.from_numpy(array)
#         #target=target.to(device)

#         node_feat=read_fvecs('/home/zjlab/tyk/pgformer/data/'+dataset+'/'+dataset+'_base.fvecs')

#         if node_feat.shape[0] >= 100000:
#             continue
        
#         print(node_feat.shape)
#         if node_feat.shape[1] > 128:
#             node_feat = pca.fit_transform(node_feat)
        
#         #node_feat=node_feat[:80000]
#         #hidden_channels 128
#         #node_feat,_ =train_test_split(node_feat, test_size = 0.1)

#         print(node_feat.shape)
#         node_feat=torch.Tensor(node_feat)
#         node_feat=node_feat.to(device)

#         adjs=None
#         out = model(node_feat, adjs, now, args.tau)
#         out=out.squeeze(1)
#         target=target.squeeze(1)

#         target=target.to(torch.float).to(device)

#         # print(target)
#         # print(out)
#         val_loss = criterion(out,target)
#         LOSS+=val_loss.item()

#         print('val:',epoch,dataset)
        
#         if (epoch+1)%10==0:
#             print('epoch:',epoch)
#             print(out)
#             print('dataset:',dataset ,'  val loss:',val_loss)
#             print()
#     return LOSS

