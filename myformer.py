import math,os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import degree
from einops import rearrange, repeat
from sgformer import *
from difformer import *
from nodeformer import *

class myformer(nn.Module):
    def __init__(self,
                 node_in_channels=128,
                 graph_in_channels=6,
                 dif_hidden_channels=512,
                 hidden_channels=512,
                 dif_out_channels=128,
                 out_channels=1,
                 num_layers=2,
                 num_heads=8,
                 alpha=0.5,
                 dropout=0.5,
                 use_bn=True,
                 use_residual=True,
                 use_graph=False,
                 use_weight=True,
                 kernel='simple',
                 use_former='sgformer'
                ):
        super(myformer,self).__init__()
        # print("in_channels=",in_channels)
        # print("out_channels=",out_channels)
        # self.difformer = DIFFormer(node_in_channels,dif_hidden_channels, dif_out_channels, num_layers, alpha=alpha, dropout=dropout, num_heads=num_heads, kernel=kernel,
        #                use_bn=use_bn, use_residual=use_residual, use_graph=use_graph, use_weight=use_weight)
        if use_former == 'sgformer':
            self.Former = SGFormer(node_in_channels,dif_hidden_channels,dif_out_channels,num_layers,num_heads,dropout,gnn_use_weight=use_weight,
                                 gnn_use_init=False,gnn_use_bn=use_bn,gnn_use_residual=True,gnn_use_act=True,alpha=alpha,
                                 use_graph=use_graph)
            self.Former2 = SGFormer(dif_out_channels,dif_hidden_channels,dif_out_channels,1,num_heads,dropout,use_weight=use_weight,
                                 gnn_use_init=False,gnn_use_bn=use_bn,gnn_use_residual=True,gnn_use_act=True,alpha=alpha,
                                 use_graph=True)
        elif use_former == 'difformer':
            self.Former = DIFFormer(node_in_channels,dif_hidden_channels,dif_out_channels,num_layers,num_heads,dropout=dropout,use_weight=use_weight,
                                 use_bn=use_bn,use_residual=True,alpha=alpha,use_graph=False)
            self.Former2 = DIFFormer(node_in_channels,dif_hidden_channels,dif_out_channels,1,num_heads,dropout=dropout,use_weight=use_weight,
                                 use_bn=use_bn,use_residual=True,alpha=alpha,use_graph=True)
        elif use_former == 'nodeformer':
            # TODO: 添加nodeformer
            self.Former = NodeFormer(node_in_channels,dif_hidden_channels,dif_out_channels,num_layers,num_heads,dropout,kernel_transformation=softmax_kernel_transformation,
                                     nb_random_features=30,use_bn=use_bn,use_gumbel=True,use_residual=use_residual,use_act=False,use_jk=False,nb_gumbel_sample=10,rb_order=0,
                                     rb_trans='sigmoid',use_edge_loss=True)
        else:
            raise ValueError('Invalid method')
        
        self.fcs = nn.ModuleList()
        self.bns = nn.ModuleList()
        # self.fcs.append(nn.Linear(dif_out_channels+6, hidden_channels))
        # self.bns.append(nn.LayerNorm(dif_out_channels))
        # self.fcs.append(nn.Linear(hidden_channels, out_channels))
        # self.bns.append(nn.LayerNorm(hidden_channels))
        self.bns.append(nn.LayerNorm(dif_out_channels))
        self.fcs.append(nn.Linear(dif_out_channels, hidden_channels))
        self.bns.append(nn.LayerNorm(hidden_channels))
        self.fcs.append(nn.Linear(graph_in_channels, hidden_channels))
        self.bns.append(nn.LayerNorm(hidden_channels))
        
        self.fcs.append(nn.Linear(hidden_channels, hidden_channels))
        self.bns.append(nn.LayerNorm(hidden_channels))
        self.fcs.append(nn.Linear(hidden_channels, out_channels))
        # self.bns.append(nn.LayerNorm(out_channels))
        
        self.cls_token = nn.Parameter(torch.randn(1, dif_out_channels))					# nn.Parameter()定义可学习参数
        
        self.activation = F.relu
        self.use_bn = use_bn
        self.drop_out = dropout
    def forward(self, x, now):
        # x = x.unsqueeze(0) #B=1 denotes number of graph
        n = x.size(0)
        edge_list = torch.zeros((2, n), dtype=torch.int64)
        edge_list[0] = torch.zeros(n, dtype=torch.int64)
        edge_list[1] = torch.arange(1, n+1, dtype=torch.int64)
        # # print(edge_list)
        # # print("input node shape:",x.shape)
        # # print("input graph shape:",now.shape)
        # # x = torch.cat((self.cls_token, x), dim=0)
        # # print("拼接后",x.shape)
        x = self.Former(x,None)
        # if self.use_bn:
        #     x = self.bns[0](x)
        # x = self.activation(x)
        # x = F.dropout(x,p=self.drop_out,training=self.training)
        
        
        
        # x = x.mean(dim=1)
        # a = now.shape[0]
        # x = x.repeat(a,1)
        # x = torch.cat([x, now], dim=1)
        # print(x.shape)
        
        x = torch.cat((self.cls_token, x), dim=0)
        x = self.Former2(x,edge_list)
        # x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x,p=self.drop_out,training=self.training)
        
        x = x[0]
        
        
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[1](x)
        x = self.activation(x)
        x = F.dropout(x,p=self.drop_out,training=self.training)
        now = self.fcs[1](now)
        if self.use_bn:
            now = self.bns[2](now)
        now = self.activation(now)
        now = F.dropout(now,p=self.drop_out,training=self.training)
        
        x = self.fcs[2](x)
        if self.use_bn:
            x = self.bns[3](x)
        x = self.activation(x)
        x = F.dropout(x,p=self.drop_out,training=self.training)
        
        now = self.fcs[2](now)
        if self.use_bn:
            now = self.bns[3](now)
        now = self.activation(now)
        now = F.dropout(now,p=self.drop_out,training=self.training)
        repeat_x = x.repeat(now.size(0), 1)
        # print("x",repeat_x.shape)
        # print("now",now.shape)
        # additional_now = now.unsqueeze(1)
        y = repeat_x + now
        
        x_out = self.fcs[-1](y)
        x_out = F.sigmoid(x_out)
        # # print("after difformer node shape:",x.shape)
        # # x = torch.cat((self.cls_token, x), dim=0)
        # # x = self.sgFormer2(x,edge_list)
        # x = x[0]
        
        # x = self.fcs[0](x)
        # if self.use_bn:
        #     x = self.bns[0](x)
        # now = self.fcs[1](now)
        # if self.use_bn:
        #     now = self.bns[1](now)
            
        # # 特征融合
        # repeat_x = x.repeat(now.size(0), 1, 1)
        # additional_now = now.unsqueeze(1)
        # y = repeat_x + additional_now
        # # print("after merge:",y.shape)
        # # y = y.mean(dim=2)
        # # sum_tensor = torch.sum(y, dim=1)  # 在第二个维度上求和，结果大小为 (b, d)
        # # mean_tensor = sum_tensor / x.size(1) 
        # # print("after mean:",mean_tensor.shape)
        # result = self.fcs[-1](y)
        # if self.use_bn:
        #     result = self.bns[-1](result)
        return x_out
    
    def reset_parameters(self):
        # for conv in self.convs:
        #     conv.reset_parameters()
        self.Former.reset_parameters()
        self.Former2.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()