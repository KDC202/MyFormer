from myformer import *
from DifFormer import *
#from data_utils import normalize

#根据程序指定参数 选择相应的图神经网络
def parse_method(args, d, c, device):
    # if args.method == 'link':
    #     model = LINK(n, c).to(device)
    # elif args.method == 'gcn':
    #     if args.dataset == 'ogbn-proteins':
    #         # Pre-compute GCN normalization.
    #         dataset.graph['edge_index'] = normalize(dataset.graph['edge_index'])
    #         model = GCN(in_channels=d,
    #                     hidden_channels=args.hidden_channels,
    #                     out_channels=c,
    #                     dropout=args.dropout,
    #                     save_mem=True,
    #                     use_bn=args.use_bn).to(device)
    #     else:
    #         model = GCN(in_channels=d,
    #                     hidden_channels=args.hidden_channels,
    #                     out_channels=c,
    #                     num_layers=args.num_layers,
    #                     dropout=args.dropout,
    #                     use_bn=args.use_bn).to(device)
    # elif args.method == 'mlp' or args.method == 'cs':
    #     model = MLP(in_channels=d, hidden_channels=args.hidden_channels,
    #                 out_channels=c, num_layers=args.num_layers,
    #                 dropout=args.dropout).to(device)
    # elif args.method == 'sgc':
    #     if args.cached:
    #         model = SGC(in_channels=d, out_channels=c, hops=args.hops).to(device)
    #     else:
    #         model = SGCMem(in_channels=d, out_channels=c,
    #                        hops=args.hops).to(device)
    # elif args.method == 'gprgnn':
    #     model = GPRGNN(d, args.hidden_channels, c, alpha=args.gpr_alpha).to(device)
    # elif args.method == 'appnp':
    #     model = APPNP_Net(d, args.hidden_channels, c, alpha=args.gpr_alpha).to(device)
    # elif args.method == 'gat':
    #     model = GAT(d, args.hidden_channels, c, num_layers=args.num_layers,
    #                 dropout=args.dropout, use_bn=args.use_bn, heads=args.gat_heads, out_heads=args.out_heads).to(device)
    # elif args.method == 'lp':
    #     mult_bin = args.dataset=='ogbn-proteins'
    #     model = MultiLP(c, args.lp_alpha, args.hops, mult_bin=mult_bin)
    # elif args.method == 'mixhop':
    #     model = MixHop(d, args.hidden_channels, c, num_layers=args.num_layers,
    #                    dropout=args.dropout, hops=args.hops).to(device)
    # elif args.method == 'gcnjk':
    #     model = GCNJK(d, args.hidden_channels, c, num_layers=args.num_layers,
    #                     dropout=args.dropout, jk_type=args.jk_type).to(device)
    # elif args.method == 'gatjk':
    #     model = GATJK(d, args.hidden_channels, c, num_layers=args.num_layers,
    #                     dropout=args.dropout, heads=args.gat_heads,
    #                     jk_type=args.jk_type).to(device)
    # elif args.method == 'h2gcn':
    #     model = H2GCN(d, args.hidden_channels, c, dataset.graph['edge_index'],
    #                     dataset.graph['num_nodes'],
    #                     num_layers=args.num_layers, dropout=args.dropout,
    #                     num_mlp_layers=args.num_mlp_layers).to(device)
    #调用nodeformer
    # if args.method == 'myformer':
    model=myformer(node_in_channels=d,
        graph_in_channels=6,
        dif_hidden_channels=512,
        hidden_channels=512,
        dif_out_channels=128,
        out_channels=1,
        num_layers=2,
        num_heads=8,
        alpha=0.5,
        dropout=0.1,
        use_bn=True,
        use_residual=True,
        use_graph=False,
        use_weight=True,
        kernel='simple',
        use_former=args.method
        ).to(device)
    # elif args.method == 'DIFFormer':
    #     model = DIFFormer(d, hidden_channels=args.hidden_channels, out_channels=c, num_layers=args.num_layers,
    #                       num_heads=args.num_heads, kernel='simple', alpha=0.5, dropout=args.dropout, use_bn=args.use_bn,
    #                       use_residual=args.use_residual,use_weight=True,use_graph=False).to(device)
    # else:
    #     raise ValueError('Invalid method')
    return model


def parser_add_main_args(parser):
    # dataset, protocol
    parser.add_argument('--id', type=int, default=34)
    parser.add_argument('--method', '-m', type=str, default='difformer')
    #parser.add_argument('--dataset', type=str, default='cora')
    #parser.add_argument('--sub_dataset', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='/home/sfy/study/myformer/data/train_nsw.csv')
    parser.add_argument('--target', type=str, default='recall')
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=360)

    
    # parser.add_argument('--eval_step', type=int,
    #                     default=1, help='how often to print')
    
    # parser.add_argument('--cpu', action='store_true')
    # parser.add_argument('--runs', type=int, default=1,
    #                     help='number of distinct runs')
    # parser.add_argument('--train_prop', type=float, default=.5,
    #                     help='training label proportion')
    # parser.add_argument('--valid_prop', type=float, default=.25,
    #                     help='validation label proportion')
    # parser.add_argument('--protocol', type=str, default='semi',
    #                     help='protocol for cora datasets with fixed splits, semi or supervised')
    # parser.add_argument('--rand_split', action='store_true', help='use random splits')
    # parser.add_argument('--rand_split_class', action='store_true',
    #                     help='use random splits with a fixed number of labeled nodes for each class')
    # parser.add_argument('--label_num_per_class', type=int, default=20, help='labeled nodes randomly selected')
    # parser.add_argument('--metric', type=str, default='acc', choices=['acc', 'rocauc', 'f1'],
    #                     help='evaluation metric')
    # parser.add_argument('--knn_num', type=int, default=3, help='number of k for KNN graph')
    parser.add_argument('--save_model', action='store_true', help='whether to save model',default='True')
    parser.add_argument('--model_dir', type=str, default='/home/sfy/study/myformer/model/')

    # hyper-parameter for model arch and training
    parser.add_argument('--hidden_channels', type=int, default=256)    #隐藏层神经元个数改变    1W 1024 显存15G
    
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers for deep methods')
    
    # hyper-parameter for nodeformer
    parser.add_argument('--num_heads', type=int, default=8)

    # 使用随机特征映射时的特征数量，通常用于近似核函数
    # parser.add_argument('--M', type=int, default=60, help='number of random features')

    #use_gumbel会使得显存占用增加 
    # parser.add_argument('--use_gumbel', action='store_true', help='use gumbel softmax for message passing',default='True')
    parser.add_argument('--use_residual', action='store_true', help='use residual link for each GNN layer', default='True')
    parser.add_argument('--use_bn', action='store_true', help='use layernorm', default='False')
    # parser.add_argument('--use_act', action='store_true', help='use non-linearity for each layer', default='True')
    # parser.add_argument('--use_jk', action='store_true', help='concat the layer-wise results in the final layer', default='True')

    # 影响nodeformer计算过程中矩阵维度，增加K会极大增加显存
    # parser.add_argument('--K', type=int, default=2, help='num of samples for gumbel softmax sampling')
    parser.add_argument('--tau', type=float, default=0.25, help='temperature for gumbel softmax')

    # parser.add_argument('--lamda', type=float, default=0.1, help='weight for edge reg loss')

    # 对输入图结构的操作
    # parser.add_argument('--rb_order', type=int, default=0, help='order for relational bias, 0 for not use')
    # parser.add_argument('--rb_trans', type=str, default='sigmoid', choices=['sigmoid', 'identity'],
    #                     help='non-linearity for relational bias')
    
    # parser.add_argument('--directed', action='store_true',
    #                      help='set to not symmetrize adjacency')

    # # hyper-parameter for gnn baseline
    # parser.add_argument('--hops', type=int, default=1,
    #                     help='power of adjacency matrix for certain methods')
    # parser.add_argument('--cached', action='store_true',
    #                     help='set to use faster sgc')
    # parser.add_argument('--gat_heads', type=int, default=8,
    #                     help='attention heads for gat')
    # parser.add_argument('--out_heads', type=int, default=1,
    #                     help='out heads for gat')
    # parser.add_argument('--projection_matrix_type', type=bool, default=True,
    #                     help='use projection matrix or not')
    # parser.add_argument('--lp_alpha', type=float, default=.1,
    #                     help='alpha for label prop')
    # parser.add_argument('--gpr_alpha', type=float, default=.1,
    #                     help='alpha for gprgnn')
    # parser.add_argument('--jk_type', type=str, default='max', choices=['max', 'lstm', 'cat'],
    #                     help='jumping knowledge type')
    # parser.add_argument('--num_mlp_layers', type=int, default=1,
    #                     help='number of mlp layers in h2gcn')




