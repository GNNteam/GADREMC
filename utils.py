import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch
import torch.nn as nn
import scipy.io as sio
import random
import dgl

###############################################
# Forked from GRAND-Lab/CoLA                  #
###############################################

# 从文件中解析 Skipgram 格式的图数据，返回节点特征矩阵
def parse_skipgram(fname):
    with open(fname) as f:
        toks = list(f.read().split())
    nb_nodes = int(toks[0])
    nb_features = int(toks[1])
    ret = np.empty((nb_nodes, nb_features))
    it = 2
    for i in range(nb_nodes):
        cur_nd = int(toks[it]) - 1
        it += 1
        for j in range(nb_features):
            cur_ft = float(toks[it])
            ret[cur_nd][j] = cur_ft
            it += 1
    return ret

# 处理图数据集（如 TUDataset），将图数据转换为特征矩阵、邻接矩阵、标签等格式，便于后续处理
def process_tu(data, nb_nodes):
    nb_graphs = len(data)
    ft_size = data.num_features

    features = np.zeros((nb_graphs, nb_nodes, ft_size))
    adjacency = np.zeros((nb_graphs, nb_nodes, nb_nodes))
    labels = np.zeros(nb_graphs)
    sizes = np.zeros(nb_graphs, dtype=np.int32)
    masks = np.zeros((nb_graphs, nb_nodes))
       
    for g in range(nb_graphs):
        sizes[g] = data[g].x.shape[0]
        features[g, :sizes[g]] = data[g].x
        labels[g] = data[g].y[0]
        masks[g, :sizes[g]] = 1.0
        e_ind = data[g].edge_index
        coo = sp.coo_array((np.ones(e_ind.shape[1]), (e_ind[0, :], e_ind[1, :])), shape=(nb_nodes, nb_nodes))
        adjacency[g] = coo.todense()

    return features, adjacency, labels, sizes, masks

# 计算二分类或多分类任务的微 F1 分数，适用于图神经网络的分类任务评估。
def micro_f1(logits, labels):
    preds = torch.round(nn.Sigmoid()(logits))
    preds = preds.long()
    labels = labels.long()

    tp = torch.nonzero(preds * labels).shape[0] * 1.0
    tn = torch.nonzero((preds - 1) * (labels - 1)).shape[0] * 1.0
    fp = torch.nonzero(preds * (labels - 1)).shape[0] * 1.0
    fn = torch.nonzero((preds - 1) * labels).shape[0] * 1.0

    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = (2 * prec * rec) / (prec + rec)
    return f1

# 将邻接矩阵转换为偏置矩阵，用于图神经网络中的邻域聚合。
def adj_to_bias(adj, sizes, nhood=1):
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)


###############################################
# This section of code adapted from tkipf/gcn #
###############################################

# 解析索引文件，返回索引列表。
def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

# 根据索引生成掩码，用于划分训练集、验证集和测试集
def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

# 将稀疏矩阵转换为元组表示形式，便于 PyTorch 处理
def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""
    """Set insert_batch=True if you want to insert a batch dimension."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

# 对特征矩阵进行标准化处理，使其均值为 0，标准差为 1
def standardize_data(f, train_mask):
    """Standardize feature matrix and convert to tuple representation"""
    # standardize data
    f = f.todense()
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = f[:, np.squeeze(np.array(sigma > 0))]
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = (f - mu) / sigma
    return f

# 对特征矩阵进行行归一化处理，使其每一行的和为 1
def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)

# 对邻接矩阵进行对称归一化，常用于图神经网络的预处理
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

# 将邻接矩阵转换为稀疏矩阵的元组表示形式，便于 PyTorch 等框架使用
def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

# 将 SciPy 稀疏矩阵转换为 PyTorch 稀疏张量
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

# 将邻接矩阵转换为字典形式，表示每个节点的邻域节点，支持多跳扩展
def adj_to_dict(adj,hop=1,min_len=8):
    adj = np.array(adj.todense(),dtype=np.float64)
    num_node = adj.shape[0]
    # adj += np.eye(num_node)

    adj_diff = adj
    if hop > 1:
        for _ in range(hop - 1):
            adj_diff = adj_diff.dot(adj)


    dict = {}
    for i in range(num_node):
        dict[i] = []
        for j in range(num_node):
            if adj_diff[i,j] > 0:
                dict[i].append(j)

    final_dict = dict.copy()

    for i in range(num_node):
        while len(final_dict[i]) < min_len:
            final_dict[i].append(random.choice(dict[random.choice(dict[i])]))
    return dict

# 将类别标签从标量转换为独热编码向量
def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset+labels_dense.ravel()] = 1
    return labels_one_hot



# 合成数据集
# 加载 .mat 格式的图数据（如节点特征、邻接矩阵、标签等），并划分训练集、验证集和测试集
def load_mat(dataset, train_rate=0.3, val_rate=0.1):
    # data = sio.loadmat("./dataset/{}.mat".format(dataset))
    data = sio.loadmat("/home/sp429zxw/clw/dataset/citeseer.mat".format(dataset))
    label = data['Label'] if ('Label' in data) else data['gnd']
    # label=label.T
    attr = data['Attributes'] if ('Attributes' in data) else data['X']
    network = data['Network'] if ('Network' in data) else data['A']

    adj = sp.csr_matrix(network)
    feat = sp.lil_matrix(attr)

    labels = np.squeeze(np.array(data['Class'],dtype=np.int64) - 1)
    num_classes = np.max(labels) + 1
    labels = dense_to_one_hot(labels,num_classes)

    ano_labels = np.squeeze(np.array(label))
    if 'str_anomaly_label' in data:
        str_ano_labels = np.squeeze(np.array(data['str_anomaly_label']))
        attr_ano_labels = np.squeeze(np.array(data['attr_anomaly_label']))
    else:
        str_ano_labels = None
        attr_ano_labels = None

    num_node = adj.shape[0]
    num_train = int(num_node * train_rate)
    num_val = int(num_node * val_rate)
    all_idx = list(range(num_node))
    random.shuffle(all_idx)
    idx_train = all_idx[ : num_train]
    idx_val = all_idx[num_train : num_train + num_val]
    idx_test = all_idx[num_train + num_val : ]

    return adj, feat, labels, idx_train, idx_val, idx_test, ano_labels, str_ano_labels, attr_ano_labels



# 将邻接矩阵转换为 DGL（Deep Graph Library）图对象，便于使用 DGL 框架进行图神经网络训练。
def adj_to_dgl_graph(adj):
    nx_graph = nx.from_scipy_sparse_array(adj)
    dgl_graph = dgl.DGLGraph(nx_graph)
    return dgl_graph

# 基于随机游走（Random Walk with Restart）生成子图，用于图采样或子图训练
def generate_rwr_subgraph(dgl_graph, subgraph_size):
    all_idx = list(range(dgl_graph.number_of_nodes()))
    reduced_size = subgraph_size - 1
    traces = dgl.contrib.sampling.random_walk_with_restart(dgl_graph, all_idx, restart_prob=1.0,
                                                             max_nodes_per_seed=subgraph_size * 3)
    subv = []

    for i, trace in enumerate(traces):
        subv.append(torch.unique(torch.cat(trace), sorted=False).tolist())
        retry_time = 0
        while len(subv[i]) < reduced_size:
            cur_trace = dgl.contrib.sampling.random_walk_with_restart(dgl_graph, [i], restart_prob=0.9,
                                                                      max_nodes_per_seed=subgraph_size * 5)
            subv[i] = torch.unique(torch.cat(cur_trace[0]), sorted=False).tolist()
            retry_time += 1
            if (len(subv[i]) <= reduced_size) and (retry_time > 10):
                subv[i] = (subv[i] * reduced_size)

        subv[i] = subv[i][:reduced_size]
        subv[i].append(i)
    return subv



# 添加
# def sigmoid_loss(X, X_hat, reduction='mean'):
#     """
#     计算基于sigmoid激活函数的损失。

#     参数:
#     X -- 真实值，形状为 (n, d)，其中 n 是样本数，d 是特征维度。
#     X_hat -- 预测值，形状为 (n, d)。
#     reduction -- 指定应用于损失的 reduction，可以是 'none'、'mean' 或 'sum'。

#     返回:
#     损失值，根据 reduction 参数的不同，形状可能不同。
#     """
#     # 计算绝对误差
#     abs_diff = torch.abs(X - X_hat)

#     # 应用sigmoid函数
#     sigmoid = torch.sigmoid(abs_diff)

#     # 计算损失
#     loss = 1 - sigmoid

#     # 根据reduction参数应用不同的聚合方式
#     if reduction == 'mean':
#         return loss.mean()
#     elif reduction == 'sum':
#         return loss.sum()
#     elif reduction == 'none':
#         return loss
#     else:
#         raise ValueError("Invalid reduction. Choose from 'none', 'mean', or 'sum'.")
    # 添加
def sigmoid_loss(X, X_hat, reduction='mean'):
    """
    计算基于sigmoid激活函数的损失。

    参数:
    X -- 真实值，形状为 (n, d)，其中 n 是样本数，d 是特征维度。
    X_hat -- 预测值，形状为 (n, d)。
    reduction -- 指定应用于损失的 reduction，可以是 'none'、'mean' 或 'sum'。

    返回:
    损失值，根据 reduction 参数的不同，形状可能不同。
    """
    # 计算绝对误差并平方
    squared_abs_diff = torch.pow(torch.abs(X - X_hat), 2)

    # 应用sigmoid函数
    sigmoid = torch.sigmoid(squared_abs_diff)

    # 计算损失
    loss = 1 - sigmoid

    # 根据reduction参数应用不同的聚合方式
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError("Invalid reduction. Choose from 'none', 'mean', or 'sum'.")