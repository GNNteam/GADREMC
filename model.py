import torch
import torch.nn as nn
import torch.nn.functional as F

class GCN(nn.Module):
    #     """
    #     Forked from GRAND-Lab/CoLA
    #     """
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias

        return self.act(out)


# 读出（Readout）模块
# 读出模块用于将节点特征聚合为图的全局表示，支持多种聚合方式：
# AvgReadout：对节点特征取平均。
# MaxReadout：对节点特征取最大值。
# MinReadout：对节点特征取最小值。
# WSReadout：加权求和，权重通过节点特征的相似性计算。
class AvgReadout(nn.Module):
    """
    Forked from GRAND-Lab/CoLA
    """

    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq):
        return torch.mean(seq, 1)


class MaxReadout(nn.Module):
    """
    Forked from GRAND-Lab/CoLA
    """

    def __init__(self):
        super(MaxReadout, self).__init__()

    def forward(self, seq):
        return torch.max(seq, 1).values


class MinReadout(nn.Module):
    """
    Forked from GRAND-Lab/CoLA
    """

    def __init__(self):
        super(MinReadout, self).__init__()

    def forward(self, seq):
        return torch.min(seq, 1).values


class WSReadout(nn.Module):
    """
    Forked from GRAND-Lab/CoLA
    """

    def __init__(self):
        super(WSReadout, self).__init__()

    def forward(self, seq, query):
        query = query.permute(0, 2, 1)
        sim = torch.matmul(seq, query)
        sim = F.softmax(sim, dim=1)
        sim = sim.repeat(1, 1, 64)
        out = torch.mul(seq, sim)
        out = torch.sum(out, 1)
        return out


# 判别器（Discriminator）
# 判别器模块用于对比学习任务，判断特征是否来自同一分布：
# 使用双线性层（nn.Bilinear）计算特征之间的相似性。
# 支持负样本采样，通过循环移位生成负样本。
class Discriminator(nn.Module):
    """
    Forked from GRAND-Lab/CoLA
    """

    def __init__(self, n_h, negsamp_round):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

        self.negsamp_round = negsamp_round

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, s_bias1=None, s_bias2=None):
        scs = []
        scs.append(self.f_k(h_pl, c))
        c_mi = c
        for _ in range(self.negsamp_round):
            c_mi = torch.cat((c_mi[-1, :].unsqueeze(0), c_mi[:-1, :]), dim=0)
            scs.append(self.f_k(h_pl, c_mi))
        logits = torch.cat(tuple(scs))
        return logits


# 定义一个补丁判别器
class Patch_Discriminator(nn.Module):
    def __init__(self, n_h, negsamp_round):
        """
        初始化补丁判别器
        :param n_h: 隐藏层维度
        :param negsamp_round: 负样本采样轮数
        """
        super(Patch_Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)  # 双线性层
        for m in self.modules():
            self.weights_init(m)  # 初始化权重
        self.negsamp_round = negsamp_round  # 负样本采样轮数

    def weights_init(self, m):
        """
        权重初始化
        :param m: 模块
        """
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)  # 使用 Xavier 初始化权重
            if m.bias is not None:
                m.bias.data.fill_(0.0)  # 如果有偏置项，初始化为 0

    def forward(self, h_ano, h_unano, s_bias1=None, s_bias2=None):
        """
        前向传播
        :param h_ano: 异常特征
        :param h_unano: 非异常特征
        :param s_bias1: 负样本偏置 1
        :param s_bias2: 负样本偏置 2
        :return: 判别器输出
        """
        scs = []  # 保存所有相似度
        scs.append(self.f_k(h_unano, h_ano))  # 计算正样本的相似度
        h_mi = h_ano
        for _ in range(self.negsamp_round):
            h_mi = torch.cat((h_mi[-2:-1, :], h_mi[:-1, :]), 0)  # 生成负样本
            scs.append(self.f_k(h_unano, h_mi))  # 计算负样本的相似度
        logits = torch.cat(tuple(scs))  # 将所有相似度拼接
        return logits


class Model(nn.Module):
    def __init__(self, n_in, n_h, activation, negsamp_round_patch, negsamp_round_context, readout):
        super(Model, self).__init__()
        self.read_mode = readout
        self.activation = nn.PReLU() if activation == 'prelu' else activation

        # 编码器1：三个卷积层 + 一个线性层 + 激活函数
        self.conv1_enc1 = nn.Linear(n_in, n_h, bias=False)
        self.conv2_enc1 = nn.Linear(n_h, n_h, bias=False)
        self.conv3_enc1 = nn.Linear(n_h, n_h, bias=False)
        self.linear_enc1 = nn.Linear(n_h, n_h, bias=True)

        # 编码器2：一层GCN
        self.gcn_enc2 = GCN(n_in, n_h, activation)

        # 解码器：一层GCN
        self.gcn_dec = GCN(n_h, n_in, activation)

        # 读出模块
        if readout == 'max':
            self.read = MaxReadout()
        elif readout == 'min':
            self.read = MinReadout()
        elif readout == 'avg':
            self.read = AvgReadout()
        elif readout == 'weighted_sum':
            self.read = WSReadout()

        # 判别器模块
        self.disc1 = Discriminator(n_h, negsamp_round_context)
        self.disc2 = Discriminator(n_h, negsamp_round_context)
        self.disc3 = Patch_Discriminator(n_h, negsamp_round_patch)
        self.disc4 = Patch_Discriminator(n_h, negsamp_round_patch)
        self.pdist = nn.PairwiseDistance(p=2)

    def forward(self, seq1, seq2, seq3, seq4, adj1, adj2, sparse=False, msk=None, samp_bias1=None, samp_bias2=None):
        # 编码器2：处理 h_1
        h_1 = self.gcn_enc2(seq1, adj1, sparse)
        h_11 = self.gcn_enc2(seq1, adj1, sparse)
        h_2 = self.gcn_enc2(seq2, adj2, sparse)
        h_22 = self.gcn_enc2(seq1, adj1, sparse)
        # 编码器1：处理 h_3
        h_3 = self.encoder1(seq3, adj1, sparse)
        h_4 = self.encoder1(seq4, adj2, sparse)
        # 解码器部分：一层GCN
        f_1 = self.gcn_dec(h_3, adj1, sparse)
        f_2 = self.gcn_dec(h_4, adj2, sparse)
        # 读出部分
        if self.read_mode != 'weighted_sum':
            h_mv_1 = h_1[:, -1, :]
            c1 = self.read(h_1[:, :-1, :])
            h_unano1 = h_11[:, -1, :]  # 非异常特征
            h_ano1 = h_11[:, -2, :]  # 异常特征

            h_mv_2 = h_2[:, -1, :]
            c2 = self.read(h_2[:, :-1, :])
            h_unano2 = h_22[:, -1, :]  # 非异常特征
            h_ano2 = h_22[:, -2, :]  # 异常特征
        else:
            h_mv_1 = h_1[:, -1, :]
            c1 = self.read(h_1[:, :-1, :], h_1[:, -2:-1, :])
            h_unano1 = h_11[:, -1, :]  # 非异常特征
            h_ano1 = h_11[:, -2, :]  # 异常特征

            h_mv_2 = h_2[:, -1, :]
            c2 = self.read(h_2[:, :-1, :], h_2[:, -2:-1, :])
            h_unano2 = h_22[:, -1, :]  # 非异常特征
            h_ano2 = h_22[:, -2, :]  # 异常特征

        # 判别器部分

        ret1 = self.disc1(c1, h_mv_1, samp_bias1, samp_bias2)
        ret2 = self.disc2(c2, h_mv_2, samp_bias1, samp_bias2)
        ret3 = self.disc3(h_ano1, h_unano1, samp_bias1, samp_bias2)  # 补丁判别器输出
        ret4 = self.disc4(h_ano2, h_unano2, samp_bias1, samp_bias2)  # 补丁判别器输出

        return ret1, ret2, ret3, ret4, f_1, f_2

    def encoder1(self, seq, adj, sparse):
        # 第一层卷积
        seq_fts1 = self.conv1_enc1(seq)
        if sparse:
            seq_fts1 = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts1, 0)), 0)
        else:
            seq_fts1 = torch.bmm(adj, seq_fts1)

        # 第二层卷积
        seq_fts2 = self.conv2_enc1(seq_fts1)
        if sparse:
            seq_fts2 = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts2, 0)), 0)
        else:
            seq_fts2 = torch.bmm(adj, seq_fts2)

        # 第三层卷积
        seq_fts3 = self.conv3_enc1(seq_fts2)
        if sparse:
            seq_fts3 = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts3, 0)), 0)
        else:
            seq_fts3 = torch.bmm(adj, seq_fts3)

        # 线性层
        out = self.linear_enc1(seq_fts3)
        out = self.activation(out)
        return out

    def inference(self, seq1, seq2, seq3, seq4, adj1, adj2, sparse=False, samp_bias1=None, samp_bias2=None):
        # 编码器2：处理 h_1
        h_1 = self.gcn_enc2(seq1, adj1, sparse)
        h_11 = self.gcn_enc2(seq1, adj1, sparse)
        h_2 = self.gcn_enc2(seq2, adj2, sparse)
        h_22 = self.gcn_enc2(seq1, adj1, sparse)
        # 编码器1：处理 h_3
        h_3 = self.encoder1(seq3, adj1, sparse)
        h_4 = self.encoder1(seq4, adj2, sparse)
        # 解码器部分：一层GCN
        f_1 = self.gcn_dec(h_3, adj1, sparse)
        f_2 = self.gcn_dec(h_4, adj2, sparse)

        # 计算重建误差
        dist1 = self.pdist(f_1[:, -2, :], seq3[:, -1, :])
        dist2 = self.pdist(f_2[:, -2, :], seq4[:, -1, :])
        dist = 0.5 * (dist1 + dist2)

        # 读出部分
        if self.read_mode != 'weighted_sum':
            h_mv_1 = h_1[:, -1, :]
            c1 = self.read(h_1[:, :-1, :])
            h_unano1 = h_11[:, -1, :]  # 非异常特征
            h_ano1 = h_11[:, -2, :]  # 异常特征

            h_mv_2 = h_2[:, -1, :]
            c2 = self.read(h_2[:, :-1, :])
            h_unano2 = h_22[:, -1, :]  # 非异常特征
            h_ano2 = h_22[:, -2, :]  # 异常特征
        else:
            h_mv_1 = h_1[:, -1, :]
            c1 = self.read(h_1[:, :-1, :], h_1[:, -2:-1, :])
            h_unano1 = h_11[:, -1, :]  # 非异常特征
            h_ano1 = h_11[:, -2, :]  # 异常特征

            h_mv_2 = h_2[:, -1, :]
            c2 = self.read(h_2[:, :-1, :], h_2[:, -2:-1, :])
            h_unano2 = h_22[:, -1, :]  # 非异常特征
            h_ano2 = h_22[:, -2, :]  # 异常特征

        # 判别器部分

        ret1 = self.disc1(c1, h_mv_1, samp_bias1, samp_bias2)
        ret2 = self.disc2(c2, h_mv_2, samp_bias1, samp_bias2)
        ret3 = self.disc3(h_ano1, h_unano1, samp_bias1, samp_bias2)  # 补丁判别器输出
        ret4 = self.disc4(h_ano2, h_unano2, samp_bias1, samp_bias2)  # 补丁判别器输出

        return ret1, ret2, ret3, ret4, dist










