from model import Model
from utils import *
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import random
import os
import dgl
import argparse

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

parser = argparse.ArgumentParser(description='SL-GAD')


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# parser.add_argument('--expid', type=int)

parser.add_argument('--expid', type=int, default=1)
# parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--cuda', type=bool, default=True, help='Use CUDA if available')
parser.add_argument('--dataset', type=str, default='Citeseer')
# parser.add_argument('--lr', type=float)

parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--embedding_dim', type=int, default=64)
parser.add_argument('--patience', type=int, default=400)
# parser.add_argument('--num_epoch', type=int)

parser.add_argument('--num_epoch', type=int, default=100)
parser.add_argument('--drop_prob', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=300)
parser.add_argument('--subgraph_size', type=int, default=4)
parser.add_argument('--readout', type=str, default='avg')
parser.add_argument('--auc_test_rounds', type=int, default=256)
# parser.add_argument('--negsamp_ratio', type=int, default=1)
parser.add_argument('--negsamp_ratio_patch', type=int, default=1)
parser.add_argument('--negsamp_ratio_context', type=int, default=1)
parser.add_argument('--alpha', type=float, default=0.8)
parser.add_argument('--beta', type=float, default=0.6)
parser.add_argument('--gamma', type=float, default=0.2)

args = parser.parse_args()

# 这段代码是 run.py 的核心部分，负责：
# 设置实验环境（设备、随机种子等）。
# 加载和预处理数据。
# 初始化模型、优化器和损失函数。
# 准备多次运行的基础设施，确保实验的可重复性和稳定性。
# 为训练过程（早停机制）和测试过程（AUC 评估）做好准备。
if __name__ == '__main__':
    # 实验参数检查与设备设置
    assert args.expid is not None, "experiment id needs to be assigned."
    device = torch.device("cuda:1" if args.cuda else "cpu")
    print(device)
    print('Dataset: {}'.format(args.dataset), flush=True)
    seeds = [i + 1 for i in range(args.runs)]

    # 学习率和训练轮数的默认设置
    if args.lr is None:
        if args.dataset in ['cora', 'citeseer', 'pubmed', 'Flickr']:
            args.lr = 1e-3
        elif args.dataset == 'ACM':
            args.lr = 1e-3
        elif args.dataset == 'BlogCatalog':
            args.lr = 3e-3

    if args.num_epoch is None:
        if args.dataset in ['cora', 'citeseer', 'pubmed']:
            args.num_epoch = 100
        elif args.dataset in ['BlogCatalog', 'Flickr', 'ACM']:
            args.num_epoch = 400

    batch_size = args.batch_size
    subgraph_size = args.subgraph_size

    # 数据加载与预处理
    adj, features, labels, idx_train, idx_val, \
        idx_test, ano_label, str_ano_label, attr_ano_label = load_mat(args.dataset)

    raw_features = features.todense()
    features, _ = preprocess_features(features)

    dgl_graph = adj_to_dgl_graph(adj)

    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    nb_classes = labels.shape[1]

    adj = normalize_adj(adj)
    adj = (adj + sp.eye(adj.shape[0])).todense()

    # 数据和模型的设备转换
    features = torch.FloatTensor(features[np.newaxis]).to(device)
    raw_features = torch.FloatTensor(raw_features[np.newaxis]).to(device)
    adj = torch.FloatTensor(adj[np.newaxis]).to(device)
    labels = torch.FloatTensor(labels[np.newaxis]).to(device)
    idx_train = torch.LongTensor(idx_train).to(device)
    idx_val = torch.LongTensor(idx_val).to(device)
    idx_test = torch.LongTensor(idx_test).to(device)

    # 多次运行的准备
    all_auc = []
    for run in range(args.runs):
        # setup seed
        seed = seeds[run]
        print('\n# Run:{}'.format(run), flush=True)
        # 随机种子固定
        dgl.random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['OMP_NUM_THREADS'] = '1'
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # 模型初始化与优化器配置
        model = Model(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio_patch, args.negsamp_ratio_context,
                      args.readout).to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # 损失函数定义
        # b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]).to(device))
        # 定义两种损失函数
        b_xent_patch = nn.BCEWithLogitsLoss(reduction='none',
                                            pos_weight=torch.tensor([args.negsamp_ratio_patch]).to(device))
        b_xent_context = nn.BCEWithLogitsLoss(reduction='none',
                                              pos_weight=torch.tensor([args.negsamp_ratio_context]).to(device))
        mse_loss = nn.MSELoss(reduction='mean')

        # 早停机制初始化
        cnt_wait = 0
        best = 1e9
        best_t = 0
        batch_num = nb_nodes // batch_size + 1

        # 训练过程
        # 模型初始化：
        # 初始化Model，并将其移至指定设备。
        # 使用Adam优化器进行优化。
        # 损失函数：
        # 对比学习损失：使用BCEWithLogitsLoss计算对比学习任务的损失。
        # 特征重建损失：使用MSELoss计算重建特征与原始特征之间的均方误差。
        # 总损失：结合对比学习损失和特征重建损失，通过超参数alpha和beta调整权重。
        # 训练过程：
        # 对每个epoch，随机打乱节点顺序，生成子图。
        # 使用随机游走（generate_rwr_subgraph）生成子图。
        # 对每个子图，计算对比学习损失和特征重建损失，并进行反向传播。
        # 早停机制：
        # 如果连续patience轮训练未改善损失，则提前终止训练。
        for epoch in range(args.num_epoch):
            model.train()
            all_idx = list(range(nb_nodes))
            random.shuffle(all_idx)
            total_loss = 0.

            subgraphs_1 = generate_rwr_subgraph(dgl_graph, subgraph_size)
            subgraphs_2 = generate_rwr_subgraph(dgl_graph, subgraph_size)

            for batch_idx in range(batch_num):
                optimiser.zero_grad()

                is_final_batch = (batch_idx == (batch_num - 1))
                if not is_final_batch:
                    idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                else:
                    idx = all_idx[batch_idx * batch_size:]

                cur_batch_size = len(idx)

                # lbl = torch.unsqueeze(torch.cat((torch.ones(cur_batch_size),
                #                                  torch.zeros(cur_batch_size * args.negsamp_ratio))), 1).to(device)
                # 创建标签张量
                lbl_patch = torch.unsqueeze(torch.cat(
                    (torch.ones(cur_batch_size), torch.zeros(cur_batch_size * args.negsamp_ratio_patch))), 1).to(device)
                lbl_context = torch.unsqueeze(torch.cat(
                    (torch.ones(cur_batch_size), torch.zeros(cur_batch_size * args.negsamp_ratio_context))), 1).to(
                    device)

                ba1 = []
                ba2 = []
                bf1 = []
                bf2 = []
                raw_bf1 = []
                raw_bf2 = []
                added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size)).to(device)
                added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1)).to(device)
                added_adj_zero_col[:, -1, :] = 1.
                added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size)).to(device)

                for i in idx:
                    cur_adj_1 = adj[:, subgraphs_1[i], :][:, :, subgraphs_1[i]]
                    cur_feat_1 = features[:, subgraphs_1[i], :]
                    raw_cur_feat_1 = raw_features[:, subgraphs_1[i], :]
                    cur_adj_2 = adj[:, subgraphs_2[i], :][:, :, subgraphs_2[i]]
                    cur_feat_2 = features[:, subgraphs_2[i], :]
                    raw_cur_feat_2 = raw_features[:, subgraphs_2[i], :]
                    ba1.append(cur_adj_1)
                    bf1.append(cur_feat_1)
                    raw_bf1.append(raw_cur_feat_1)
                    ba2.append(cur_adj_2)
                    bf2.append(cur_feat_2)
                    raw_bf2.append(raw_cur_feat_2)

                ba1 = torch.cat(ba1)
                ba1 = torch.cat((ba1, added_adj_zero_row), dim=1)
                ba1 = torch.cat((ba1, added_adj_zero_col), dim=2)
                ba2 = torch.cat(ba2)
                ba2 = torch.cat((ba2, added_adj_zero_row), dim=1)
                ba2 = torch.cat((ba2, added_adj_zero_col), dim=2)

                bf1 = torch.cat(bf1)
                bf1 = torch.cat((bf1[:, :-1, :], added_feat_zero_row, bf1[:, -1:, :]), dim=1)
                bf2 = torch.cat(bf2)
                bf2 = torch.cat((bf2[:, :-1, :], added_feat_zero_row, bf2[:, -1:, :]), dim=1)

                raw_bf1 = torch.cat(raw_bf1)
                raw_bf1 = torch.cat((raw_bf1[:, :-1, :], added_feat_zero_row, raw_bf1[:, -1:, :]), dim=1)
                raw_bf2 = torch.cat(raw_bf2)
                raw_bf2 = torch.cat((raw_bf2[:, :-1, :], added_feat_zero_row, raw_bf2[:, -1:, :]), dim=1)

                logits_1, logits_2, logits_3, logits_4, f_1, f_2 = model(bf1, bf2, raw_bf1, raw_bf2, ba1, ba2)
                # lbl = lbl.float()   # 添加
                logits_12 = 0.5 * (logits_1 + logits_2)
                logits_34 = 0.5 * (logits_3 + logits_4)

                # 计算两种损失
                loss_all_1 = b_xent_context(logits_12, lbl_context)  # 上下文级损失
                loss_1 = torch.mean(loss_all_1)
                loss_all_2 = b_xent_patch(logits_34, lbl_patch)  # 补丁级损失
                loss_2 = torch.mean(loss_all_2)

                loss_3 = 0.5 * (mse_loss(f_1[:, -2, :], raw_bf1[:, -1, :]) + mse_loss(f_2[:, -2, :], raw_bf2[:, -1, :]))
                loss = args.alpha * loss_1 + args.gamma * loss_2 + args.beta * loss_3

                loss.backward()
                optimiser.step()

                loss = loss.detach().cpu().numpy()

                if not is_final_batch:
                    total_loss += loss

            mean_loss = (total_loss * batch_size + loss * cur_batch_size) / nb_nodes

            if mean_loss < best:
                best = mean_loss
                best_t = epoch
                cnt_wait = 0
                torch.save(model.state_dict(), '/home/sp429zxw/clw/checkpoints/exp_{}.pkl'.format(args.expid))
            else:
                cnt_wait += 1

            if cnt_wait == args.patience:
                print('Early stopping!', flush=True)
                break

            print('Epoch:{} Loss:{:.8f}'.format(epoch, mean_loss), flush=True)

        print('Loading {}th epoch'.format(best_t), flush=True)
        model.load_state_dict(torch.load('/home/sp429zxw/clw/checkpoints/exp_{}.pkl'.format(args.expid)))

        # 测试过程
        # 异常分数计算：
        # 使用训练好的模型对每个节点计算异常分数。
        # 支持多轮测试（auc_test_rounds），取平均值作为最终异常分数。
        # 异常分数由对比学习输出和特征重建误差组成，通过alpha和beta调整权重。
        # 性能评估：
        # 使用AUC（ROC曲线下面积）评估异常检测性能。
        # 输出每轮测试的AUC和最终平均AUC。
        multi_round_ano_score = np.zeros((args.auc_test_rounds, nb_nodes))

        print('Testing AUC!', flush=True)

        for round in range(args.auc_test_rounds):

            all_idx = list(range(nb_nodes))
            random.shuffle(all_idx)

            subgraphs_1 = generate_rwr_subgraph(dgl_graph, subgraph_size)
            subgraphs_2 = generate_rwr_subgraph(dgl_graph, subgraph_size)

            for batch_idx in range(batch_num):

                optimiser.zero_grad()

                is_final_batch = (batch_idx == (batch_num - 1))

                if not is_final_batch:
                    idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                else:
                    idx = all_idx[batch_idx * batch_size:]

                cur_batch_size = len(idx)

                ba1 = []
                ba2 = []
                bf1 = []
                bf2 = []
                raw_bf1 = []
                raw_bf2 = []
                added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size)).to(device)
                added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1)).to(device)
                added_adj_zero_col[:, -1, :] = 1.
                added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size)).to(device)

                for i in idx:
                    cur_adj_1 = adj[:, subgraphs_1[i], :][:, :, subgraphs_1[i]]
                    cur_adj_2 = adj[:, subgraphs_2[i], :][:, :, subgraphs_2[i]]
                    cur_feat_1 = features[:, subgraphs_1[i], :]
                    cur_feat_2 = features[:, subgraphs_2[i], :]
                    raw_cur_feat_1 = raw_features[:, subgraphs_1[i], :]
                    raw_cur_feat_2 = raw_features[:, subgraphs_2[i], :]
                    ba1.append(cur_adj_1)
                    bf1.append(cur_feat_1)
                    ba2.append(cur_adj_2)
                    bf2.append(cur_feat_2)
                    raw_bf1.append(raw_cur_feat_1)
                    raw_bf2.append(raw_cur_feat_2)

                ba1 = torch.cat(ba1)
                ba1 = torch.cat((ba1, added_adj_zero_row), dim=1)
                ba1 = torch.cat((ba1, added_adj_zero_col), dim=2)
                ba2 = torch.cat(ba2)
                ba2 = torch.cat((ba2, added_adj_zero_row), dim=1)
                ba2 = torch.cat((ba2, added_adj_zero_col), dim=2)

                bf1 = torch.cat(bf1)
                bf1 = torch.cat((bf1[:, :-1, :], added_feat_zero_row, bf1[:, -1:, :]), dim=1)
                bf2 = torch.cat(bf2)
                bf2 = torch.cat((bf2[:, :-1, :], added_feat_zero_row, bf2[:, -1:, :]), dim=1)

                raw_bf1 = torch.cat(raw_bf1)
                raw_bf1 = torch.cat((raw_bf1[:, :-1, :], added_feat_zero_row, raw_bf1[:, -1:, :]), dim=1)
                raw_bf2 = torch.cat(raw_bf2)
                raw_bf2 = torch.cat((raw_bf2[:, :-1, :], added_feat_zero_row, raw_bf2[:, -1:, :]), dim=1)

                with torch.no_grad():
                    logits_1, logits_2, logits_3, logits_4, dist = model.inference(bf1, bf2, raw_bf1, raw_bf2, ba1, ba2)
                    logits_1 = torch.sigmoid(torch.squeeze(logits_1))
                    logits_2 = torch.sigmoid(torch.squeeze(logits_2))
                    logits_3 = torch.sigmoid(torch.squeeze(logits_3))
                    logits_4 = torch.sigmoid(torch.squeeze(logits_4))
                    logits_12 = 0.5 * (logits_1 + logits_2)
                    logits_34 = 0.5 * (logits_3 + logits_4)

                # 修改后的异常分数计算部分
                with torch.no_grad():
                    logits_1, logits_2, logits_3, logits_4, dist = model.inference(bf1, bf2, raw_bf1, raw_bf2, ba1, ba2)
                    logits_1 = torch.sigmoid(torch.squeeze(logits_1))
                    logits_2 = torch.sigmoid(torch.squeeze(logits_2))
                    logits_3 = torch.sigmoid(torch.squeeze(logits_3))
                    logits_4 = torch.sigmoid(torch.squeeze(logits_4))
                    logits_12 = 0.5 * (logits_1 + logits_2)  # 上下文级分数平均
                    logits_34 = 0.5 * (logits_3 + logits_4)  # 补丁级分数平均

                # 初始化三个分数分量
                ano_score_context = np.zeros(cur_batch_size)
                ano_score_patch = np.zeros(cur_batch_size)
                ano_score_recon = dist.cpu().numpy()

                # 计算上下文级异常分数
                if args.alpha != 0.0:
                    if args.negsamp_ratio_context == 1:
                        ano_score_context = - (logits_12[:cur_batch_size] - logits_12[cur_batch_size:]).cpu().numpy()
                    else:
                        pos_score = logits_12[:cur_batch_size]
                        neg_score = logits_12[cur_batch_size:].view(cur_batch_size, args.negsamp_ratio_context).mean(
                            dim=1)
                        ano_score_context = - (pos_score - neg_score).cpu().numpy()

                # 计算补丁级异常分数
                if args.gamma != 0.0:
                    if args.negsamp_ratio_patch == 1:
                        ano_score_patch = - (logits_34[:cur_batch_size] - logits_34[cur_batch_size:]).cpu().numpy()
                    else:
                        pos_score = logits_34[:cur_batch_size]
                        neg_score = logits_34[cur_batch_size:].view(cur_batch_size, args.negsamp_ratio_patch).mean(
                            dim=1)
                        ano_score_patch = - (pos_score - neg_score).cpu().numpy()

                # 归一化各分数分量
                if args.alpha != 0.0 or args.gamma != 0.0 or args.beta != 0.0:
                    scaler = MinMaxScaler()
                    if args.alpha != 0.0:
                        ano_score_context = scaler.fit_transform(ano_score_context.reshape(-1, 1)).reshape(-1)
                    if args.gamma != 0.0:
                        ano_score_patch = scaler.fit_transform(ano_score_patch.reshape(-1, 1)).reshape(-1)
                    if args.beta != 0.0:
                        ano_score_recon = scaler.fit_transform(ano_score_recon.reshape(-1, 1)).reshape(-1)

                # 组合最终异常分数
                ano_score = (args.alpha * ano_score_context +
                             args.gamma * ano_score_patch +
                             args.beta * ano_score_recon)

                # 确保至少有一个分数分量被使用
                if args.alpha == 0.0 and args.gamma == 0.0 and args.beta == 0.0:
                    raise ValueError("At least one of alpha, gamma or beta should be non-zero")

                multi_round_ano_score[round, idx] = ano_score

        ano_score_final = np.mean(multi_round_ano_score, axis=0) + np.std(multi_round_ano_score, axis=0)
        auc = roc_auc_score(ano_label, ano_score_final)
        all_auc.append(auc)
        print('Testing AUC:{:.4f}'.format(auc), flush=True)

    print('\n==============================')
    print(all_auc)
    print('FINAL TESTING AUC:{:.4f}'.format(np.mean(all_auc)))
    print('==============================')