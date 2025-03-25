import importlib
import pickle
import sys
import time
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster import hierarchy

import config as CFG
from utils import LogPrint, get_R, cluster
from dataset import CLIPDataset
from torch.utils.data import DataLoader

import os
import numpy as np
import scanpy as sc
import argparse

parser = argparse.ArgumentParser(description='inference for CLIP')
parser.add_argument('--path_dir', type=str, default='ours', help='BLEEP ours')  # modify
parser.add_argument('--pt_path', type=str, default='./result/ours/ours-her2st-best.pt', help='')  # modify
parser.add_argument('--aggregation_method', type=str, default='average', help='weighted_average average or simple')  # modify
parser.add_argument('--top_K', type=int, default=100, help='1 50 or 100')  # modify
parser.add_argument('--image_encoder', type=str, default='densenet121', help='')  # modify

# dataset her2st
parser.add_argument('--dataset', type=str, default='her2st', help='dataset name:{"GSE240429", "her2st"}.')  # modify
parser.add_argument('--fold', type=int, default=33, help='0 6 12 18 24 27 31 33')  # modify
parser.add_argument('--neighbor', type=int, default=4, help='the number of neighbors in the GNN.')
parser.add_argument('--prune', type=str, default='Grid', help='how to prune the edge:{"Grid","NA"}')

args = parser.parse_args()

if 'checkpoint' in args.path_dir:
    PATH = args.path_dir
else:
    PATH = './result/' + args.path_dir + '/'
LOG = LogPrint(PATH)
# LOG = LogPrint(PATH + 'inference_')

from datetime import datetime
datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
LOG.logging(f'\n\n------------------------  datetime: {datetime}  ------------------------\n')

if args.aggregation_method == "simple":
    args.top_K = 1
    LOG.logging("find args.aggregation_method==simple, set args.top_K=1")

LOG.logging("current model: %s" % PATH)
LOG.logging("current pt_path: %s" % args.pt_path)
LOG.logging("current aggregation_method: %s" % args.aggregation_method)
LOG.logging("current top_K: %s" % args.top_K)
LOG.logging("current dataset: %s" % str(args.dataset))
PTID = args.pt_path.split("/")[-1].split(".")[0]
PATH = PATH + PTID + "/"
if not os.path.exists(PATH) and '*' not in PTID:
    os.makedirs(PATH)


# her2st
class her2st:
    label = None
    pred = None
    gt = None
    ct = None
    img_embeddings_test = None
    spot_embeddings_test = None
    img_embeddings_train = None
    spot_embeddings_train = None
    gt_train = None


# print the current scanpy version
# LOG.logging(sc.__version__)

def visualize_umap_clusters(expr_matrix, preprocess=True, normalize_and_log=True, batch_idx=None, n_neighbors=150,
                            n_top_genes=1024, max_value=10, legend_loc='on data', show=False, save=True,
                            save_name=f'{PATH}/umap_clusters.pdf'):
    # ! scanpy input matrix has cells as rows and genes as columns, same as this function
    if preprocess:
        # # Filter out genes with expression in less than 50 spots (for a ~8000 spot dataset over 4 slices)
        expressed = np.sum(expr_matrix>0, axis=0)
        expr_matrix = expr_matrix[:,expressed>50]

        # Create AnnData object with batch index as an observation
        adata = sc.AnnData(X=expr_matrix, dtype=expr_matrix.dtype)
        if batch_idx is not None:
            adata.obs['batch_idx'] = batch_idx

        # Preprocess the data
        if normalize_and_log:
            sc.pp.filter_cells(adata, min_genes=200)
            sc.pp.filter_genes(adata, min_cells=5)
            # sc.pp.normalize_total(adata, target_sum=1e4)
            # adata.X[adata.X <= 0] = 1e-10
            sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
        LOG.logging("n_top_genes: %s" % str(adata.var['highly_variable'].sum()))
        # sc.pp.scale(adata, max_value=max_value)
    else:
        adata = sc.AnnData(X=expr_matrix, dtype=expr_matrix.dtype)
        if batch_idx is not None:
            adata.obs['batch_idx'] = batch_idx

    # Run UMAP and clustering on the preprocessed data
    # sc.pp.scale(adata, max_value=max_value)
    sc.pp.pca(adata, n_comps=50, use_highly_variable=preprocess)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=50)

    sc.tl.umap(adata)
    LOG.logging("Running Leiden clustering")
    sc.tl.leiden(adata)
    LOG.logging("n_clusters: %s" % str(adata.obs['leiden'].nunique()))
    LOG.logging("Plotting UMAP clusters")

    # Plot the UMAP embedding with cell clusters and batch index
    if batch_idx is None:
        fig, ax = plt.subplots(figsize=(6, 5))
        sc.pl.umap(adata, color='leiden', ax=ax, show=show, legend_loc=legend_loc)

        # Save the figure
        if save:
            fig.savefig(save_name, dpi=300, bbox_inches='tight')
        plt.close(fig)

        return adata

    else:
        fig, axs = plt.subplots(ncols=2, figsize=(10, 5))

        # Plot the UMAP embedding with cell clusters
        sc.pl.umap(adata, color='leiden', ax=axs[0], show=False, legend_loc=legend_loc)

        # Plot the UMAP embedding with batch information
        sc.pl.umap(adata, color='batch_idx', ax=axs[1], show=False, legend_loc=legend_loc)

        # Save the figure
        if save:
            fig.savefig(save_name, dpi=300, bbox_inches='tight')
        plt.close(fig)

        return adata


def gene_count(true_data, pred_data, save_path):
    # 假设 real_gene_expression 和 predicted_gene_expression 是你的两个数据集
    # 这里我们用随机数据来模拟这两个数据集
    real_gene_expression = true_data
    predicted_gene_expression = pred_data

    # 计算每个基因的总表达量
    real_gene_sums = real_gene_expression.sum(axis=0)
    predicted_gene_sums = predicted_gene_expression.sum(axis=0)

    # 计算每个基因的表达比例
    real_gene_proportions = real_gene_sums / real_gene_sums.sum()
    predicted_gene_proportions = predicted_gene_sums / predicted_gene_sums.sum()

    # 对基因计数比例进行升序排列，以真实的gene进行升序排列
    sorted_indices = np.argsort(real_gene_proportions)
    real_gene_proportions_sorted = real_gene_proportions[sorted_indices]
    predicted_gene_proportions_sorted = predicted_gene_proportions[sorted_indices]
    real_gene_proportions_sorted = np.clip(real_gene_proportions_sorted, None, 0.003)
    predicted_gene_proportions_sorted = np.clip(predicted_gene_proportions_sorted, None, 0.003)

    # 设置图表大小
    fig, ax = plt.subplots(figsize=(10, 8))

    # 绘制真实基因表达的基因计数比例散点图
    ax.scatter(range(len(real_gene_proportions_sorted)), real_gene_proportions_sorted,
                label='Real Gene Counts Proportion', color='blue', s=0.02)

    # 绘制预测基因表达的基因计数比例散点图
    ax.scatter(range(len(predicted_gene_proportions_sorted)), predicted_gene_proportions_sorted,
                label='Predicted Gene Counts Proportion', color='red', s=0.08)

    # 设置y轴的范围
    ax.set_ylim(-0.0002, 0.0032)
    # 自定义y轴最上面的刻度标签为"≥0.003"
    ax.set_yticks([0, 0.001, 0.002, 0.003])
    ax.set_yticklabels(["0.000", "0.001", "0.002", "≥0.003"])

    # 设置标题和坐标轴标签
    ax.set_xlabel('Ranked Gene')
    ax.set_ylabel('Gene Counts Proportion')

    ax.legend(loc='upper left', markerscale=20)

    ax_inset = fig.add_axes([0.18, 0.37, 0.5, 0.4])  # [left, bottom, width, height]
    ax_inset.scatter(range(len(real_gene_proportions_sorted)), real_gene_proportions_sorted, color='blue', s=0.02)
    ax_inset.scatter(range(len(predicted_gene_proportions_sorted)), predicted_gene_proportions_sorted, color='red', s=0.08)
    x1, x2 = 2100, 3200
    ax_inset.set_xlim(x1, x2)
    ax_inset.set_ylim(predicted_gene_proportions_sorted[x1], predicted_gene_proportions_sorted[x2])
    ax_inset.set_xticks([])  # 移除x轴刻度
    ax_inset.set_yticks([])  # 移除y轴刻度

    for spine in ax_inset.spines.values():
        spine.set_edgecolor('purple')  # 设置边框颜色为紫色
        spine.set_linewidth(2)  # 设置边框线宽
        spine.set_linestyle('--')

    rect = plt.Rectangle((x1, predicted_gene_proportions_sorted[x1]), x2 - x1, predicted_gene_proportions_sorted[x2]-predicted_gene_proportions_sorted[x1],
                         fill=False, edgecolor='purple', lw=2, ls='--')
    ax.add_patch(rect)

    # 添加注释，指向放大区域
    # ax.annotate('', xy=(x2, 0.0025), xytext=(x2, 0.0025),
    #             xycoords='data', textcoords='data',
    #             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

    # save
    # plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path, bbox_inches='tight', dpi=500)
    # 显示图表
    # plt.show()


def gene_variation(true_data, pred_data, save_path):
    # 假设 real_gene_expression 和 predicted_gene_expression 是你的两个数据集
    # 这里我们用随机数据来模拟这两个数据集
    real_gene_expression = true_data
    predicted_gene_expression = pred_data

    # 计算每个基因的标准差
    real_gene_std = np.std(real_gene_expression, axis=0)
    predicted_gene_std = np.std(predicted_gene_expression, axis=0)

    # 计算标准差的总和
    real_std_sum = np.sum(real_gene_std)
    predicted_std_sum = np.sum(predicted_gene_std)

    # 计算每个基因的标准差比例
    real_gene_variation_proportions = real_gene_std / real_std_sum
    predicted_gene_variation_proportions = predicted_gene_std / predicted_std_sum

    # 对基因变化比例进行升序排列，以真实的gene进行升序排列
    sorted_indices = np.argsort(real_gene_variation_proportions)
    real_gene_variation_proportions_sorted = real_gene_variation_proportions[sorted_indices]
    predicted_gene_variation_proportions_sorted = predicted_gene_variation_proportions[sorted_indices]
    real_gene_variation_proportions_sorted = np.clip(real_gene_variation_proportions_sorted, None, 0.003)
    predicted_gene_variation_proportions_sorted = np.clip(predicted_gene_variation_proportions_sorted, None, 0.003)

    # 设置图表大小
    fig, ax = plt.subplots(figsize=(10, 8))

    # 绘制真实基因表达的基因变化比例散点图
    ax.scatter(range(len(real_gene_variation_proportions_sorted)), real_gene_variation_proportions_sorted,
                label='Real Gene Variation Proportion', color='blue', s=0.02)

    # 绘制预测基因表达的基因变化比例散点图
    ax.scatter(range(len(predicted_gene_variation_proportions_sorted)), predicted_gene_variation_proportions_sorted,
                label='Predicted Gene Variation Proportion', color='red', s=0.08)

    # 设置y轴的范围
    ax.set_ylim(-0.0002, 0.0032)
    # 自定义y轴最上面的刻度标签为"≥0.003"
    ax.set_yticks([0, 0.001, 0.002, 0.003])
    ax.set_yticklabels(["0.000", "0.001", "0.002", "≥0.003"])

    # 设置标题和坐标轴标签
    ax.set_xlabel('Ranked Gene')
    ax.set_ylabel('Gene Variation Proportion')

    # 添加图例
    ax.legend(loc='upper left', markerscale=20)  # 设置图例位置和标记大小

    ax_inset = fig.add_axes([0.15, 0.4, 0.6, 0.3])  # [left, bottom, width, height]
    ax_inset.scatter(range(len(real_gene_variation_proportions_sorted)), real_gene_variation_proportions_sorted, color='blue', s=0.02)
    ax_inset.scatter(range(len(predicted_gene_variation_proportions_sorted)), predicted_gene_variation_proportions_sorted, color='red', s=0.08)
    x1, x2 = 1000, 2500
    ax_inset.set_xlim(x1, x2)
    ax_inset.set_ylim(real_gene_variation_proportions_sorted[x1]-0.0001, real_gene_variation_proportions_sorted[x2])
    ax_inset.set_xticks([])  # 移除x轴刻度
    ax_inset.set_yticks([])  # 移除y轴刻度

    for spine in ax_inset.spines.values():
        spine.set_edgecolor('purple')  # 设置边框颜色为紫色
        spine.set_linewidth(2)  # 设置边框线宽
        spine.set_linestyle('--')

    rect = plt.Rectangle((x1, real_gene_variation_proportions_sorted[x1]-0.0001), x2 - x1,
                         real_gene_variation_proportions_sorted[x2] - real_gene_variation_proportions_sorted[x1] + 0.0001,
                         fill=False, edgecolor='purple', lw=2, ls='--')
    ax.add_patch(rect)

    # save
    plt.savefig(save_path, bbox_inches='tight', dpi=500)
    # plt.show()


def build_loaders_inference():
    LOG.logging("Building loaders")
    # (3467, 2378) (3467, 2349) (3467, 2277) (3467, 2265)

    # processing raw data
    dataset1 = CLIPDataset(image_path="./dataset/GSE240429_data/images/GEX_C73_A1_Merged.tiff",
                           spatial_pos_path="./dataset/GSE240429_data/data/tissue_pos_matrices/tissue_positions_list_1.csv",
                           reduced_mtx_path="./dataset/GSE240429_data/data/filtered_expression_matrices/1/harmony_matrix.npy",
                           barcode_path="./dataset/GSE240429_data/data/filtered_expression_matrices/1/barcodes.tsv")
    dataset2 = CLIPDataset(image_path="./dataset/GSE240429_data/images/GEX_C73_B1_Merged.tiff",
                           spatial_pos_path="./dataset/GSE240429_data/data/tissue_pos_matrices/tissue_positions_list_2.csv",
                           reduced_mtx_path="./dataset/GSE240429_data/data/filtered_expression_matrices/2/harmony_matrix.npy",
                           barcode_path="./dataset/GSE240429_data/data/filtered_expression_matrices/2/barcodes.tsv")
    dataset3 = CLIPDataset(image_path="./dataset/GSE240429_data/images/GEX_C73_C1_Merged.tiff",
                           spatial_pos_path="./dataset/GSE240429_data/data/tissue_pos_matrices/tissue_positions_list_3.csv",
                           reduced_mtx_path="./dataset/GSE240429_data/data/filtered_expression_matrices/3/harmony_matrix.npy",
                           barcode_path="./dataset/GSE240429_data/data/filtered_expression_matrices/3/barcodes.tsv")
    dataset4 = CLIPDataset(image_path="./dataset/GSE240429_data/images/GEX_C73_D1_Merged.tiff",
                           spatial_pos_path="./dataset/GSE240429_data/data/tissue_pos_matrices/tissue_positions_list_4.csv",
                           reduced_mtx_path="./dataset/GSE240429_data/data/filtered_expression_matrices/4/harmony_matrix.npy",
                           barcode_path="./dataset/GSE240429_data/data/filtered_expression_matrices/4/barcodes.tsv")
    # use processed data
    # with open('./dataset/GSE240429_data/test/dataset1.pkl', 'rb') as f:
    #     dataset1 = pickle.load(f)
    # LOG.logging("dataset1 load")
    # with open('./dataset/GSE240429_data/test/dataset2.pkl', 'rb') as f:
    #     dataset2 = pickle.load(f)
    # LOG.logging("dataset2 load")
    # with open('./dataset/GSE240429_data/test/dataset3.pkl', 'rb') as f:
    #     dataset3 = pickle.load(f)
    # LOG.logging("dataset3 load")
    # with open('./dataset/GSE240429_data/test/dataset4.pkl', 'rb') as f:
    #     dataset4 = pickle.load(f)
    LOG.logging("dataset4 load")

    dataset = torch.utils.data.ConcatDataset([dataset1, dataset2, dataset3, dataset4])
    test_loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)

    LOG.logging("Finished building loaders")
    return test_loader


def get_image_embeddings(model, test_loader):
    test_image_embeddings = []
    with torch.no_grad():
        if args.dataset == 'her2st':
            tqdm_object = tqdm(test_loader, total=len(test_loader))
            for patch, position, exp, adj, *_, center in tqdm_object:
                batch = {}
                batch["image"] = patch.cuda().squeeze(0).permute(0, 2, 1, 3)
                batch["reduced_expression"] = exp.cuda().squeeze(0)

                image_features = model.image_encoder(batch["image"].permute(0, 2, 1, 3).cuda())
                image_embeddings = model.image_projection(image_features)
                test_image_embeddings.append(image_embeddings)
        else:
            for batch in tqdm(test_loader):
                image_features = model.image_encoder(batch["image"].permute(0, 2, 1, 3).cuda())
                image_embeddings = model.image_projection(image_features)
                test_image_embeddings.append(image_embeddings)
    return torch.cat(test_image_embeddings)


def get_spot_embeddings(model, test_loader):
    spot_embeddings = []
    with torch.no_grad():
        if args.dataset == 'her2st':
            tqdm_object = tqdm(test_loader, total=len(test_loader))
            for patch, position, exp, adj, *_, center in tqdm_object:
                batch = {}
                batch["image"] = patch.cuda().squeeze(0).permute(0, 2, 1, 3)
                batch["reduced_expression"] = exp.cuda().squeeze(0)

                if 'BLEEP' in args.path_dir:  # BLEEP
                    spot_features = batch["reduced_expression"].cuda()
                else:  # ours
                    spot_features = model.spot_encoder(batch["reduced_expression"].cuda())

                her2st.ct = center.squeeze().cpu().numpy()
                her2st.gt = exp.squeeze().cpu().numpy()

                spot_features = model.spot_projection(spot_features)
                spot_embeddings.append(spot_features)
        else:
            for batch in tqdm(test_loader):
                if 'BLEEP' in args.path_dir:  # BLEEP
                    spot_features = batch["reduced_expression"].cuda()
                else:  # ours
                    spot_features = model.spot_encoder(batch["reduced_expression"].cuda())
                spot_features = model.spot_projection(spot_features)
                spot_embeddings.append(spot_features)
    return torch.cat(spot_embeddings)


def save_embeddings(datasize, save_path):
    # model
    # from models import CLIPModel
    CFG.model_name = args.image_encoder
    if 'densenet' in CFG.model_name:
        image_emb = {"densenet121": 1024, "densenet161": 2208, "densenet169": 1664, "densenet201": 1920}
        CFG.image_embedding = image_emb[CFG.model_name]
    # 动态导入模型
    if args.dataset == 'her2st':
        CFG.spot_embedding = 785
        CFG.size = 112
    if 'BLEEP' in args.path_dir:
        model_path = "models_" + 'BLEEP'
    elif 'ablation' in args.path_dir and 'NO' in args.path_dir:
        model_path = "models_" + 'NO'
    elif 'ablation' in args.path_dir and 'Mamba' in args.path_dir:
        model_path = "models_" + 'Mamba'
    elif 'ablation' in args.path_dir and 'RKAN' in args.path_dir:
        model_path = "models_" + 'RKAN'
    else:
        model_path = "models_" + 'ours'
    LOG.logging(f'model_path={model_path}')
    model_type = "CLIPModel"
    model = getattr(importlib.import_module(model_path), model_type)
    model = model().cuda()

    model_path = args.pt_path
    state_dict = torch.load(model_path)
    new_state_dict = {}
    for key in state_dict.keys():
        new_key = key.replace('module.', '')  # remove the prefix 'module.'
        new_key = new_key.replace('well', 'spot')  # for compatibility with prior naming
        new_state_dict[new_key] = state_dict[key]

    model.load_state_dict(new_state_dict)
    model.eval()

    LOG.logging("Finished loading model")

    # dataloader
    if args.dataset == 'her2st':
        dataset_flag = True
        if dataset_flag:
            from utils import pk_load
            trainset = pk_load(args.fold, 'train', False, args.dataset, neighs=args.neighbor, prune=args.prune)
            # pickle.dump(trainset, open('./dataset/her2st_data/her2st_trainset.pkl', 'wb'))
            # LOG.logging(f'save trainset in ./dataset/her2st_data/her2st_trainset.pkl')
            testset = pk_load(args.fold, 'test', False, args.dataset, neighs=args.neighbor, prune=args.prune)
            # pickle.dump(testset, open('./dataset/her2st_data/her2st_testset.pkl', 'wb'))
            # LOG.logging(f'save testset in ./dataset/her2st_data/her2st_testset.pkl')
            if args.fold in [0, 6, 12, 18, 24, 27, 31, 33]:
                her2st.label = testset.label[testset.names[0]]
                # loaded_object = pickle.load(open(f'./dataset/her2st_label_{args.fold}.pkl', 'rb'))
                # pickle.dump(her2st.label, open(f'./dataset/her2st_label_{args.fold}.pkl', 'wb'))
                # sys.exit(0)
        else:
            trainset = pickle.load(open('./dataset/her2st_data/her2st_trainset.pkl', 'rb'))
            testset = pickle.load(open('./dataset/her2st_data/her2st_testset.pkl', 'rb'))
        LOG.logging(f'trainset length: {len(trainset)}')
        train_loader = DataLoader(trainset, batch_size=1, num_workers=0, shuffle=True)
        test_loader = DataLoader(testset, batch_size=1, num_workers=0, shuffle=False)
        her2st.img_embeddings_test = get_image_embeddings(model, test_loader).cpu().numpy()
        her2st.spot_embeddings_test = get_spot_embeddings(model, test_loader).cpu().numpy()

        gt_train = None
        spot_embedding = None
        image_embedding = None
        with torch.no_grad():
            tqdm_object = tqdm(train_loader, total=len(train_loader))
            for patch, position, exp, adj, *_, center in tqdm_object:
                batch = {}
                batch["image"] = patch.cuda().squeeze(0).permute(0, 2, 1, 3)
                batch["reduced_expression"] = exp.cuda().squeeze(0)

                if 'BLEEP' in args.path_dir:  # BLEEP
                    spot_features = batch["reduced_expression"].cuda()
                else:  # ours
                    spot_features = model.spot_encoder(batch["reduced_expression"].cuda())

                if gt_train is None:
                    gt_train = exp.squeeze(0).cpu().numpy()
                else:
                    gt_train = np.concatenate([gt_train, exp.squeeze(0).cpu().numpy()], axis=0)

                spot_features = model.spot_projection(spot_features)
                if spot_embedding is None:
                    spot_embedding = spot_features.cpu().numpy()
                else:
                    spot_embedding = np.concatenate([spot_embedding, spot_features.cpu().numpy()], axis=0)

                image_features = model.image_encoder(batch["image"].permute(0, 2, 1, 3).cuda())
                image_features = model.image_projection(image_features)
                if image_embedding is None:
                    image_embedding = image_features.cpu().numpy()
                else:
                    image_embedding = np.concatenate([image_embedding, image_features.cpu().numpy()], axis=0)

        her2st.img_embeddings_train = image_embedding
        her2st.spot_embeddings_train = spot_embedding
        her2st.gt_train = gt_train
        return
    else:
        test_loader = build_loaders_inference()

    # get image feature
    img_embeddings_all = get_image_embeddings(model, test_loader)
    # get spot feature
    spot_embeddings_all = get_spot_embeddings(model, test_loader)


    img_embeddings_all = img_embeddings_all.cpu().numpy()
    spot_embeddings_all = spot_embeddings_all.cpu().numpy()
    LOG.logging(img_embeddings_all.shape)
    LOG.logging(spot_embeddings_all.shape)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(4):
        index_start = sum(datasize[:i])
        index_end = sum(datasize[:i + 1])
        image_embeddings = img_embeddings_all[index_start:index_end]
        spot_embeddings = spot_embeddings_all[index_start:index_end]
        LOG.logging(image_embeddings.shape)
        LOG.logging(spot_embeddings.shape)
        np.save(save_path + "img_embeddings_" + str(i + 1) + ".npy", image_embeddings.T)
        np.save(save_path + "spot_embeddings_" + str(i + 1) + ".npy", spot_embeddings.T)


# 2265x256, 2277x256
def find_matches(spot_embeddings, query_embeddings, top_k=1):
    # find the closest matches
    spot_embeddings = torch.tensor(spot_embeddings)
    query_embeddings = torch.tensor(query_embeddings)
    query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
    spot_embeddings = F.normalize(spot_embeddings, p=2, dim=-1)
    dot_similarity = query_embeddings @ spot_embeddings.T  # 2277x2265
    LOG.logging(dot_similarity.shape)
    _, indices = torch.topk(dot_similarity.squeeze(0), k=top_k)

    return indices.cpu().numpy()


#plot heatmap of top 50 genes ranked by mean
def plot_heatmap(expression_gt, matched_spot_expression_pred, save_path, top_k=50):
    #take mean of expression
    mean = np.mean(expression_gt, axis=1)
    #take ind of top 100
    ind = np.argpartition(mean, -top_k)[-top_k:]

    # Compute the correlation matrix
    corr_matrix = np.corrcoef(expression_gt[ind,:])
    dendrogram = hierarchy.dendrogram(hierarchy.linkage(corr_matrix, method='ward'), no_plot=True)
    cluster_idx = dendrogram['leaves']

    corr_matrix = np.corrcoef(matched_spot_expression_pred[ind,:])
    corr_matrix = corr_matrix[cluster_idx, :]
    corr_matrix = corr_matrix[:, cluster_idx]

    # Reorder the correlation matrix and plot the heatmap
    plt.figure(dpi=300, figsize=(8, 8))
    sns.heatmap(corr_matrix, cmap='viridis', xticklabels=False, yticklabels=False, cbar= False, vmin=-1, vmax=1)

    # 保存图形到文件
    # plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path, bbox_inches='tight')
    # plt.show()


# dataset3 = CLIPDataset(image_path="./dataset/GSE240429_data/images/GEX_C73_C1_Merged.tiff",
#                        spatial_pos_path="./dataset/GSE240429_data/data/tissue_pos_matrices/tissue_positions_list_3.csv",
#                        reduced_mtx_path="./dataset/GSE240429_data/data/filtered_expression_matrices/3/harmony_matrix.npy",
#                        barcode_path="./dataset/GSE240429_data/data/filtered_expression_matrices/3/barcodes.tsv")
def get_img(exp_matrix, gene='MYEF2'):
    import cv2
    whole_image = cv2.imread("./dataset/GSE240429_data/images/GEX_C73_C1_Merged.tiff")
    spatial_pos_csv = pd.read_csv("./dataset/GSE240429_data/data/tissue_pos_matrices/tissue_positions_list_3.csv", sep=",", header=None)
    # expression_mtx = csr_matrix(sio.mmread(expression_mtx_path)).toarray()
    barcode_tsv = pd.read_csv("./dataset/GSE240429_data/data/filtered_expression_matrices/3/barcodes.tsv", sep="\t", header=None)
    reduced_matrix = np.load("./dataset/GSE240429_data/data/filtered_expression_matrices/3/harmony_matrix.npy").T  # cell x features
    gene_names = pd.read_csv("./dataset/GSE240429_data/data/filtered_expression_matrices/3/features.tsv", header=None, sep="\t").iloc[:, 1].values
    hvg_b = np.load("./dataset/GSE240429_data/data/filtered_expression_matrices/hvg_union.npy")
    gene_names_3467 = gene_names[hvg_b]
    # whole_image_rgb = cv2.cvtColor(whole_image, cv2.COLOR_BGR2RGB)

    indices = np.where(np.array(gene_names_3467) == gene)[0]
    exp_true = reduced_matrix[:, indices].flatten()
    exp_pred = exp_matrix[:, indices].flatten()

    from scipy.stats import pearsonr
    LOG.logging(f'gene: {gene}, Pearson Correlation Coefficient: {pearsonr(exp_true, exp_pred)[0]}')

    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    cmap = plt.get_cmap('hot')
    vmin = np.percentile(exp_true, 10)
    vmax = np.percentile(exp_true, 90)
    # vmin = min(np.min(exp_true), np.min(exp_pred))
    # vmax = max(np.max(exp_true), np.max(exp_pred))
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    colors_true = sm.to_rgba(exp_true, bytes=True)[:, :3]
    colors_pred = sm.to_rgba(exp_pred, bytes=True)[:, :3]
    colors_true = colors_true[:, ::-1]
    colors_pred = colors_pred[:, ::-1]
    whole_image_rgb_true = whole_image.copy()
    whole_image_rgb_pred = whole_image.copy()

    for idx in range(len(barcode_tsv)):
        barcode = barcode_tsv.values[idx, 0]
        v1 = spatial_pos_csv.loc[spatial_pos_csv[0] == barcode, 4].values[0]
        v2 = spatial_pos_csv.loc[spatial_pos_csv[0] == barcode, 5].values[0]
        radius = 100
        cur_color = tuple([int(x) for x in colors_true[idx]])
        whole_image_rgb_true = cv2.circle(whole_image_rgb_true, (v2, v1), radius, cur_color, -1)
        cur_color = tuple([int(x) for x in colors_pred[idx]])
        whole_image_rgb_pred = cv2.circle(whole_image_rgb_pred, (v2, v1), radius, cur_color, -1)

    whole_image_rgb_true = whole_image_rgb_true[2500:17500, 0:22000]
    whole_image_rgb_pred = whole_image_rgb_pred[2500:17500, 0:22000]

    fontsize = 18

    plt.figure(figsize=(10, 10), dpi=300)
    plt.imshow(whole_image_rgb_true)
    plt.title(f'True Gene: {gene}, PCC: {pearsonr(exp_true, exp_true)[0]:.3f}', fontsize=fontsize)
    plt.axis('off')
    plt.savefig(f'{PATH}/{gene}_C1_image_true.png', dpi=300, bbox_inches='tight', pad_inches=0)

    plt.figure(figsize=(10, 10), dpi=300)
    plt.imshow(whole_image_rgb_pred)
    plt.title(f'Pred Gene: {gene}, PCC: {pearsonr(exp_true, exp_pred)[0]:.3f}', fontsize=fontsize)
    plt.axis('off')
    plt.savefig(f'{PATH}/{gene}_C1_image_pred.png', dpi=300, bbox_inches='tight', pad_inches=0)

    # plt.show()


def get_image(exp_matrix, gene='MYEF2', save_path=f'{PATH}/C1_image.png'):
    import cv2
    whole_image = cv2.imread("./dataset/GSE240429_data/images/GEX_C73_C1_Merged.tiff")
    spatial_pos_csv = pd.read_csv("./dataset/GSE240429_data/data/tissue_pos_matrices/tissue_positions_list_3.csv", sep=",", header=None)
    # expression_mtx = csr_matrix(sio.mmread(expression_mtx_path)).toarray()
    barcode_tsv = pd.read_csv("./dataset/GSE240429_data/data/filtered_expression_matrices/3/barcodes.tsv", sep="\t", header=None)
    reduced_matrix = np.load("./dataset/GSE240429_data/data/filtered_expression_matrices/3/harmony_matrix.npy").T  # cell x features
    gene_names = pd.read_csv("./dataset/GSE240429_data/data/filtered_expression_matrices/3/features.tsv", header=None, sep="\t").iloc[:, 1].values
    hvg_b = np.load("./dataset/GSE240429_data/data/filtered_expression_matrices/hvg_union.npy")
    gene_names_3467 = gene_names[hvg_b]
    # whole_image_rgb = cv2.cvtColor(whole_image, cv2.COLOR_BGR2RGB)

    indices = np.where(np.array(gene_names_3467) == gene)[0]
    exp_true = reduced_matrix[:, indices].flatten()
    exp_pred = exp_matrix[:, indices].flatten()

    from scipy.stats import pearsonr
    # LOG.logging(f'gene: {gene}, Pearson Correlation Coefficient: {pearsonr(exp_true, exp_pred)[0]}')

    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    cmap = plt.get_cmap('hot')
    # vmin = np.percentile(exp_pred, 10)
    # vmax = np.percentile(exp_pred, 90)
    vmin = min(np.min(exp_true), np.min(exp_pred))
    vmax = max(np.max(exp_true), np.max(exp_pred))
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    colors_pred = sm.to_rgba(exp_pred, bytes=True)[:, :3]
    colors_pred = colors_pred[:, ::-1]
    whole_image_rgb_pred = whole_image.copy()

    for idx in range(len(barcode_tsv)):
        barcode = barcode_tsv.values[idx, 0]
        v1 = spatial_pos_csv.loc[spatial_pos_csv[0] == barcode, 4].values[0]
        v2 = spatial_pos_csv.loc[spatial_pos_csv[0] == barcode, 5].values[0]
        cur_color = tuple([int(x) for x in colors_pred[idx]])
        whole_image_rgb_pred = cv2.circle(whole_image_rgb_pred, (v2, v1), 100, cur_color, -1)

    whole_image_rgb_pred = whole_image_rgb_pred[2500:17500, 0:22000]

    plt.figure(figsize=(10, 10), dpi=300)
    plt.imshow(whole_image_rgb_pred)
    LOG.logging(f'{args.path_dir} Gene={gene}, PCC={pearsonr(exp_true, exp_pred)[0]:.3f}')
    plt.axis('off')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    LOG.logging(f'saved in {save_path}')
    # plt.show()


def get_cluster(pred, true, save_path=f'{PATH}/cluster_pred.png'):
    from PIL import Image
    import cv2
    import anndata as ad
    adata = ad.AnnData(pred)
    barcode_tsv = pd.read_csv("./dataset/GSE240429_data/data/filtered_expression_matrices/3/barcodes.tsv", sep="\t", header=None)
    spatial_pos_csv = pd.read_csv("./dataset/GSE240429_data/data/tissue_pos_matrices/tissue_positions_list_3.csv", sep=",", header=None)
    ct = []
    for idx in range(len(barcode_tsv)):
        barcode = barcode_tsv.values[idx, 0]
        v1 = spatial_pos_csv.loc[spatial_pos_csv[0] == barcode, 4].values[0]
        v2 = spatial_pos_csv.loc[spatial_pos_csv[0] == barcode, 5].values[0]
        ct.append([v1, v2])
    ct = np.flip(np.array(ct), axis=1)
    adata.obsm['spatial'] = ct

    img = cv2.imread("./dataset/GSE240429_data/images/GEX_C73_C1_Merged.tiff")
    label = visualize_umap_clusters(expr_matrix=true, save_name=f'{PATH}/umap_clusters_true.pdf', save=False)
    label = label.obs['leiden']

    fig_clusters, ax_clusters = plt.subplots()
    clus, ARI = cluster(adata, label)
    LOG.logging(f'GSE240429 result: ARI for pred: {ARI}')
    # adata.obs['kmeans'] = adata.obs['kmeans'].map({'0': '4', '1': '3', '2': '2', '3': '1', '4': '0'})
    sc.pl.spatial(adata, img=img, color='kmeans', spot_size=224, title='', ax=ax_clusters, show=False)
    ax_clusters.get_legend().remove()
    ax_clusters.set_axis_off()
    save_path = f'{PATH}/cluster_result_pred.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    LOG.logging(f'saved in {save_path}')

    adata = ad.AnnData(true)
    adata.obsm['spatial'] = ct
    fig_clusters, ax_clusters = plt.subplots()
    clus, ARI = cluster(adata, label)
    LOG.logging(f'GSE240429 result: ARI for true: {ARI}')
    sc.pl.spatial(adata, img=img, color='kmeans', spot_size=224, title='', ax=ax_clusters, show=False)
    ax_clusters.get_legend().remove()
    ax_clusters.set_axis_off()
    save_path = f'{PATH}/cluster_result_GT.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    LOG.logging(f'saved in {save_path}')


def main():
    #outputs:
    #data sizes: (3467, 2378) (3467, 2349) (3467, 2277) (3467, 2265)

    LOG.logging("****************test starting...****************")

    datasize = [2378, 2349, 2277, 2265]
    save_path = PATH + "/inference/"

    save_embeddings(datasize=datasize, save_path=save_path)

    if args.dataset == 'GSE240429':
        #infer spot embeddings and expression
        spot_expression1 = np.load("./dataset/GSE240429_data/data/filtered_expression_matrices/1/harmony_matrix.npy")
        spot_expression2 = np.load("./dataset/GSE240429_data/data/filtered_expression_matrices/2/harmony_matrix.npy")
        spot_expression3 = np.load("./dataset/GSE240429_data/data/filtered_expression_matrices/3/harmony_matrix.npy")
        spot_expression4 = np.load("./dataset/GSE240429_data/data/filtered_expression_matrices/4/harmony_matrix.npy")

        # save_path = "./inference/embeddings/"
        spot_embeddings1 = np.load(save_path + "spot_embeddings_1.npy")
        spot_embeddings2 = np.load(save_path + "spot_embeddings_2.npy")
        spot_embeddings3 = np.load(save_path + "spot_embeddings_3.npy")
        spot_embeddings4 = np.load(save_path + "spot_embeddings_4.npy")
        image_embeddings3 = np.load(save_path + "img_embeddings_3.npy")

    #query
    if args.dataset == 'her2st':
        # img feature
        image_query = her2st.img_embeddings_test
        # gen expression raw-data
        expression_gt = her2st.gt
        # gen feature
        spot_key = her2st.spot_embeddings_train
        # gen expression raw-data
        expression_key = her2st.gt_train
    else:
        # img feature
        image_query = image_embeddings3
        # gen expression raw-data
        expression_gt = spot_expression3
        # gen feature
        spot_key = np.concatenate([spot_embeddings1, spot_embeddings2, spot_embeddings4], axis=1)
        # gen expression raw-data
        expression_key = np.concatenate([spot_expression1, spot_expression2, spot_expression4], axis=1)

    method = args.aggregation_method
    # save_path = ""
    if image_query.shape[1] != 256:
        image_query = image_query.T
        LOG.logging("image query shape: %s" % str(image_query.shape))
    if expression_gt.shape[0] != image_query.shape[0]:
        expression_gt = expression_gt.T
        LOG.logging("expression_gt shape: %s" % str(expression_gt.shape))
    if spot_key.shape[1] != 256:
        spot_key = spot_key.T
        LOG.logging("spot_key shape: %s" % str(spot_key.shape))
    if expression_key.shape[0] != spot_key.shape[0]:
        expression_key = expression_key.T
        LOG.logging("expression_key shape: %s" % str(expression_key.shape))

    matched_spot_embeddings_pred = None
    matched_spot_expression_pred = None

    if method == "simple":
        indices = find_matches(spot_key, image_query, top_k=args.top_K)
        matched_spot_embeddings_pred = spot_key[indices[:, 0], :]
        LOG.logging("matched spot embeddings pred shape: %s" % str(matched_spot_embeddings_pred.shape))
        matched_spot_expression_pred = expression_key[indices[:, 0], :]
        LOG.logging("matched spot expression pred shape: %s" % str(matched_spot_expression_pred.shape))

    if method == "average":
        LOG.logging("finding matches, using average of top %d expressions" % args.top_K)
        indices = find_matches(spot_key, image_query, top_k=args.top_K)
        matched_spot_embeddings_pred = np.zeros((indices.shape[0], spot_key.shape[1]))
        matched_spot_expression_pred = np.zeros((indices.shape[0], expression_key.shape[1]))
        for i in range(indices.shape[0]):
            matched_spot_embeddings_pred[i, :] = np.average(spot_key[indices[i, :], :], axis=0)
            matched_spot_expression_pred[i, :] = np.average(expression_key[indices[i, :], :], axis=0)

        LOG.logging("matched spot embeddings pred shape: %s" % str(matched_spot_embeddings_pred.shape))
        LOG.logging("matched spot expression pred shape: %s" % str(matched_spot_expression_pred.shape))

    if method == "weighted_average":
        LOG.logging("finding matches, using weighted average of top %d expressions" % args.top_K)
        indices = find_matches(spot_key, image_query, top_k=args.top_K)
        matched_spot_embeddings_pred = np.zeros((indices.shape[0], spot_key.shape[1]))
        matched_spot_expression_pred = np.zeros((indices.shape[0], expression_key.shape[1]))
        for i in range(indices.shape[0]):
            a = np.sum((spot_key[indices[i, 0], :] - image_query[i, :]) ** 2)  # the smallest MSE
            weights = np.exp(-(np.sum((spot_key[indices[i, :], :] - image_query[i, :]) ** 2, axis=1) - a + 1))
            if i == 0:
                LOG.logging("weights: %s" % str(weights))
            matched_spot_embeddings_pred[i, :] = np.average(spot_key[indices[i, :], :], axis=0, weights=weights)
            matched_spot_expression_pred[i, :] = np.average(expression_key[indices[i, :], :], axis=0, weights=weights)

        LOG.logging("matched spot embeddings pred shape: %s" % str(matched_spot_embeddings_pred.shape))
        LOG.logging("matched spot expression pred shape: %s" % str(matched_spot_expression_pred.shape))

    true = expression_gt
    pred = matched_spot_expression_pred

    if args.dataset == 'her2st':

        her2st.pred = pred
        her2st.true = true

        import anndata as ad
        adata = ad.AnnData(her2st.pred)
        adata.obsm['spatial'] = her2st.ct
        adata_gt = ad.AnnData(her2st.gt)
        adata_gt.obsm['spatial'] = her2st.ct
        pred, gt = adata, adata_gt
        R = get_R(pred, gt)[0]

        idx = np.argsort(R[~np.isnan(R)])[-5:][::-1]
        gene_list = list(np.load('./dataset/her2st_data/her_hvg_cut_1000.npy', allow_pickle=True))
        gene_names = np.array(gene_list)[idx]
        gene_PCC = np.array(R)[idx]

        clus, ARI = cluster(pred, her2st.label)
        LOG.logging(f'her2st result: fold={args.fold}  ** {PTID} ** Pearson Correlation: {np.nanmean(R):.3f}, ARI for pred: {ARI}')

        for idx in range(len(gene_names)):
            max_length = max(len(gene_names[idx]), 10)
            LOG.logging(f'her2st result: fold={args.fold}  top-{idx} Gene: {gene_names[idx]:{max_length}}, PCC: {gene_PCC[idx]:.3f}')

        fig, ax = plt.subplots()
        from PIL import Image
        img_id = sorted(os.listdir('./dataset/her2st_data/her2st/data/ST-cnts/'))[1:][args.fold].split('.')[0]
        img_dir = f'./dataset/her2st_data/her2st/data/ST-imgs/{img_id[0:1]}/{img_id}/'
        img = Image.open(img_dir + os.listdir(img_dir)[0])
        sc.pl.spatial(pred, img=img, color='kmeans', spot_size=112, ax=ax, show=False)
        plt.savefig(f'{PATH}/{img_id}_spatial_plot_pred.png', dpi=300)
        plt.close()  # close show fig

        fig, ax = plt.subplots()
        clus, ARI = cluster(gt, her2st.label)
        LOG.logging(f'her2st result: fold={args.fold}  ARI for true: {ARI}')
        sc.pl.spatial(gt, img=img, color='kmeans', spot_size=112, ax=ax, show=False)
        plt.savefig(f'{PATH}/{img_id}_spatial_plot_true.png', dpi=300)
        plt.close()  # close show fig

        LOG.logging(f'---------------------  dataset: {args.dataset}, inference over  ---------------------')
        LOG.logging('\n')
        sys.exit(0)

    # for item in ['SAA2', 'CCL21', 'MT-CO2', 'MGP', 'ORM1', 'GLUL', 'CYP2E1', 'CYP1A2', 'CYP3A4', 'MYEF2']:
    # for item in ["HAL", "CYP3A4", "VWF", "SOX9", "KRT7", "ANXA4", "ACTA2", "DCN"]:
    # for item in ['SAA2', 'CCL21', 'MT-CO2']:
    #     get_image(pred, gene=item, save_path=f'{PATH}/image/image_C1_{item}_{args.path_dir}.png')
    #     get_image(true, gene=item, save_path=f'{PATH}/image/image_C1_{item}_GT.png')

    true_m = np.load("./dataset/GSE240429_data/data/filtered_expression_matrices/3/harmony_matrix.npy").T
    get_cluster(pred, true_m, save_path=f'{PATH}/cluster_pred-harmony.png')

    LOG.logging(pred.shape)
    LOG.logging(true.shape)

    # gene-wise correlation
    corr = np.zeros(pred.shape[0])
    for i in range(pred.shape[0]):
        corr[i] = np.corrcoef(pred[i, :], true[i, :], )[0, 1]
    corr = corr[~np.isnan(corr)]
    LOG.logging("Mean correlation across cells: %s" % str(np.mean(corr)))

    corr = np.zeros(pred.shape[1])
    for i in range(pred.shape[1]):
        corr[i] = np.corrcoef(pred[:, i], true[:, i], )[0, 1]
    corr = corr[~np.isnan(corr)]
    LOG.logging("number of non-zero genes: %s" % str(corr.shape[0]))
    if corr.shape[0] < 50:
        LOG.logging("*************ERROR: current model: Insufficient valid data*************")
        sys.exit()
    LOG.logging("max correlation: %s" % str(np.max(corr)))


    LOG.logging("result...")
    # HEG
    ind = np.argsort(np.sum(true, axis=0))[-50:]
    result_HEG = np.mean(corr[ind])
    result_HEG_SE = np.std(corr[ind], ddof=1) / np.sqrt(50)
    LOG.logging("mean correlation highly expressed genes: %s" % str(np.mean(corr[ind])))
    # HVG
    ind = np.argsort(np.var(true, axis=0))[-50:]
    result_HVG = np.mean(corr[ind])
    result_HVG_SE = np.std(corr[ind], ddof=1) / np.sqrt(50)
    LOG.logging("mean correlation highly variable genes: %s" % str(np.mean(corr[ind])))
    # marker genes
    marker_gene_list = ["HAL", "CYP3A4", "VWF", "SOX9", "KRT7", "ANXA4", "ACTA2", "DCN"]  # markers from macparland paper
    gene_names = pd.read_csv("./dataset/GSE240429_data/data/filtered_expression_matrices/3/features.tsv", header=None,
                             sep="\t").iloc[:, 1].values
    hvg_b = np.load("./dataset/GSE240429_data/data/filtered_expression_matrices/hvg_union.npy")
    marker_gene_ind = np.zeros(len(marker_gene_list))
    for i in range(len(marker_gene_list)):
        marker_gene_ind[i] = np.where(gene_names[hvg_b] == marker_gene_list[i])[0]
    result_MG = np.mean(corr[marker_gene_ind.astype(int)])
    result_MG_SE = np.std(corr[marker_gene_ind.astype(int)], ddof=1) / np.sqrt(len(corr[marker_gene_ind.astype(int)]))
    LOG.logging("mean correlation marker genes: %s" % str(np.mean(corr[marker_gene_ind.astype(int)])))

    corr_temp = corr
    gene_names_temp = gene_names[hvg_b]
    top_5_indices = np.argsort(corr_temp)[-5:]
    # get top 5  for Table 2
    top_5_value = corr_temp[top_5_indices]
    top_5_name = gene_names_temp[top_5_indices]

    # result
    LOG.logging("")
    LOG.logging("tab1--Table 1 or 3 for %s" % args.path_dir)
    LOG.logging("tab1--result: MG=%.3f " % (result_MG))
    LOG.logging("tab1--result: HEG=%.3f" % (result_HEG))
    LOG.logging("tab1--result: HVG=%.3f" % (result_HVG))
    LOG.logging("tab2--Table 2 for %s" % args.path_dir)
    LOG.logging("tab2--result: %s  %.3f" % (top_5_name[4], top_5_value[4]))
    LOG.logging("tab2--result: %s  %.3f" % (top_5_name[3], top_5_value[3]))
    LOG.logging("tab2--result: %s  %.3f" % (top_5_name[2], top_5_value[2]))
    LOG.logging("tab2--result: %s  %.3f" % (top_5_name[1], top_5_value[1]))
    LOG.logging("tab2--result: %s  %.3f" % (top_5_name[0], top_5_value[0]))
    LOG.logging("")

    np.save(save_path + "matched_spot_embeddings_pred.npy", matched_spot_embeddings_pred.T)
    np.save(save_path + "matched_spot_expression_pred.npy", matched_spot_expression_pred.T)

    #construct heatmap of the GGC matrix
    expression_gt = np.load("./dataset/GSE240429_data/data/filtered_expression_matrices/3/harmony_matrix.npy")
    matched_spot_expression_pred_1 = np.load(save_path + "matched_spot_expression_pred.npy")

    # matched_spot_expression_pred_2 = sc.read_h5ad('data/from_collab/harmony_C1_HisToGene_adata_pred.h5ad')
    # matched_spot_expression_pred_2 = matched_spot_expression_pred_2.X.T
    # matched_spot_expression_pred_3 = sc.read_h5ad('data/from_collab/harmony_C1_STNet_adata_pred.h5ad')
    # matched_spot_expression_pred_3 = matched_spot_expression_pred_3.X.T

    LOG.logging(expression_gt.shape)
    LOG.logging(matched_spot_expression_pred_1.shape)

    save_path = save_path.split("infer")[0]
    # plot_heatmap(expression_gt, expression_gt, save_path=save_path + 'heatmap_GT.pdf', top_k=args.top_K)
    # plot_heatmap(expression_gt, matched_spot_expression_pred_1, save_path=save_path + f'heatmap_{args.path_dir}.pdf', top_k=args.top_K)
    # plot_heatmap(expression_gt, matched_spot_expression_pred_2, top_k=args.top_K)
    # plot_heatmap(expression_gt, matched_spot_expression_pred_3, top_k=args.top_K)
    # LOG.logging("result: Figure 4 hotmap saved in %s" % save_path)

    gene_count(true_data=true, pred_data=pred, save_path=save_path + f'gene_count_{args.path_dir}.png')
    gene_variation(true_data=true, pred_data=pred, save_path=save_path + f'gene_variation_{args.path_dir}.png')

    LOG.logging("test finished")


if __name__ == "__main__":
    running_time = time.time()
    main()
    running_time = time.time() - running_time
    hour = running_time // 3600
    minute = (running_time - 3600 * hour) // 60
    second = running_time - 3600 * hour - 60 * minute
    LOG.logging('\n time cost : %d:%d:%d \n' % (hour, minute, second))
