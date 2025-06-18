import os
import pickle
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from predict import model_predict, sr_predict
from dataset import ViT_HER2ST, ViT_SKIN
import argparse
from argparse import ArgumentParser
import torch.nn.functional as F
import numpy as np
import scanpy as sc, anndata as ad
from sklearn.cluster import KMeans


# modify python path
path = os.path.abspath(__file__)
path = os.path.dirname(path)
path = os.path.dirname(path)
path = os.path.dirname(path)
sys.path.append(path)

parser = argparse.ArgumentParser(description='ST-Net')
parser.add_argument('--ckpt', type=str, default="../../checkpoint/ST-Net/her2st_ST-Net_best.ckpt")  # modify
parser.add_argument('--train', type=str, default='test')  # modify
args = parser.parse_args()


from transformer import ViT
class ST_Net(pl.LightningModule):
    def __init__(self, patch_size=112, n_layers=4, n_genes=1000, dim=1024, learning_rate=1e-4, dropout=0.1, n_pos=64):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        # patch_dim = 3*patch_size*patch_size
        # self.patch_embedding = nn.Linear(patch_dim, dim)
        # self.x_embed = nn.Embedding(n_pos,dim)
        # self.y_embed = nn.Embedding(n_pos,dim)
        # self.vit = ViT(dim=dim, depth=n_layers, heads=16, mlp_dim=2*dim, dropout = dropout, emb_dropout = dropout)

        import timm
        self.densenet = timm.create_model(
            'densenet121', pretrained=False, num_classes=0, global_pool="avg"
        )
        # 加载本地权重
        local_weight_path = '../../info/densenet121-a639ec97.pth'
        local_weight = torch.load(local_weight_path)
        self.densenet.load_state_dict(local_weight, strict=False)
        print(self.densenet.default_cfg['input_size'])
        for p in self.densenet.parameters():
            p.requires_grad = True

        self.gene_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, n_genes)
        )

    def forward(self, patches, centers):
        patches = patches.squeeze(0).view(patches.shape[1], 112, 112, 3).permute(0, 3, 1, 2)

        # patches = self.patch_embedding(patches)
        # centers_x = self.x_embed(centers[:,:,0])
        # centers_y = self.y_embed(centers[:,:,1])
        # x = patches + centers_x + centers_y
        # h = self.vit(x)

        h = self.densenet(patches)
        x = self.gene_head(h)
        return x

    def training_step(self, batch, batch_idx):
        patch, center, exp = batch
        pred = self(patch, center)
        loss = F.mse_loss(pred.view_as(exp), exp)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        patch, center, exp = batch
        pred = self(patch, center)
        loss = F.mse_loss(pred.view_as(exp), exp)
        self.log('valid_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        patch, center, exp = batch
        pred = self(patch, center)
        loss = F.mse_loss(pred.view_as(exp), exp)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser


if args.train == 'train':
    dataset = ViT_HER2ST(train=True, fold=0)
    train_loader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=True)
    model = ST_Net(n_layers=8, n_genes=785, learning_rate=1e-6)
    # trainer = pl.Trainer(gpus=1, max_epochs=100)
    trainer = pl.Trainer(max_epochs=100, accelerator='gpu', devices=[0])
    trainer.fit(model, train_loader)
    trainer.save_checkpoint(args.ckpt)


from sklearn.metrics import adjusted_rand_score as ari_score
def cluster(adata,label):
    idx=label!='undetermined'
    tmp=adata[idx]
    l=label[idx]
    sc.pp.pca(tmp)
    sc.tl.tsne(tmp)
    kmeans = KMeans(n_clusters=len(set(l)), init="k-means++", random_state=0).fit(tmp.obsm['X_pca'])
    p=kmeans.labels_.astype(str)
    lbl=np.full(len(adata),str(len(set(l))))
    lbl[idx]=p
    adata.obs['kmeans']=lbl
    return p,round(ari_score(p,l),3)


from scipy.stats import pearsonr
def get_R(data1,data2,dim=1,func=pearsonr):
    adata1=data1.X
    adata2=data2.X
    r1,p1=[],[]
    for g in range(data1.shape[dim]):
        if dim==1:
            r,pv=func(adata1[:,g],adata2[:,g])
        elif dim==0:
            r,pv=func(adata1[g,:],adata2[g,:])
        r1.append(r)
        p1.append(pv)
    r1=np.array(r1)
    p1=np.array(p1)
    return r1,p1


for fold in [0, 6, 12, 18, 24, 27, 31, 33]:
    model = ST_Net.load_from_checkpoint(args.ckpt, n_layers=8, n_genes=785, learning_rate=1e-5)
    device = torch.device('cpu')
    dataset = ViT_HER2ST(train=False, sr=False, fold=fold)
    # dataset = ViT_HER2ST(train=('test'=='train'),fold=fold,flatten=True,ori=True,neighs=4,adj=True,prune='Grid',r=4)
    test_loader = DataLoader(dataset, batch_size=1, num_workers=4)
    adata_pred, adata_truth = model_predict(model, test_loader, attention=False, device=device)
    label = pickle.load(open(f'../../dataset/her2st_label_{fold}.pkl', 'rb'))
    clus, ARI = cluster(adata_pred, label)

    R = get_R(adata_pred, adata_truth)[0]
    idx = np.argsort(R[~np.isnan(R)])[-5:][::-1]
    gene_list = list(np.load('../../dataset/her2st_data/her_hvg_cut_1000.npy', allow_pickle=True))
    gene_names = np.array(gene_list)[idx]
    gene_PCC = np.array(R)[idx]
    for idx in range(len(gene_names)):
        max_length = max(len(gene_names[idx]), 10)
        print(f'her2st result for ST-Net: fold={fold}  top-{idx} Gene: {gene_names[idx]:{max_length}}, PCC: {gene_PCC[idx]:.3f}')
    print(f'her2st result for ST-Net: fold={fold} Pearson Correlation: {np.nanmean(R):.3f}, ARI for pred: {ARI}')

