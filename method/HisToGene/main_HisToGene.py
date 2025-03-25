import os
import sys

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from vis_model import HisToGene
from utils import *
from predict import model_predict, sr_predict
from dataset import ViT_HER2ST, ViT_SKIN, ViT_HER2ST_HisToGene
import argparse

# modify python path
path = os.path.abspath(__file__)
path = os.path.dirname(path)
path = os.path.dirname(path)
path = os.path.dirname(path)
sys.path.append(path)

parser = argparse.ArgumentParser(description='HisToGene')
parser.add_argument('--ckpt', type=str, default="../../checkpoint/HisToGene/her2st_HisToGene_best.ckpt")  # modify
parser.add_argument('--train', type=str, default='test')  # modify
args = parser.parse_args()

tag = '-htg_her2st_785_32'

if args.train == 'train':
    dataset = ViT_HER2ST_HisToGene(train=True, fold=0)
    # dataset = ViT_HER2ST(train=('train'=='train'),fold=0,flatten=True,ori=True,neighs=4,adj=True,prune='Grid',r=4)
    train_loader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=True)
    model = HisToGene(n_layers=8, n_genes=785, learning_rate=1e-5)
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
    model = HisToGene.load_from_checkpoint(args.ckpt, n_layers=8, n_genes=785,
                                           learning_rate=1e-5)
    device = torch.device('cpu')
    dataset = ViT_HER2ST_HisToGene(train=False, sr=False, fold=fold)
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
        print(f'her2st result for HisToGene: fold={fold}  top-{idx} Gene: {gene_names[idx]:{max_length}}, PCC: {gene_PCC[idx]:.3f}')
    print(f'her2st result for HisToGene: fold={fold} Pearson Correlation: {np.nanmean(R):.3f}, ARI for pred: {ARI}')

