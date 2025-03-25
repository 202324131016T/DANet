import importlib
import os
import pickle
import sys
import time

from tqdm import tqdm
import torch
from torch import nn
import torch.distributed as dist
import torch.utils.data.distributed

import config as CFG
from dataset import CLIPDataset
from utils import AvgMeter, get_lr, LogPrint
from torch.utils.data import DataLoader

# her2st
from utils import pk_load_main

import argparse

parser = argparse.ArgumentParser(description='main for CLIP')
parser.add_argument('--batch_size', type=int, default=512, help='')  # modify
parser.add_argument('--lr', type=float, default=0.001, help='')  # modify
parser.add_argument('--max_epochs', type=int, default=150, help='')  # modify
parser.add_argument('--num_workers', type=int, default=0, help='')
parser.add_argument('--init_method', default='tcp://127.0.0.1:3456', type=str, help='')
# parser.add_argument('--dist-backend', default='gloo', type=str, help='')
parser.add_argument('--dist-backend', default='nccl', type=str, help='')
parser.add_argument('--world_size', default=1, type=int, help='')
parser.add_argument('--distributed', action='store_true', help='')
parser.add_argument('--path_dir', type=str, default='debug', help='')  # modify
parser.add_argument('--image_encoder', type=str, default='densenet121', help='resnet50 densenet121')  # modify

# dataset her2st
parser.add_argument('--dataset', type=str, default='her2st', help='dataset name:{"GSE240429", "her2st"}.')  # modify
parser.add_argument('--fold', type=int, default=11, help='0 6 12 18 24 27 31 33')  # modify
parser.add_argument('--neighbor', type=int, default=4, help='the number of neighbors in the GNN.')
parser.add_argument('--prune', type=str, default='Grid', help='how to prune the edge:{"Grid","NA"}')

args = parser.parse_args()

from datetime import datetime
datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

PATH = './result/' + args.path_dir + '/'
LOG = LogPrint(PATH)
LOG.logging(f'\n\n------------------------  datetime: {datetime}  ------------------------\n')
LOG.logging("current model: %s" % PATH)
LOG.logging("current epoch: %s" % str(args.max_epochs))
LOG.logging("current batch_size: %s" % str(args.batch_size))
LOG.logging("current lr: %s" % str(args.lr))
LOG.logging("current dataset: %s" % str(args.dataset))


def build_loaders(args):
    # slice 3 randomly chosen to be test and will be left out during training
    LOG.logging("Building loaders")

    # processing raw data
    dataset1 = CLIPDataset(image_path = "./dataset/GSE240429_data/images/GEX_C73_A1_Merged.tiff",
               spatial_pos_path = "./dataset/GSE240429_data/data/tissue_pos_matrices/tissue_positions_list_1.csv",
               reduced_mtx_path = "./dataset/GSE240429_data/data/filtered_expression_matrices/1/harmony_matrix.npy",
               barcode_path = "./dataset/GSE240429_data/data/filtered_expression_matrices/1/barcodes.tsv")
    dataset2 = CLIPDataset(image_path = "./dataset/GSE240429_data/images/GEX_C73_B1_Merged.tiff",
                spatial_pos_path = "./dataset/GSE240429_data/data/tissue_pos_matrices/tissue_positions_list_2.csv",
                reduced_mtx_path = "./dataset/GSE240429_data/data/filtered_expression_matrices/2/harmony_matrix.npy",
                barcode_path = "./dataset/GSE240429_data/data/filtered_expression_matrices/2/barcodes.tsv")
    dataset4 = CLIPDataset(image_path = "./dataset/GSE240429_data/images/GEX_C73_D1_Merged.tiff",
                spatial_pos_path = "./dataset/GSE240429_data/data/tissue_pos_matrices/tissue_positions_list_4.csv",
                reduced_mtx_path = "./dataset/GSE240429_data/data/filtered_expression_matrices/4/harmony_matrix.npy",
                barcode_path = "./dataset/GSE240429_data/data/filtered_expression_matrices/4/barcodes.tsv")
    # use processed data
    # with open('./dataset/GSE240429_data/train/dataset1.pkl', 'rb') as f:
    #     dataset1 = pickle.load(f)
    # LOG.logging("dataset1 load")
    # with open('./dataset/GSE240429_data/train/dataset2.pkl', 'rb') as f:
    #     dataset2 = pickle.load(f)
    # LOG.logging("dataset2 load")
    # with open('./dataset/GSE240429_data/train/dataset4.pkl', 'rb') as f:
    #     dataset4 = pickle.load(f)
    LOG.logging("dataset4 load")

    dataset = torch.utils.data.ConcatDataset([dataset1, dataset2, dataset4])

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    LOG.logging('train_data length %d' % len(train_dataset))
    LOG.logging('test_data length %d' % len(test_dataset))
    LOG.logging("train/test split completed")

    # Set up distributed sampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) #by default, rank and world sizes are retrieved from env variables
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.num_workers, sampler=train_sampler, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    LOG.logging("Finished building loaders")
    return train_loader, test_loader


def cleanup():
    dist.destroy_process_group()


def train_epoch(model, train_loader, optimizer, args, lr_scheduler=None):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        batch = {k: v.cuda() for k, v in batch.items() if k == "image" or k == "reduced_expression"}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()

        # for param in model.parameters():
        #     torch.distributed.all_reduce(param.grad.data, op=torch.distributed.ReduceOp.SUM)
        #     param.grad.data /= args.world_size

        optimizer.step()
        # if step == "batch":
        #   lr_scheduler.step()

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))

    return loss_meter


def train_epoch_her2st(model, train_loader, optimizer, args, lr_scheduler=None):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for patch, position, exp, adj, *_, center in tqdm_object:
        batch = {}
        batch["image"] = patch.cuda().squeeze(0).permute(0, 2, 1, 3)
        batch["reduced_expression"] = exp.cuda().squeeze(0)

        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)
        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))

    return loss_meter


def test_epoch_her2st(model, test_loader):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(test_loader, total=len(test_loader))
    for patch, position, exp, adj, *_, center in tqdm_object:
        batch = {}
        batch["image"] = patch.cuda().squeeze(0).permute(0, 2, 1, 3)
        batch["reduced_expression"] = exp.cuda().squeeze(0)
        loss = model(batch)

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter


def test_epoch(model, test_loader):
    loss_meter = AvgMeter()

    tqdm_object = tqdm(test_loader, total=len(test_loader))
    for batch in tqdm_object:
        batch = {k: v.cuda() for k, v in batch.items() if k == "image" or k == "reduced_expression"}
        loss = model(batch)

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter


def main():
    LOG.logging("****************train starting...****************")

    ngpus_per_node = torch.cuda.device_count()
    LOG.logging(" gpu num for per node (total gpu nmu): %d" % ngpus_per_node)

    local_rank = int(os.environ.get("SLURM_LOCALID"))
    rank = int(os.environ.get("SLURM_NODEID"))*ngpus_per_node + local_rank

    current_device = local_rank
    LOG.logging("current device: %d" % local_rank)
    torch.cuda.set_device(current_device)

    """ 
    this block initializes a process group and initiate communications
		between all processes running on all nodes 
	"""

    LOG.logging('From Rank: {}, ==> Initializing Process Group...'.format(rank))
    #init the process group
    dist.init_process_group(backend=args.dist_backend, init_method=args.init_method, world_size=args.world_size, rank=rank)
    LOG.logging("process group ready!")

    CFG.model_name = args.image_encoder
    if 'densenet' in CFG.model_name:
        image_emb = {"densenet121": 1024, "densenet161": 2208, "densenet169": 1664, "densenet201": 1920}
        CFG.image_embedding = image_emb[CFG.model_name]
        LOG.logging(f'CFG.image_embedding = {CFG.image_embedding}')

    #make the model
    LOG.logging('From Rank: {}, ==> Making model..'.format(rank))
    # from models import CLIPModel
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

    model = nn.parallel.DistributedDataParallel(model, device_ids=[current_device])

    #load the data
    LOG.logging('From Rank: {}, ==> Preparing data..'.format(rank))
    if args.dataset == 'her2st':
        dataset_flag = True
        if dataset_flag:
            trainset = pk_load_main(args.fold, 'train', False, args.dataset, neighs=args.neighbor, prune=args.prune)
            # pickle.dump(trainset, open('./dataset/her2st_data/her2st_trainset.pkl', 'wb'))
            # LOG.logging(f'save trainset in ./dataset/her2st_data/her2st_trainset.pkl')
            testset = pk_load_main(args.fold, 'test', False, args.dataset, neighs=args.neighbor, prune=args.prune)
            # pickle.dump(testset, open('./dataset/her2st_data/her2st_testset.pkl', 'wb'))
            # LOG.logging(f'save testset in ./dataset/her2st_data/her2st_testset.pkl')
        else:
            trainset = pickle.load(open('./dataset/her2st_data/her2st_trainset.pkl', 'rb'))
            testset = pickle.load(open('./dataset/her2st_data/her2st_testset.pkl', 'rb'))
        LOG.logging(f'trainset length: {len(trainset)}')
        train_loader = DataLoader(trainset, batch_size=1, num_workers=0, shuffle=True)
        test_loader = DataLoader(testset, batch_size=1, num_workers=0, shuffle=False)
    else:
        train_loader, test_loader = build_loaders(args)

    # Initialize optimizer and learning rate scheduler
    if args.dataset == 'her2st':
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=CFG.weight_decay
            # model.parameters(), lr=args.lr
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=CFG.weight_decay
        )
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
    # )

    # Train the model for a fixed number of epochs
    best_loss = float('inf')
    best_epoch = 0
    for epoch in range(args.max_epochs):
        LOG.logging(f"Epoch: {epoch}")
        # step = "epoch"

        # Train the model
        model.train()
        if args.dataset == 'her2st':
            train_loss = train_epoch_her2st(model, train_loader, optimizer, args)
        else:
            train_loss = train_epoch(model, train_loader, optimizer, args)

        # Evaluate the model
        model.eval()
        with torch.no_grad():
            if args.dataset == 'her2st':
                test_loss = test_epoch_her2st(model, test_loader)
            else:
                test_loss = test_epoch(model, test_loader)

        if test_loss.avg < best_loss and rank == 0:
            best_loss = test_loss.avg
            best_epoch = epoch
            torch.save(model.state_dict(), PATH + "best.pt")
            torch.save(model.state_dict(), PATH + "best_" + str(epoch) + ".pt")
            LOG.logging("Saved Best Model! Loss: {}".format(best_loss))

    LOG.logging("Done!, final loss: {}".format(best_loss))
    LOG.logging("Best epoch: {}".format(best_epoch))
    LOG.logging("train finished")
    cleanup()


if __name__ == "__main__":
    begin_time = time.time()
    main()
    running_time = time.time() - begin_time
    hour = running_time // 3600
    minute = (running_time - 3600 * hour) // 60
    second = running_time - 3600 * hour - 60 * minute
    LOG.logging('\n time cost : %d:%d:%d \n' % (hour, minute, second))

