import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import timm.optim.optim_factory as optim_factory

from util.misc import (
    count_parameters, init_distributed_mode, get_rank, get_world_size, 
    is_main_process, load_model
)

import models_metage
from engine import pretrain_one_epoch
from contextlib import suppress
import h5py

def get_args_parser():
    parser = argparse.ArgumentParser('flow mamba pre-training', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--steps', default=150000, type=int)
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--save_steps_freq', default=10000, type=int)
    parser.add_argument('--model', default='MetaGE_pretrain', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--mask_ratio', default=0.15, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--embed_dim', type=int, default=256, metavar='PCT',
                        help='embed_dim (default: 256)')
    parser.add_argument('--decoder_embed_dim', type=int, default=64, metavar='PCT',
                        help='decoder_embed_dim (default: 64)')
    parser.add_argument('--num_kmers', type=int, default=64, metavar='PCT',
                        help='num_kmers (default: 64)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--data_path', default='pretrain.h5', type=str,
                        help='dataset path')
    parser.add_argument('--output_dir', default='./output/pretrain',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output/pretrain',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda:1',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    return parser

class Dataset(torch.utils.data.Dataset):
    def __init__(self, features, edge_features):
        self.features = features
        self.edge_features = edge_features

    def __len__(self):
        return (self.features.shape[0])

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx])
        edge_features = torch.tensor(self.edge_features[idx])
        return (feature, edge_features)

def main(args):

    device = torch.device(args.device)
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    with h5py.File(args.data_path, 'r') as f:
        train_features = f['features'][:]
        train_edge_features = f['edge_features'][:]

    print("Data loaded successfully!")

    dataset_train = Dataset(train_features, train_edge_features)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True
    )

    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # define the model
    model = models_metage.__dict__[args.model](
        embed_dim=args.embed_dim,
        decoder_embed_dim=args.decoder_embed_dim,
        num_kmers=args.num_kmers,
    )

    model.to(device)

    model_without_ddp = model
    # print("Model = %s" % str(model_without_ddp))
    trainable_params, all_param = count_parameters(model)
    print("trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
        trainable_params, all_param, 100 * trainable_params / all_param
    ))

    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)

    amp_autocast = suppress
    loss_scaler = "none"
    load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    epochs = int(args.steps / len(data_loader_train)) + 1
    args.epochs = epochs

    print(f"Start training for {args.steps} steps")
    start_time = time.time()
    for epoch in range(0, epochs):
        train_stats = pretrain_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler, amp_autocast,
            log_writer=log_writer,
            model_without_ddp=model_without_ddp,
            args=args
        )
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch, }
        if args.output_dir and is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    with open(os.path.join(args.output_dir, "train_stats.json"), mode="w", encoding="utf-8") as f:
        json.dump({
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "total_time": total_time,
        }, f, indent=2)

    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
