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

from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import util.lr_decay as lrd
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.misc import count_parameters

import models_metage
from contextlib import suppress
from engine import train_one_epoch, evaluate
import h5py
import pickle


def get_args_parser():
    parser = argparse.ArgumentParser('MetaGE fine-tuning for traffic classification', add_help=False)
    # 64
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--save_steps_freq', default=200, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='MetaGE_classifier', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--drop_path', type=float, default=0, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--embed_dim', type=int, default=256, metavar='PCT',
                        help='embed_dim (default: 128)')
    parser.add_argument('--decoder_embed_dim', type=int, default=64, metavar='PCT',
                        help='decoder_embed_dim (default: 64)')
    parser.add_argument('--num_kmers', type=int, default=64, metavar='PCT',
                        help='num_kmers (default: 64)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=2e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.95,
                        help='layer-wise lr decay from ELECTRA/BEiT')
    # default=0.75
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=1, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--train_data_path', default='./fine_tune_train_kmer3.h5', type=str,
                        help='dataset path')
    parser.add_argument('--test_data_path', default='./fine_tune_test_kmer3.h5', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=2, type=int,
                        help='number of the classification types')
    parser.add_argument('--output_dir', default='./output/finetune',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output/finetune',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)


    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # amp about
    parser.add_argument('--if_amp', action='store_true')
    parser.add_argument('--no_amp', action='store_false', dest='if_amp')
    parser.set_defaults(if_amp=True)

    return parser

class Dataset(torch.utils.data.Dataset):
    def __init__(self, features, labels, edge_features, epsilon=1e-12):
        self.features = features
        self.labels = labels
        self.edge_features = edge_features
        self.epsilon = epsilon

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.int64)
        edge_features = torch.tensor(self.edge_features[idx])
        # edge_features = torch.tensor([M.toarray() for M in edge_features])
        return (feature, label, edge_features)

def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    with h5py.File(args.train_data_path, 'r') as f:
        train_features = f['features'][:]
        train_labels = f['labels'][:]
        train_edge_features = f['edge_features'][:]

    with h5py.File(args.test_data_path, 'r') as f:
        test_features = f['features'][:]
        test_labels = f['labels'][:]
        test_edge_features = f['edge_features'][:]

    print("Data loaded successfully!")

    dataset_train = Dataset(train_features, train_labels, train_edge_features)
    dataset_val = Dataset(test_features, test_labels, test_edge_features)
    dataset_test = Dataset(test_features, test_labels, test_edge_features)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False
    )


    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    model = models_metage.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        embed_dim=args.embed_dim,
        decoder_embed_dim=args.decoder_embed_dim,
        num_kmers=args.num_kmers,
    )

    trainable_params, all_param = count_parameters(model)
    print("trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
        trainable_params, all_param, 100 * trainable_params / all_param
    ))
 
    if os.path.exists(args.finetune) and not args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['d2.weight', 'd2.bias', 'd1.weight', 'd1.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
                                        no_weight_decay_list=model_without_ddp.no_weight_decay(),
                                        layer_decay=args.layer_decay
                                        )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    amp_autocast = suppress
    loss_scaler = "none"

    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    max_f1 = 0.0

    best_acc = 0
    best_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler, amp_autocast,
            args.clip_grad, mixup_fn,
            log_writer=log_writer,
            args=args
        )

        test_stats = evaluate(data_loader_val, model, device)
        if test_stats["acc"] > best_acc:
            best_acc = test_stats["acc"]
            best_epoch = epoch
            misc.save_model(args=args, model=model, model_without_ddp=model,
                            optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch,
                            name="best")

        print(f"Accuracy of the network on the {len(dataset_val)} test: {test_stats['acc1']:.4f}")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f"F1 of the network on the {len(dataset_val)} test: {test_stats['weighted_f1']:.4f}")
        max_f1 = max(max_f1, test_stats["weighted_f1"])
        print(f'Max Accuracy: {max_accuracy:.4f}')
        print(f'Max F1: {max_f1:.4f}')

        test_stats.pop("cm") # do not print confusion matrix
        test_stats.pop("pre_per_class")
        test_stats.pop("rec_per_class")
        test_stats.pop("f1_per_class")
        test_stats.pop("support_per_class")

        log_stats = {'epoch': epoch,
                     **{f'train_{k}': v if not isinstance(v, np.ndarray) else v.tolist() for k, v in train_stats.items()},
                     **{f'valid_{k}': v if not isinstance(v, np.ndarray) else v.tolist() for k, v in test_stats.items()},
                     'n_parameters': n_parameters}
        
        if args.output_dir and misc.is_main_process():
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
            "best_valid_acc": best_acc,
            "best_epoch": best_epoch,
        }, f, indent=2)
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    best_checkpoint = torch.load(os.path.join(args.output_dir, "checkpoint-best.pth"), map_location='cpu')
    model.load_state_dict(best_checkpoint['model'])
    test_result = evaluate(data_loader_test, model, device)
    test_result["cm"] = test_result["cm"].tolist()
    with open(os.path.join(args.output_dir, "test_stats.json"), mode="w", encoding="utf-8") as f:
        json.dump({
            "loss": test_result["loss"],
           "acc": test_result["acc"],
            "weighted_pre": test_result["weighted_pre"],
            "weighted_rec": test_result["weighted_rec"],
            "weighted_f1": test_result["weighted_f1"],
            "cm": test_result["cm"],
        } , f, indent=2)

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)