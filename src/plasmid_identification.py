import torch
import argparse
import h5py
import models_metage
from pathlib import Path
import util.misc as misc
import pandas as pd
from Bio import SeqIO
import torch.nn.functional as F

def get_args_parser():
    parser = argparse.ArgumentParser('MetaGE fine-tuning for classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--model', default='MetaGE_classifier', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--resume', default='plasmid_identification.pth', type=str, metavar='RESUME',
                        help='Load model checkpoint')
    parser.add_argument('--result', default='prediction.csv', type=str, metavar='RESULT',
                        help='Result path')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--test_data_path', default='test_data_features.h5', type=str,
                        help='dataset path')
    parser.add_argument('--fasta_file', default='test_dataset.fasta', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=2, type=int,
                        help='number of the classification types')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--embed_dim', type=int, default=256, metavar='PCT',
                        help='embed_dim (default: 120)')
    parser.add_argument('--decoder_embed_dim', type=int, default=64, metavar='PCT',
                        help='decoder_embed_dim (default: 64)')
    return parser

class Dataset(torch.utils.data.Dataset):
    def __init__(self, features, labels, edge_features):
        self.features = features
        self.labels = labels
        self.edge_features = edge_features

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.int64)
        edge_features = torch.tensor(self.edge_features[idx])
        return (feature, label, edge_features)
    
def main(args):
    device = torch.device(args.device)
    fasta_file = args.fasta_file
    contig_names = [record.id for record in SeqIO.parse(fasta_file, "fasta")]

    with h5py.File(args.test_data_path, 'r') as f:
        test_features = f['features'][:]
        test_labels = f['labels'][:]
        test_edge_features = f['edge_features'][:]

    print("Data loaded successfully!")
    dataset_test = Dataset(test_features, test_labels, test_edge_features)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False
    )

    model = models_metage.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        embed_dim=args.embed_dim,
        decoder_embed_dim=args.decoder_embed_dim,
    )

    try:
        checkpoint = torch.load(args.resume, map_location='cpu')
        print("Load model checkpoint from: %s" % args.resume)
        checkpoint_model = checkpoint['model']
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
    except Exception as e:
        print(f"Failed to load model checkpoint: {e}")
        print("Evaluate the model from scratch.")
    model.to(device)
    print(f"Batch size: {args.batch_size}")

    model.eval()
    pred_all = []
    score_all = []

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    for batch in metric_logger.log_every(data_loader_test, 10, header):
        images, target, edge = batch
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        edge = edge.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            output = model(images, edge)
        logits = F.softmax(output, dim=1)
        pred = logits.argmax(dim=1)
        probs = F.softmax(logits, dim=1)
        score = probs[:, 1]
        pred = pred.t()
        score = score.t()
        pred_all.extend(pred.view(-1).tolist())
        score_all.extend(score.view(-1).tolist())

    df = pd.DataFrame({
    "Contig": contig_names,
    "Pred": pred_all,
    "Score": score_all
    })
    df.to_csv(args.result, sep="\t", index=False)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    main(args)

