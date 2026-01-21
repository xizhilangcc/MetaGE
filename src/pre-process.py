import random
import numpy as np
import h5py
from Bio import SeqIO
from tqdm import tqdm
import itertools
from multiprocessing import Pool
import argparse
from collections import Counter


random.seed(42)


parser = argparse.ArgumentParser(description="Process k-mer parameters for fine-tuning.")
parser.add_argument('--data_path', type=str, default='test_dataset.fasta', help='Path to the input FASTA file.')
parser.add_argument('--k', type=int, default=3, help='Length of k-mer.')
parser.add_argument('--d', type=int,  default=3, help='Distance between k-mers.')
parser.add_argument('--save_data_path', type=str, default='test_data.h5', help='Path to save processed data.')
args = parser.parse_args()


data_path = args.data_path
k = args.k
d = args.d
save_data_path = args.save_data_path


def load_data_with_label(file_path, label):
    records = []
    for record in tqdm(SeqIO.parse(file_path, "fasta")):
        records.append((str(record.seq), label))
    return records

all_samples = load_data_with_label(data_path, label=0)
def build_kmer_index(k = 3):
    bases = ['A', 'T', 'C', 'G']
    kmers = list(itertools.product(bases, repeat=k))
    kmers_new = [''.join(mer) for mer in itertools.product(bases, repeat=k)]
    kmers_dict = {''.join(kmer): idx for idx, kmer in enumerate(kmers)}
    return kmers_new, kmers_dict

def valid_dna(s, alphabet=set("ACGT")):
    return all(ch in alphabet for ch in s)

def symmetric_normalize(A, add_self_loop=True, eps=1e-12):
    degree = A.sum(axis=1)
    inv_sqrt = 1.0 / np.sqrt(np.clip(degree, eps, None))
    A_norm = (A * inv_sqrt[:, None]) * inv_sqrt[None, :]
    return A_norm

def gkm_matrices(seq, kmers, kmer_to_idx, k=3, d=0):
    seq = seq.upper().strip()
    n = len(kmers)
    L = len(seq)
    kmers_in_seq = [seq[i:i+k] for i in range(L - k + 1)]
    counter = Counter(kmers_in_seq)
    all_kmers = [kmer for kmer, idx in sorted(kmer_to_idx.items(), key=lambda x: x[1])]
    counts = {kmer: counter.get(kmer, 0) for kmer in all_kmers}

    matrices_forward  = [np.zeros((n+1, n+1)) for _ in range(d + 1)]
    matrices_backward = [np.zeros((n+1, n+1)) for _ in range(d + 1)]

    for g in range(d + 1):
        max_start_g = L - (2 * k + g) + 1
        if max_start_g <= 0:
            continue

        for i in range(max_start_g):
            k1 = seq[i: i + k]
            k2_start = i + k + g
            k2 = seq[k2_start: k2_start + k]

            if not (valid_dna(k1) and valid_dna(k2)):
                continue
            if (k1 not in kmer_to_idx) or (k2 not in kmer_to_idx):
                continue

            i1 = kmer_to_idx[k1]
            i2 = kmer_to_idx[k2]
            if i1 < i2:
                matrices_forward[g][i1, i2] += 1
            elif i1 > i2:
                matrices_backward[g][i1, i2] += 1
            else:
                matrices_forward[g][i1, i2] += 1
                matrices_backward[g][i1, i2] += 1
    N = n + 1
    for g in range(d + 1):
        matrices_forward[g][N-1, :] = 1
        matrices_forward[g][:, N-1] = 1
        matrices_forward[g] = symmetric_normalize(matrices_forward[g])

        matrices_backward[g][N-1, :] = 1
        matrices_backward[g][:, N-1] = 1
        matrices_backward[g] = symmetric_normalize(matrices_backward[g])

    matrices = matrices_forward + matrices_backward
    return matrices, counts

def process_subsequence(subseq, k=3, d=1):
    kmers, kmers_dict = build_kmer_index(k)
    idx_to_kmers_dict = {v: k for k, v in kmers_dict.items()}
    counts_init = {kmer: 0 for kmer in kmers}
    avg_positions = {}
    for idx, kmer in enumerate(kmers):
        positions = []
        pos = subseq.find(kmer)
        while pos != -1:
            positions.append(pos + 1)
            pos = subseq.find(kmer, pos + 1)
        count = len(positions)
        if count > 0:
            avg_positions[idx] = sum(positions) / count
        else:
            avg_positions[idx] = float("inf")
    sorted_items = sorted(
        [(i, p) for i, p in avg_positions.items()],
        key=lambda item: item[1]
    )
    sorted_items_less = sorted(
        [(i, p) for i, p in avg_positions.items() if p != float("inf")],
        key=lambda item: item[1]
    )
    position_sequence_len = len(sorted_items_less)
    position_sequence = [item[0] for item in sorted_items]
    position_sequence_expand = list(position_sequence)
    n = len(avg_positions)
    position_sequence_final = [n] * n  
    for new_idx, (old_idx, _) in enumerate(sorted_items_less):
        position_sequence_final[new_idx] = old_idx
    kmers_reordered = [kmers[old_idx] for old_idx, _ in sorted_items]
    kmers_dict_reordered = {kmer: i for i, kmer in enumerate(kmers_reordered)}
    matrices, counts = gkm_matrices(subseq, kmers_reordered, kmers_dict_reordered, k=k, d=d)
    values_to_rank = [counts[idx_to_kmers_dict[i]] for i in position_sequence_expand]
    ranks = np.argsort(np.argsort(values_to_rank)[::-1])
    importance_sequence = ranks
    importance_sequence[position_sequence_len:] = n
    feature_vector = np.concatenate([position_sequence, importance_sequence])
    return feature_vector, matrices


def _worker_compute(args):
    fragment, label, k, d = args
    feature, edge_feature = process_subsequence(fragment, k, d)
    has64 = np.any(feature == 64)
    return feature, edge_feature, label, has64

if __name__ == "__main__":
    count_samples_with_64 = 0
    features = []
    labels = []
    edge_features = []
    task_iter = ((fragment, label, k, d) for (fragment, label) in all_samples)

    with Pool(processes=16) as pool:
        for feature, edge_feature, label, has64 in tqdm(
            pool.imap(_worker_compute, task_iter, chunksize=32),
            total=len(all_samples)
        ):
            if has64:
                count_samples_with_64 += 1
            features.append(feature)
            labels.append(label)
            edge_features.append(edge_feature)

    features = np.stack(features, axis=0)
    labels = np.array(labels)
    edge_features = np.stack(edge_features, axis=0)

    with h5py.File(save_data_path, 'w') as f:
        f.create_dataset('features', data=features, compression='lzf')
        f.create_dataset('labels', data=labels, compression='lzf')
        f.create_dataset('edge_features', data=edge_features, compression='lzf')
