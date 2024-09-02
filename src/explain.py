import sys
import pandas as pd
import numpy as np
import time
import random
import models
import pysam
import utils
import captum
from captum import *
import torch
from torch.utils.data import Dataset, DataLoader
import os
import argparse

linear_units_dict = {
    "DeepSEA": {
        "200bp": 35520,
        "400bp": 83520,
        "600bp": 131520,
        "800bp": 179520,
        "1000bp": 227520,
    },
    "Basset": {
        "200bp": 4600,
        "400bp": 9600,
        "600bp": 14600,
        "800bp": 19600,
        "1000bp": 24600,
    },
    "DanQ": {
        "200bp": 9600,
        "400bp": 19200,
        "600bp": 29440,
        "800bp": 39040,
        "1000bp": 48640,
    },
    "ExplaiNN": {
        "200bp": 200,
        "400bp": 400,
        "600bp": 600,
        "800bp": 800,
        "1000bp": 1000,
    },
    "SATORI": {
        "200bp": 10240,
        "400bp": 20480,
        "600bp": 30720,
        "800bp": 40960,
        "1000bp": 51200,
    },
    "CNN_Transformer": {
        "200bp": 6000,
        "400bp": 12000,
        "600bp": 18000,
        "800bp": 24000,
        "1000bp": 30000,
    },
    "CNN_Attention": {
        "200bp": 6000,
        "400bp": 12000,
        "600bp": 18000,
        "800bp": 24000,
        "1000bp": 30000,
    },
    "CNN": {
        "200bp": 9900,
        "400bp": 19800,
        "600bp": 30000,
        "800bp": 39900,
        "1000bp": 49800,
    },
}


def set_random_seed(random_seed=40):
    # set random_seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.random.seed = random_seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)


class BinaryDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# Create a custom collate function
def get_custom_collate_fn(fasta):
    def custom_collate_fn(batch):
        # 初始化序列和标签的列表
        sequences = []

        # 遍历batch中的每个(data, label)对
        for item in batch:
            sequence = utils.onehot_seq(
                fasta.fetch(item[0], int(item[1]), int(item[2]))
            )
            sequences.append(sequence)

        # 将序列和标签列表转换为张量
        sequences_tensor = torch.FloatTensor(np.array(sequences))

        return sequences_tensor

    return custom_collate_fn


def calc_motif_IC(motif, background=0.25):
    """IC Bernouli"""
    H = (motif * np.log2(motif / background + 1e-6)).sum()
    return H


class ActivateFeaturesHook:
    def __init__(self, module) -> None:
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output.cpu().data.numpy()

    def get_features(self):
        return self.features

    def close(self):
        self.hook.remove()


def get_fmap(model, hook_module, data_loader, device):
    fmap, X = [], []
    model.eval()
    with torch.no_grad():
        activations = ActivateFeaturesHook(hook_module)
        for x_tensor in data_loader:
            x_tensor = x_tensor.float().to(device).transpose(2, 1)
            _ = model(x_tensor)
            X.append(x_tensor.cpu().numpy())
            fmap.append(activations.get_features())
            # atten_matrix.append(atten.cpu().detach().numpy())
        fmap = np.vstack(fmap)
        X = np.vstack(X)
        # atten_matrix = np.vstack(atten_matrix)
        activations.close()

    return fmap, X


def get_activate_W_from_fmap(fmap, X, padding, pool=1, threshold=0.99, motif_width=10):
    """
    get learned motif pwm based on motif_width
    """
    motif_nb = fmap.shape[1]
    X_dim, seq_len = X.shape[1], X.shape[-1]

    W = []
    seq_ls = []
    for filter_index in range(motif_nb):
        # find regions above threshold
        data_index, pos_index = np.where(
            fmap[:, filter_index, :] > np.max(fmap[:, filter_index, :]) * threshold
        )  # np.where(fmap[:,filter_index,:] > np.max(fmap[:,filter_index,:], axis=1, keepdims=True)*threshold)

        seq_align = []
        count_matrix = []
        for i in range(len(pos_index)):
            # pad 1-nt
            start = pos_index[i] - padding
            end = start + motif_width + 2
            # handle boundary conditions
            if end > seq_len:
                end = seq_len
                start = end - motif_width - 2
            if start < 0:
                start = 0
                end = start + motif_width + 2

            seq = X[data_index[i], :, start * pool : end * pool]
            seq_align.append(seq)
            count_matrix.append(np.sum(seq, axis=0, keepdims=True))

        seq_align = np.array(seq_align)
        seq_ls.append(seq_align)
        count_matrix = np.array(count_matrix)

        # normalize counts
        seq_align = (
            np.sum(seq_align, axis=0) / np.sum(count_matrix, axis=0)
        ) * np.ones((X_dim, (motif_width + 2) * pool))
        seq_align[np.isnan(seq_align)] = 0
        W.append(seq_align)

    W = np.array(W)
    return W, seq_ls


def meme_generate_top(W, tf_list, output_file="meme.txt", prefix="Motif_"):
    # background frequency
    nt_freqs = [1.0 / 4 for i in range(4)]

    # open file for writing
    f = open(output_file, "w")

    # print intro material
    f.write("MEME version 4\n")
    f.write("\n")
    f.write("ALPHABET= ACGT\n")
    f.write("\n")
    f.write("strands: + -\n")
    f.write("\n")
    f.write("Background letter frequencies:\n")
    f.write("A %.4f C %.4f G %.4f T %.4f \n" % tuple(nt_freqs))
    f.write("\n")

    for j in range(len(W)):
        if j in tf_list:
            pwm = W[j]
            if np.sum(pwm) > 0:
                f.write("MOTIF %s%d %d\n" % (prefix, j, j))
                f.write("\n")
                f.write(
                    "letter-probability matrix: alength= 4 w= %d nsites= %d E= 0\n"
                    % (pwm.shape[1], pwm.shape[1])
                )
                for i in range(pwm.shape[1]):
                    f.write("  %.4f\t  %.4f\t  %.4f\t  %.4f\t\n" % tuple(pwm[:, i]))
                f.write("\n")

    f.close()


def meme_generate(W, output_file="meme.txt", prefix="Motif_"):
    # background frequency
    nt_freqs = [1.0 / 4 for i in range(4)]

    # open file for writing
    f = open(output_file, "w")

    # print intro material
    f.write("MEME version 4\n")
    f.write("\n")
    f.write("ALPHABET= ACGT\n")
    f.write("\n")
    f.write("strands: + -\n")
    f.write("\n")
    f.write("Background letter frequencies:\n")
    f.write("A %.4f C %.4f G %.4f T %.4f \n" % tuple(nt_freqs))
    f.write("\n")

    for j in range(len(W)):
        pwm = W[j]
        if np.sum(pwm) > 0:
            f.write("MOTIF %s%d %d\n" % (prefix, j, j))
            f.write("\n")
            f.write(
                "letter-probability matrix: alength= 4 w= %d nsites= %d E= 0\n"
                % (pwm.shape[1], pwm.shape[1])
            )
            for i in range(pwm.shape[1]):
                f.write("  %.4f\t  %.4f\t  %.4f\t  %.4f\t\n" % tuple(pwm[:, i]))
            f.write("\n")

    f.close()


if __name__ == "__main__":
    params = argparse.ArgumentParser(description="Explain model")
    params.add_argument("--data", help="peaks file", required=True)
    params.add_argument("--model", help="model name", default="Basset")
    params.add_argument("--activate", help="activate loss", required=True)
    params.add_argument("--seqlen", help="sequence length", required=True)
    params.add_argument("--seed", help="random seed", default=40, type=int)
    params.add_argument("--device", help="device cuda", default="cuda:0")
    params.add_argument("--fasta", help="fasta seq file", required=True)
    params.add_argument("--batch", help="batch size", default=256)
    params.add_argument("--model_dir", help="Model directory", required=True)
    params.add_argument("--lr", help="learn rata", default=0.00001)
    params.add_argument("--phase", help="phase", required=True)
    params.add_argument("--outpath", help="model name")
    args = params.parse_args()

    set_random_seed(args.seed)

    fasta_file = args.fasta
    fasta = pysam.FastaFile(fasta_file)
    length = args.seqlen
    phase = "GM12878"
    activate = args.activate
    method = args.model
    model_dir = args.model_dir
    peaks_file_path = args.data
    outpath = args.outpath
    phase = args.phase
    # os.chdir("/root/project/SeqFMER")
    fasta_paths = {"human": args.fasta}
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = eval(
        f'models.{method}(1, linear_units_dict["{method}"]["{length}bp"], activate="{activate}")'
    )
    model.load_state_dict(
        torch.load(
            f"{model_dir}/{method}_40_{length}_{activate}_best_network.pth",
            map_location=device,
        )
    )
    model.to(device)
    model.eval()

    data_path = f"{peaks_file_path}"
    data_ = np.load(data_path)
    val_dataset = BinaryDataset(data_["test_data"][0:2000])
    fasta = pysam.FastaFile(fasta_paths["human"])
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(args.batch),
        shuffle=False,
        collate_fn=get_custom_collate_fn(fasta),
        num_workers=10,
    )
    fmap, X = get_fmap(model, next(model.children())[0], val_loader, device=device)
    W, seq_ls = get_activate_W_from_fmap(
        fmap, X, threshold=0.85, motif_width=17, padding=9
    )
    motif_ic = []
    for i in range(len(W)):
        ic = calc_motif_IC(W[i])
        motif_ic.append(ic)
    motif_ic_df = pd.DataFrame(motif_ic)
    if not os.path.exists(f"{outpath}/{length}"):
        os.makedirs(f"{outpath}/{length}")
    motif_ic_df.to_csv(
        f"{outpath}/{length}/motif_ic_{phase}_{method}_{activate}.csv",
        index=False,
        header=False,
    )
    meme_generate(
        W, output_file=f"{outpath}/{length}/motif_{phase}_{method}_{activate}.meme"
    )
