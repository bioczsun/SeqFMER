import argparse
import os
import utils
import random
import pysam
import scipy
import numpy as np
import pandas as pd

def set_random_seed(random_seed = 40):
    # set random_seed
    random.seed(random_seed)
    np.random.seed(random_seed)

set_random_seed(40)

params = argparse.ArgumentParser(description='Generate training and testing sets')
params.add_argument("--peaks",help="peaks file",required=True)
params.add_argument("--len",help="sequence length",required=True)
params.add_argument("--fasta",help="fasta seq file",required=True)
params.add_argument("--outpath",help="output path",required=True)
args = params.parse_args()


peaks_file = args.peaks
fasta = args.fasta

def is_standard_chrom(chrom):
    if chrom.startswith("chr") and len(chrom) < 6:
        if chrom[3].isdigit() or chrom[3] == "X":
            return True
        else:
            return False
    else:
        return False


def get_train_test_data(peaks_file,nopeaks_file,fasta,outpath,seq_len=256,slide=1024):
    '''
    peals_file: peaks file
    nopeaks_file: nopeaks file
    fasta: fasta file
    seq_length: sequence length
    '''

    fasta = pysam.FastaFile(fasta)

    chromosomes = {k:v for k,v in zip(fasta.references,fasta.lengths)}#提取染色体长度


    #提取peaks
    df = pd.read_table(peaks_file, header=None)
    df.columns = ['chr', 'start', 'end', 'name', 'score', 'strand', 'signalValue', 'pValue', 'qValue', 'peak']
    # df.sort_values(by="signalValue",ascending=False,inplace=True)
    bed_peaks = open(peaks_file).readlines()

    train_data_pos = []
    test_data_pos = []

    train_label_pos = []
    test_label_pos = []

    for line in bed_peaks:
        line = line.split()
        chr = line[0]
        start = int(line[1])
        end = int(line[2])
        peaks_mid = int(line[9])
        if chr not in ["chr8","chr9"]:
            if is_standard_chrom(chr) and 0 < (start+peaks_mid-int(seq_len/2)) < chromosomes[chr] and 0 < (start+peaks_mid+int(seq_len/2))<chromosomes[chr]:
                train_data_pos.append([chr,start+peaks_mid-int(seq_len/2),start+peaks_mid+int(seq_len/2)])
                train_label_pos.append(float(line[-4]))
        else:
            if is_standard_chrom(chr) and 0 < (start+peaks_mid-int(seq_len/2)) < chromosomes[chr] and 0 < (start+peaks_mid+int(seq_len/2))<chromosomes[chr]:
                test_data_pos.append([chr,start+peaks_mid-int(seq_len/2),start+peaks_mid+int(seq_len/2)])
                test_label_pos.append(float(line[-4]))
                    

    train_data = train_data_pos
    train_label = train_label_pos
    # if len(train_data_neg) > 3*len(train_data_pos):
    #     train_data = train_data_pos + random.choices(train_data_neg,k=len(train_data_pos))
    #     train_label = train_label_pos + random.choices(train_label_neg,k=len(train_data_pos))
    # else:
    #     train_data = train_data_pos + random.choices(train_data_neg,k=len(train_data_pos))
    #     train_label = train_label_pos + random.choices(train_label_neg,k=len(train_data_pos))
    test_data = test_data_pos
    test_label = test_label_pos
    data_outpath = "%s/train_test_foldchange_%s.npz"%(outpath,seq_len)
    print("save data to %s"%data_outpath)

    np.savez(data_outpath,train_data=train_data,train_label=train_label,test_data=test_data,test_label=test_label)#,train_cov = train_cov_ls,test_cov=test_cov_ls)

get_train_test_data(peaks_file,None,fasta,seq_len=int(args.len),slide=4096,outpath=args.outpath)
        





