import argparse
import os

from sympy import sequence
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
params.add_argument("--nopeaks",help="nopeaks file",required=True)
params.add_argument("--len",help="sequence length",required=True)
params.add_argument("--fasta",help="fasta seq file",required=True)
params.add_argument("--outpath",help="output path",required=True)
args = params.parse_args()


peaks_file = args.peaks
nopeaks_file = args.nopeaks
fasta = args.fasta

def is_standard_chrom(chrom):
    if chrom.startswith("chr") and len(chrom) < 6:
        if chrom[3].isdigit() or chrom[3] == "X":
            return True
        else:
            return False
    else:
        return False



def shuffle_string(string):
    # 将字符串转换为列表
    chars = list(string)
    
    # 使用random.shuffle函数随机打乱列表元素顺序
    random.shuffle(chars)
    
    # 将打乱后的列表连接成新的字符串
    shuffled_string = ''.join(chars)
    
    return shuffled_string
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
                train_label_pos.append(1)
        else:
            if is_standard_chrom(chr) and 0 < (start+peaks_mid-int(seq_len/2)) < chromosomes[chr] and 0 < (start+peaks_mid+int(seq_len/2))<chromosomes[chr]:
                test_data_pos.append([chr,start+peaks_mid-int(seq_len/2),start+peaks_mid+int(seq_len/2)])
                test_label_pos.append(1)
                    
    #提取nopeaks

    nopeaks_regions = []
    nopeaks_file = open(nopeaks_file).readlines()
    for i in nopeaks_file:
        i = i.split()
        if int(i[2]) - int(i[1]) ==slide:
            if is_standard_chrom(i[0]) and  0<(int(i[1])+int(slide/2)-int(seq_len/2))<chromosomes[i[0]] and 0<(int(i[2])-int(slide/2)+int(seq_len/2))<chromosomes[i[0]]:
                nopeaks_regions.append([i[0],int(i[1])+int(slide/2)-int(seq_len/2),int(i[2])-int(slide/2)+int(seq_len/2)])  

    #划分数据集


    train_data_neg = []
    test_data_neg = []


            
    for i in nopeaks_regions:
        if i[0] in ["chr8","chr9"]:
            test_data_neg.append(i)
        else:
            train_data_neg.append(i)
            
    #划分数据集
    train_data_sequence_pos = []
    train_data_sequence_neg = []
    train_label_neg = []
    for i in train_data_pos:
        chr = i[0]
        start = i[1]
        end = i[2]
        sequence = fasta.fetch(chr,start,end).upper()
        train_data_sequence_pos.append(sequence)

        sequence_neg = shuffle_string(sequence)
        train_data_sequence_neg.append(sequence_neg)
        train_label_neg.append(0)




    train_sequence_data = train_data_sequence_pos + train_data_sequence_neg
    train_sequence_label = train_label_pos + train_label_neg

    test_data_sequence_pos = []
    test_data_sequence_neg = []
    test_label_neg = []

    for i in test_data_pos:
        chr = i[0]
        start = i[1]
        end = i[2]
        sequence = fasta.fetch(chr,start,end)
        test_data_sequence_pos.append(sequence)

    test_data_neg = random.choices(test_data_neg,k=len(test_data_pos))

    for i in test_data_neg:
        chr = i[0]
        start = i[1]
        end = i[2]
        sequence = fasta.fetch(chr,start,end)
        test_data_sequence_neg.append(sequence)
        test_label_neg.append(0)



    test_sequence_data = test_data_sequence_pos + test_data_sequence_neg
    test_sequence_label = test_label_pos + test_label_neg

    data_outpath = "%s/train_test_shuffle_%s.npz"%(outpath,seq_len)
    print("save data to %s"%data_outpath)

    np.savez(data_outpath,train_data=train_sequence_data,train_label=train_sequence_label,test_data=test_sequence_data,test_label=test_sequence_label)#,train_cov = train_cov_ls,test_cov=test_cov_ls)

get_train_test_data(peaks_file,nopeaks_file,fasta,seq_len=int(args.len),slide=4096,outpath=args.outpath)
        





