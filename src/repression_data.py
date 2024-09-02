##处理bedgraph
##合并序列到8196 bp
import numpy as np
import pandas as pd
import argparse
import random
import pysam
import pyBigWig
import scipy
import tqdm
from multiprocessing import Manager
mseq_ls = Manager().list()


def set_random_seed(random_seed = 40):
    # set random_seed
    random.seed(random_seed)
    np.random.seed(random_seed)

set_random_seed(40)

params = argparse.ArgumentParser(description='Generate training and testing sets')
params.add_argument("--peaks",help="peaks file",required=True)
params.add_argument("--nopeaks",help="nopeaks file",required=True)
params.add_argument("--bw",help="bigwig file",required=True)
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

# bdg_data = open(args.bdg)


def interp_nan(x, kind='linear'):
    '''Linearly interpolate to fill NaN.'''

    # pad zeroes
    xp = np.zeros(len(x)+2)
    xp[1:-1] = x

    # find NaN
    x_nan = np.isnan(xp)

    if np.sum(x_nan) == 0:
    # unnecessary
        return x

    else:
    # interpolate
        inds = np.arange(len(xp))
        interpolator = scipy.interpolate.interp1d(
        inds[~x_nan],
        xp[~x_nan],
        kind=kind,
        bounds_error=False)

    loc = np.where(x_nan)
    xp[loc] = interpolator(loc)

    # slice off pad
    return xp[1:-1]



def cov_map(mseqs,res=128):
    bw = pyBigWig.open(args.bw)
    qbars = tqdm.tqdm(mseqs,desc="Processing")
    try:
        for mseq in qbars:
            chr_s = mseq[0]
            start_s = int(mseq[1])
            end = int(mseq[2])
            query = bw.values(chr_s,start_s,end)
            cov = np.array(query)
            if np.isnan(cov).sum() / len(cov) < 0.2 :
                cov = np.array(query).reshape(-1,res).mean(axis=-1)
                cov = interp_nan(cov)
                mseq_ls.append(([chr_s,start_s,end],np.round(cov,2)))
                # target_ls.append(cov)
        # outpath_data = "/home/suncz/work/s01/5-31/data/8-cell/data/%s_data.npy"%str(fname)
        # outpath_target = "/home/suncz/work/s01/5-31/data/8-cell/target/%s_target.npy"%str(fname)
    except:
        pass
    # np.save(outpath_data,np.array(mseq_ls))
    # np.save(outpath_target,np.array(target_ls))

def get_train_test_data(peaks_file,nopeaks_file,fasta,seq_len=256,slide=1024):
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



    for line in bed_peaks:
        line = line.split()
        chr = line[0]
        start = int(line[1])
        end = int(line[2])
        # peaks_mid = int(line[9])
        peaks_mid = int((end-start)/2)
        if chr not in ["chr8","chr9"]:
            if is_standard_chrom(chr) and 0 < (start+peaks_mid-int(seq_len/2)) < chromosomes[chr] and 0 < (start+peaks_mid+int(seq_len/2))<chromosomes[chr]:
                train_data_pos.append([chr,start+peaks_mid-int(seq_len/2),start+peaks_mid+int(seq_len/2)])
        else:
            if is_standard_chrom(chr) and 0 < (start+peaks_mid-int(seq_len/2)) < chromosomes[chr] and 0 < (start+peaks_mid+int(seq_len/2))<chromosomes[chr]:
                test_data_pos.append([chr,start+peaks_mid-int(seq_len/2),start+peaks_mid+int(seq_len/2)])
                    

    # peaks_regions = []
    # for line in df[(df["pValue"]>=10)&(df["signalValue"] > df["signalValue"].quantile(0.9))][["chr","start","end","peak"]].iloc:
    #     chr = line.chr
    #     start = int(line.start)
    #     end = int(line.end)
    #     peaks_mid = int(line.peak)
    #     if chr in chroms and (start+peaks_mid-int(seq_len/2))<chromosomes[chr] and (start+peaks_mid+int(seq_len/2))<chromosomes[chr]:
    #         peaks_regions.append([chr,start+peaks_mid-int(seq_len/2),start+peaks_mid+int(seq_len/2)])
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
    train_data = train_data_pos + random.choices(train_data_neg,k=len(train_data_pos))
    test_data = test_data_pos + random.choices(test_data_neg,k=len(test_data_pos))
    print(len(train_data),len(test_data))
    all_data = train_data + test_data
    return all_data







if __name__ == "__main__":
    import multiprocessing
    pool = multiprocessing.Pool(20)
    # target_ls = multiprocessing.Manager().list()
    fname = 0
    lines = []
    n_start = 0
    n_chr = ""
    sequence_data = get_train_test_data(peaks_file,nopeaks_file,fasta,seq_len=2048,slide=4096)

    for i in sequence_data:
        lines.append(i)
        if len(lines) > 50000:
            pool.apply_async(cov_map,args=(lines,))
            fname += 1
            lines = []

    # 处理剩余的lines
    if lines:
        pool.apply_async(cov_map,args=(lines,))
        fname += 1

    pool.close()
    pool.join()

    train_data = []
    train_target = []
    test_data = []
    test_target = []

    pbar_mseqls = tqdm.tqdm(mseq_ls,desc="Processing")
    for data in pbar_mseqls:
        if data[0][0] in ["chr8","chr9"]:
            test_data.append(data[0])
            test_target.append(data[1])
        else:
            train_data.append(data[0])
            train_target.append(data[1])

    np.savez("%s/train_test_cov_%s.npz"%(args.outpath,2048),train_data=train_data,train_label=train_target,test_data=test_data,test_label=test_target)
    





