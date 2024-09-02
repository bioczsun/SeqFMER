## 环境配置

```shell
git clone https://github.com/bioczsun/SeqFMER.git
conda create -n SeqFMER python=3.11
conda activate SeqFMER
conda install mamba -y
mamba install pytorch torchvision torchaudio pytorch-cuda=11.8  -c pytorch -c nvidia
mamba install captum jupyter pandas pysam bedtools bedops pybigwig scikit-learn matplotlib -c conda-forge  -c bioconda
```

# Binary class model

## process genome fasta file
```shell
cd data
#Generate a bed file with 4096bp windows
bedtools makewindows -g ref/hg38.chrom.size -w 4096 > ref/hg38_4096.bed
# bedtools makewindows -g genome.chroms_size -w 4096 > genome.windows.bed

#Sort the bed file
sort -k1,1 -k2,2n ref/hg38_4096.bed > ref/hg38_4096_sorted.bed
# sort -k1,1 -k2,2n genome.windows.bed > genome.windows.sorted.bed
```


**Ideas** 如何排除GC含量对结果的影响？

## 生成负样本
```shell

# 生成负样本
bedops -n 1 ref/hg38_4096_sorted.bed GM12878/ENCFF470YYO.bed > GM12878/nopeaks.bed;

cd ..

mkdir -p train_results/GM12878;


```

## 划分数据集
```shell
python src/split_train_test.py --peaks data/GM12878/ENCFF470YYO.bed --nopeaks data/GM12878/nopeaks.bed --fasta data/ref/hg38.fa --len 600 --outpath data/GM12878
```


## 评估模型在600bp输入序列长度的性能(二分类)
```shell

#DanQ
# --model: ExplaiNN Basset DeepSEA SATORI CNN_Transformer CNN_Attention CNN
for activate in relu exp;do
python src/train_binary.py --data data/GM12878/train_test_allneg_600.npz --model DanQ --linear_units 29440 --activate $activate --device cuda:0 --fasta data/ref/hg38.fa --outpath train_results/GM12878 --seqlen 600 --seed 40 --batch 256 --lr 0.0001;
done

#ExplaiNN
for activate in relu exp;do
python src/train_binary.py --data data/GM12878/train_test_allneg_600.npz --model ExplaiNN --linear_units 600 --activate $activate --device cuda:0 --fasta data/ref/hg38.fa --outpath train_results/GM12878 --seqlen 600 --seed 40 --batch 256 --lr 0.0001;
done
```



### 评估集成模型
```shell
python src/train_binary_grim.py --data data/GM12878/train_test_allneg_600.npz --model DanQ_ExplaiNN --linear_units 14600 --device cuda:0 --fasta data/ref/hg38.fa --outpath train_results/GM12878 --activate exp --seqlen 600 --seed 40 --batch 128 --lr 0.0001;
```


## train repression model
### 生成回归模型数据集


```shell
#new
for i in GM12878 IMR-90  K562  kidney  liver  T-cell;
do python src/split_train_test_coverage.py --peaks data/GM12878/ENCFF470YYO.bed --nopeaks /home/suncz/work/part1/code/NCBenchmark/data/$i/nopeaks.bed --len 400 --fasta /home/suncz/genome_index/hg38/hg38.fa --outpath /home/suncz/work/part1/code/NCBenchmark/data/$i;
done

for i in GM12878 IMR-90  K562  kidney  liver  T-cell;do
bedtools coverage -a /home/suncz/work/part1/code/NCBenchmark/data/$i/train_data_1000_coverage.bed -b /home/suncz/work/embrys01/benchmark/data/$i/${i}.bam > /home/suncz/work/part1/code/NCBenchmark/data/$i/train_data_1000_reads.bed;
done

for i in GM12878 IMR-90  K562  kidney  liver  T-cell;do
bedtools coverage -a /home/suncz/work/part1/code/NCBenchmark/data/$i/test_data_1000_coverage.bed -b /home/suncz/work/embrys01/benchmark/data/$i/${i}.bam > /home/suncz/work/part1/code/NCBenchmark/data/$i/test_data_1000_reads.bed;
done

#<---------------------------------------------------------->
for i in GM12878 IMR-90  K562  kidney  liver  T-cell;do
python /home/suncz/work/part1/code/NCBenchmark/code/repression_data.py --peaks $i/$i.bed --nopeaks $i/nopeaks.bed --len 2048 --fasta /home/suncz/genome_index/hg38/hg38.fa --outpath $i/ --bw /home/suncz/work/part1/code/NCBenchmark/data/$i/$i.bigWig;
done

for i in GM12878 IMR-90  K562  kidney  liver  T-cell;do
python /home/suncz/work/part1/code/NCBenchmark/finalcode/reads_to_npy.py --train /home/suncz/work/part1/code/NCBenchmark/data/$i/train_data_1000_reads.bed --test /home/suncz/work/part1/code/NCBenchmark/data/$i/test_data_1000_reads.bed --outpath /home/suncz/work/part1/code/NCBenchmark/data/$i/train_data_1000_reads.npz;
done


### 训练回归模型
##200
#Basset

for method in DeepSEA Basset ExplaiNN DanQ SATORI CNN_Transformer CNN_Attention CNN;do 
for i in GM12878 IMR-90  K562  kidney  liver  T-cell;do 
python /home/suncz/work/part1/code/NCBenchmark/finalcode/train_repression.py --data /home/suncz/work/part1/code/NCBenchmark/data/$i/train_data_1000_reads.npz --model $method --activate relu --seqlen 1000 --device cuda:3 --fasta /home/suncz/genome_index/hg38/hg38.fa --lr 0.0001 --batch 128 --outpath /home/suncz/work/part1/code/NCBenchmark/train_results/$i; done; done

for i in GM12878 IMR-90  K562  kidney  liver  T-cell;do
python /home/suncz/work/part1/code/NCBenchmark/finalcode/train_repression.py --data /home/suncz/work/part1/code/NCBenchmark/data/$i/train_data_200_reads.npz --model DeepSEA --activate relu --seqlen 200 --device cuda:0 --fasta /home/suncz/genome_index/hg38/hg38.fa --lr 0.0001 --batch 64 --outpath /home/suncz/work/part1/code/NCBenchmark/train_results/$i;
done

  
#DeepSEA
for method in CNN_Transformer CNN_Attention CNN ;do
for i in GM12878 IMR-90  K562  kidney  liver  T-cell;do
python /home/suncz/work/part1/code/NCBenchmark/finalcode/train_repression.py --data /home/suncz/work/part1/code/NCBenchmark/data/$i/train_data_1000_reads.npz --model $method --activate relu --seqlen 1000 --device cuda:1 --fasta /home/suncz/genome_index/hg38/hg38.fa --lr 0.0001 --batch 64 --outpath /home/suncz/work/part1/code/NCBenchmark/train_results/$i;
done
done


#ExplaiNN
for i in GM12878 IMR-90  K562  kidney  liver  T-cell;do
python /home/suncz/work/part1/code/NCBenchmark/finalcode/train_repression.py --data /home/suncz/work/part1/code/NCBenchmark/data/$i/train_data_200_reads.npz --model ExplaiNN --linear_units 200 --activate relu --seqlen 200 --device cuda:0 --fasta /home/suncz/genome_index/hg38/hg38.fa --lr 0.0001 --batch 64 --outpath /home/suncz/work/part1/code/NCBenchmark/train_results/$i;
done

#DanQ
for i in GM12878 IMR-90  K562  kidney  liver  T-cell;do
python /home/suncz/work/part1/code/NCBenchmark/finalcode/train_repression.py --data /home/suncz/work/part1/code/NCBenchmark/data/$i/train_data_200_reads.npz --model DanQ --linear_units 9600 --activate relu --seqlen 200 --device cuda:1 --fasta /home/suncz/genome_index/hg38/hg38.fa --lr 0.0001 --batch 64 --outpath /home/suncz/work/part1/code/NCBenchmark/train_results/$i;
done

#SATORI
for i in GM12878 IMR-90  K562  kidney  liver  T-cell;do
python /home/suncz/work/part1/code/NCBenchmark/finalcode/train_repression.py --data /home/suncz/work/part1/code/NCBenchmark/data/$i/train_data_200_reads.npz --model SATORI --linear_units 10240 --activate relu --seqlen 200 --device cuda:2 --fasta /home/suncz/genome_index/hg38/hg38.fa --lr 0.0001 --batch 64 --outpath /home/suncz/work/part1/code/NCBenchmark/train_results/$i;
done

#CNN_Transformer
for i in GM12878 IMR-90  K562  kidney  liver  T-cell;do
python /home/suncz/work/part1/code/NCBenchmark/finalcode/train_repression.py --data /home/suncz/work/part1/code/NCBenchmark/data/$i/train_data_200_reads.npz --model CNN_Transformer --linear_units 6000 --activate relu --seqlen 200 --device cuda:3 --fasta /home/suncz/genome_index/hg38/hg38.fa --lr 0.0001 --batch 64 --outpath /home/suncz/work/part1/code/NCBenchmark/train_results/$i;
done

#CNN_Attention
for i in GM12878 IMR-90  K562  kidney  liver  T-cell;do
python /home/suncz/work/part1/code/NCBenchmark/finalcode/train_repression.py --data /home/suncz/work/part1/code/NCBenchmark/data/$i/train_data_200_reads.npz --model CNN_Attention --linear_units 6000 --activate relu --seqlen 200 --device cuda:3 --fasta /home/suncz/genome_index/hg38/hg38.fa --lr 0.0001 --batch 64 --outpath /home/suncz/work/part1/code/NCBenchmark/train_results/$i;
done

#CNN
for i in GM12878 IMR-90  K562  kidney  liver  T-cell;do
python /home/suncz/work/part1/code/NCBenchmark/finalcode/train_repression.py --data /home/suncz/work/part1/code/NCBenchmark/data/$i/train_data_200_reads.npz --model CNN --linear_units 9900 --activate relu --seqlen 200 --device cuda:0 --fasta /home/suncz/genome_index/hg38/hg38.fa --lr 0.0001 --batch 64 --outpath /home/suncz/work/part1/code/NCBenchmark/train_results/$i;
done

##400
#Basset
for i in GM12878 IMR-90  K562  kidney  liver  T-cell;do
python /home/suncz/work/part1/code/NCBenchmark/finalcode/train_repression.py --data /home/suncz/work/part1/code/NCBenchmark/data/$i/train_data_400_reads.npz --model Basset --linear_units 4600 --activate relu --seqlen 400 --device cuda:3 --fasta /home/suncz/genome_index/hg38/hg38.fa --lr 0.0001 --batch 64 --outpath /home/suncz/work/part1/code/NCBenchmark/train_results/$i;
done

#DeepSEA
for i in GM12878 IMR-90  K562  kidney  liver  T-cell;do
python /home/suncz/work/part1/code/NCBenchmark/finalcode/train_repression.py --data /home/suncz/work/part1/code/NCBenchmark/data/$i/train_data_200_reads.npz --model DeepSEA --linear_units 35520 --activate relu --seqlen 400 --device cuda:3 --fasta /home/suncz/genome_index/hg38/hg38.fa --lr 0.0001 --batch 64 --outpath /home/suncz/work/part1/code/NCBenchmark/train_results/$i;
done

#ExplaiNN
for i in GM12878 IMR-90  K562  kidney  liver  T-cell;do
python /home/suncz/work/part1/code/NCBenchmark/finalcode/train_repression.py --data /home/suncz/work/part1/code/NCBenchmark/data/$i/train_data_200_reads.npz --model ExplaiNN --linear_units 200 --activate relu --seqlen 400 --device cuda:0 --fasta /home/suncz/genome_index/hg38/hg38.fa --lr 0.0001 --batch 64 --outpath /home/suncz/work/part1/code/NCBenchmark/train_results/$i;
done

#DanQ
for i in GM12878 IMR-90  K562  kidney  liver  T-cell;do
python /home/suncz/work/part1/code/NCBenchmark/finalcode/train_repression.py --data /home/suncz/work/part1/code/NCBenchmark/data/$i/train_data_200_reads.npz --model DanQ --linear_units 9600 --activate relu --seqlen 400 --device cuda:1 --fasta /home/suncz/genome_index/hg38/hg38.fa --lr 0.0001 --batch 64 --outpath /home/suncz/work/part1/code/NCBenchmark/train_results/$i;
done

#SATORI
for i in GM12878 IMR-90  K562  kidney  liver  T-cell;do
python /home/suncz/work/part1/code/NCBenchmark/finalcode/train_repression.py --data /home/suncz/work/part1/code/NCBenchmark/data/$i/train_data_200_reads.npz --model SATORI --linear_units 10240 --activate relu --seqlen 400 --device cuda:2 --fasta /home/suncz/genome_index/hg38/hg38.fa --lr 0.0001 --batch 64 --outpath /home/suncz/work/part1/code/NCBenchmark/train_results/$i;
done

#CNN_Transformer
for i in GM12878 IMR-90  K562  kidney  liver  T-cell;do
python /home/suncz/work/part1/code/NCBenchmark/finalcode/train_repression.py --data /home/suncz/work/part1/code/NCBenchmark/data/$i/train_data_200_reads.npz --model CNN_Transformer --linear_units 6000 --activate relu --seqlen 400 --device cuda:3 --fasta /home/suncz/genome_index/hg38/hg38.fa --lr 0.0001 --batch 64 --outpath /home/suncz/work/part1/code/NCBenchmark/train_results/$i;
done

#CNN_Attention
for i in GM12878 IMR-90  K562  kidney  liver  T-cell;do
python /home/suncz/work/part1/code/NCBenchmark/finalcode/train_repression.py --data /home/suncz/work/part1/code/NCBenchmark/data/$i/train_data_200_reads.npz --model CNN_Attention --linear_units 6000 --activate relu --seqlen 400 --device cuda:3 --fasta /home/suncz/genome_index/hg38/hg38.fa --lr 0.0001 --batch 64 --outpath /home/suncz/work/part1/code/NCBenchmark/train_results/$i;
done

#CNN
for i in GM12878 IMR-90  K562  kidney  liver  T-cell;do
python /home/suncz/work/part1/code/NCBenchmark/finalcode/train_repression.py --data /home/suncz/work/part1/code/NCBenchmark/data/$i/train_data_200_reads.npz --model CNN --linear_units 9900 --activate relu --seqlen 400 --device cuda:0 --fasta /home/suncz/genome_index/hg38/hg38.fa --lr 0.0001 --batch 64 --outpath /home/suncz/work/part1/code/NCBenchmark/train_results/$i;
done

```

### 训练回归模型
```shell
##Basset
for activate in relu exp;do
for i in GM12878 IMR-90  K562  kidney  liver  T-cell;do
python /home/suncz/work/part1/code/NCBenchmark/code/train_repression.py --data /home/suncz/work/part1/code/NCBenchmark/data/$i/train_test_cov_2048.npz --model Basset --linear_units 50800 --activate $activate --device cuda:1 --fasta /home/suncz/genome_index/hg38/hg38.fa --outpath /home/suncz/work/part1/code/NCBenchmark/train_results/$i --seqlen 2048 --seed 40 --batch 128 --lr 0.0001;done
done

##DeepSEA
for activate in relu exp;do
for i in GM12878 IMR-90  K562  kidney  liver  T-cell;do
python /home/suncz/work/part1/code/NCBenchmark/code/train_repression.py --data /home/suncz/work/part1/code/NCBenchmark/data/$i/train_test_cov_2048.npz --model DeepSEA --linear_units 479040 --activate $activate --device cuda:0 --fasta /home/suncz/genome_index/hg38/hg38.fa --outpath /home/suncz/work/part1/code/NCBenchmark/train_results/$i --seqlen 2048 --seed 40 --batch 128 --lr 0.0001;done
done

##DanQ
for activate in relu exp;do
for i in GM12878 IMR-90  K562  kidney  liver  T-cell;do
python /home/suncz/work/part1/code/NCBenchmark/code/train_repression.py --data /home/suncz/work/part1/code/NCBenchmark/data/$i/train_test_cov_2048.npz --model DanQ --linear_units 100480 --activate $activate --device cuda:1 --fasta /home/suncz/genome_index/hg38/hg38.fa --outpath /home/suncz/work/part1/code/NCBenchmark/train_results/$i --seqlen 2048 --seed 40 --batch 128 --lr 0.0001;done
done

##ExplaiNN
for activate in relu exp;do
for i in GM12878 IMR-90  K562  kidney  liver  T-cell;do
python /home/suncz/work/part1/code/NCBenchmark/code/train_repression.py --data /home/suncz/work/part1/code/NCBenchmark/data/$i/train_test_cov_2048.npz --model ExplaiNN --linear_units 2048 --activate $activate --device cuda:1 --fasta /home/suncz/genome_index/hg38/hg38.fa --outpath /home/suncz/work/part1/code/NCBenchmark/train_results/$i --seqlen 2048 --seed 40 --batch 128 --lr 0.0001;done
done

##SATORI
for activate in relu exp;do
for i in GM12878 IMR-90  K562  kidney  liver  T-cell;do
python /home/suncz/work/part1/code/NCBenchmark/code/train_repression.py --data /home/suncz/work/part1/code/NCBenchmark/data/$i/train_test_cov_2048.npz --model SATORI --linear_units 104448 --activate $activate --device cuda:0 --fasta /home/suncz/genome_index/hg38/hg38.fa --outpath /home/suncz/work/part1/code/NCBenchmark/train_results/$i --seqlen 2048 --seed 40 --batch 128 --lr 0.0001;done
done

##CNN_Transformer
for activate in relu exp;do
for i in GM12878 IMR-90  K562  kidney  liver  T-cell;do
python /home/suncz/work/part1/code/NCBenchmark/code/train_repression.py --data /home/suncz/work/part1/code/NCBenchmark/data/$i/train_test_cov_2048.npz --model CNN_Transformer --linear_units 61200 --activate $activate --device cuda:0 --fasta /home/suncz/genome_index/hg38/hg38.fa --outpath /home/suncz/work/part1/code/NCBenchmark/train_results/$i --seqlen 2048 --seed 40 --batch 128 --lr 0.0001;done
done

##CNN_Attention
for activate in relu exp;do
for i in GM12878 IMR-90  K562  kidney  liver  T-cell;do
python /home/suncz/work/part1/code/NCBenchmark/code/train_repression.py --data /home/suncz/work/part1/code/NCBenchmark/data/$i/train_test_cov_2048.npz --model CNN_Attention --linear_units 61200 --activate $activate --device cuda:0 --fasta /home/suncz/genome_index/hg38/hg38.fa --outpath /home/suncz/work/part1/code/NCBenchmark/train_results/$i --seqlen 2048 --seed 40 --batch 128 --lr 0.0001;done
done


##CNN
for activate in relu exp;do
for i in GM12878 IMR-90  K562  kidney  liver  T-cell;do
python /home/suncz/work/part1/code/NCBenchmark/code/train_repression.py --data /home/suncz/work/part1/code/NCBenchmark/data/$i/train_test_cov_2048.npz --model CNN --linear_units 102300 --activate $activate --device cuda:1 --fasta /home/suncz/genome_index/hg38/hg38.fa --outpath /home/suncz/work/part1/code/NCBenchmark/train_results/$i --seqlen 2048 --seed 40 --batch 128 --lr 0.0001;done
done
```


## 评估正负样本选择对结果的影响
### 生成数据集
```shell
for i in GM12878 IMR-90  K562  kidney  liver  T-cell;do
python /home/suncz/work/part1/code/NCBenchmark/code/split_train_test_random.py --peaks $i/${i}.bed --nopeaks $i/nopeaks.bed --fasta /home/suncz/genome_index/hg38/hg38.fa --len 200 --outpath /home/suncz/work/part1/code/NCBenchmark/data/$i/;
done
```

### 生成随机打乱正样本的数据集
```shell
for i in GM12878 IMR-90  K562  kidney  liver  T-cell;do
python /home/suncz/work/part1/code/NCBenchmark/code/split_train_test_shuffle.py --peaks $i/${i}.bed --nopeaks $i/nopeaks.bed --fasta /home/suncz/genome_index/hg38/hg38.fa --len 600 --outpath /home/suncz/work/part1/code/NCBenchmark/data/$i/;
done
```

### 训练模型
```shell
for i in 1k-cell  256-cell  64-cell  dome  oblong;do mkdir /home/suncz/work/part1/code/NCBenchmark/data/embryo/zebrafish/$i; cp $i/${i}_peaks.narrowPeak /home/suncz/work/part1/code/NCBenchmark/data/embryo/zebrafish/$i/;done

##600bp

##Basset
for activate in relu exp;do
for i in GM12878 IMR-90  K562  kidney  liver  T-cell;do
python /home/suncz/work/part1/code/NCBenchmark/code/train_binary_random.py --data /home/suncz/work/part1/code/NCBenchmark/data/$i/train_test_shuffle_600.npz --model Basset --linear_units 14600 --activate $activate --device cuda:0 --fasta /home/suncz/genome_index/hg38/hg38.fa --outpath /home/suncz/work/part1/code/NCBenchmark/train_results/$i --seqlen 600 --seed 40 --batch 256 --lr 0.0001;done
done

for activate in relu;do
for i in GM12878 IMR-90  K562  kidney  liver  T-cell;do
python /home/suncz/work/part1/code/NCBenchmark/code/train_binary_grim.py --data /home/suncz/work/part1/code/NCBenchmark/data/$i/train_test_600.npz --model GRIM --linear_units 14600 --activate $activate --device cuda:0 --fasta /home/suncz/genome_index/hg38/hg38.fa --outpath /home/suncz/work/part1/code/NCBenchmark/train_results/$i --seqlen 600 --seed 40 --batch 128 --lr 0.0001;done
done


## DanQ
for seed in 10 20 30 40 50 60 70 80 90 100;do
for activate in relu exp;do
for i in GM12878 IMR-90  K562  kidney  liver  T-cell;do
python /home/suncz/work/part1/code/NCBenchmark/code/train_binary_random.py --data /home/suncz/work/part1/code/NCBenchmark/data/$i/train_test_random_600.npz --model DanQ --linear_units 29440 --activate $activate --device cuda:1 --fasta /home/suncz/genome_index/hg38/hg38.fa --outpath /home/suncz/work/part1/code/NCBenchmark/train_results/$i --seqlen 200 --seed $seed --batch 256 --lr 0.0001;done
done
done

##ExplaiNN
for i in GM12878 IMR-90  K562  kidney  liver  T-cell;do
python /home/suncz/work/part1/code/NCBenchmark/code/train_binary_random.py --data /home/suncz/work/part1/code/NCBenchmark/data/$i/train_test_random_600.npz --model ExplaiNN --linear_units 600 --activate $activate --device cuda:1 --fasta /home/suncz/genome_index/hg38/hg38.fa --outpath /home/suncz/work/part1/code/NCBenchmark/train_results/$i --seqlen 600 --seed 40 --batch 256 --lr 0.001;done
```

## Basset_ExplaiNN Embryo
```shell
## 生成训练集
python /home/suncz/work/embrys01/benchmark/muticlass/src/MMoE_data.py --fasta /home/suncz/genome_index/bowtie/danRer10/danRer10.fa --target /home/suncz/work/part1/code/NCBenchmark/data/embryo/zebrafish/target.txt --outpath /home/suncz/work/part1/code/NCBenchmark/data/embryo/zebrafish/
```
### 训练模型
```shell
##human
python /home/suncz/work/part1/code/NCBenchmark/code/train_binary_moe.py --data /home/suncz/work/part1/code/NCBenchmark/data/embryo/human/data.npz --model multiclass --linear_units 4 --activate relu --device cuda:0 --fasta /home/suncz/genome_index/hg19/hg19.fa --outpath /home/suncz/work/part1/code/NCBenchmark/train_results/Embryo/human/ --seqlen 600 --seed 40 --batch 32 --lr 0.001

##mouse
python /home/suncz/work/part1/code/NCBenchmark/code/train_binary_moe.py --data /home/suncz/work/part1/code/NCBenchmark/data/embryo/mouse/data.npz --model multiclass --linear_units 5 --activate relu --device cuda:0 --fasta /home/suncz/genome_index/mm10/mm10.fa --outpath /home/suncz/work/part1/code/NCBenchmark/train_results/Embryo/mouse/ --seqlen 600 --seed 40 --batch 32 --lr 0.001

##bostau
python /home/suncz/work/part1/code/NCBenchmark/code/train_binary_moe.py --data /home/suncz/work/part1/code/NCBenchmark/data/embryo/bostau/data.npz --model multiclass --linear_units 6 --activate relu --device cuda:0 --fasta /home/suncz/genome_index/bowtie/bostau/bosTau9.fa --outpath /home/suncz/work/part1/code/NCBenchmark/train_results/Embryo/bostau/ --seqlen 600 --seed 40 --batch 32 --lr 0.001

##chicken
python /home/suncz/work/part1/code/NCBenchmark/code/train_binary_moe.py --data /home/suncz/work/part1/code/NCBenchmark/data/embryo/chicken/data.npz --model multiclass --linear_units 8 --activate relu --device cuda:0 --fasta /home/suncz/genome_index/bowtie/galgal5/galGal5.fa --outpath /home/suncz/work/part1/code/NCBenchmark/train_results/Embryo/chicken/ --seqlen 600 --seed 40 --batch 32 --lr 0.001

##medaka
python /home/suncz/work/part1/code/NCBenchmark/code/train_binary_moe.py --data /home/suncz/work/part1/code/NCBenchmark/data/embryo/medaka/data.npz --model multiclass --linear_units 7 --activate relu --device cuda:0 --fasta /home/suncz/genome_index/bowtie/oryLat2/oryLat2.fa --outpath /home/suncz/work/part1/code/NCBenchmark/train_results/Embryo/medaka/ --seqlen 600 --seed 40 --batch 32 --lr 0.001

##zebrafish
python /home/suncz/work/part1/code/NCBenchmark/code/train_binary_moe.py --data /home/suncz/work/part1/code/NCBenchmark/data/embryo/zebrafish/data.npz --model multiclass --linear_units 5 --activate relu --device cuda:0 --fasta /home/suncz/genome_index/bowtie/danRer10/danRer10.fa --outpath /home/suncz/work/part1/code/NCBenchmark/train_results/Embryo/zebrafish/ --seqlen 600 --seed 40 --batch 32 --lr 0.001
```

### 训练脊椎动物单个阶段模型
```shell
##human
for seed in $(seq 10 10 110);do
for i in 2-cell 8-cell  icm hESC;do 
python /home/suncz/work/part1/code/NCBenchmark/code/train_binary_grim.py --data /home/suncz/work/embrys01/ATAC-seq/peaks/human/$i/${i}_peaks.narrowPeak_train_test_600.npz --model GRIM --linear_units 14600 --device cuda:0 --fasta /home/suncz/genome_index/hg19/hg19.fa --outpath /home/suncz/work/part1/code/NCBenchmark/train_results/Embryo/human/$i --activate exp --seqlen 600 --seed $seed --batch 128 --lr 0.0001;done;done

##mouse
for seed in $(seq 210 10 220);do
for i in 2-cell  4-cell  8-cell  icm  mESC;do 
python /home/suncz/work/part1/code/NCBenchmark/code/train_binary_grim_origin.py --data /home/suncz/work/embrys01/ATAC-seq/peaks/mouse/$i/${i}_peaks.narrowPeak_train_test_600.npz --model GRIM --linear_units 14600 --device cuda:0 --fasta /home/suncz/genome_index/mm10/mm10.fa --outpath /home/suncz/work/part1/code/NCBenchmark/train_results/Embryo/mouse/$i --activate exp --seqlen 600 --seed $seed --batch 128 --lr 0.0001;done;done
--data /home/suncz/work/embrys01/ATAC-seq/peaks/mouse/2-cell/2-cell_peaks.narrowPeak_train_test_600.npz
##bostau
for seed in $(seq 10 10 110);do
for i in 2-cell  4-cell  8-cell  icm morula esc;do 
python /home/suncz/work/part1/code/NCBenchmark/code/train_binary_grim.py --data /home/suncz/work/embrys01/ATAC-seq/peaks/bostau/$i/${i}_peaks.narrowPeak_train_test_600.npz --model GRIM --linear_units 14600  --device cuda:2 --fasta /home/suncz/genome_index/bowtie/bostau/bosTau9.fa --outpath /home/suncz/work/part1/code/NCBenchmark/train_results/Embryo/bostau/$i --activate exp --seqlen 600 --seed $seed --batch 128 --lr 0.0001;done;done

##chicken
for seed in $(seq 10 10 110);do
for i in HH11  HH16  HH19  HH24  HH28  HH32  HH38  HH6;do 
python /home/suncz/work/part1/code/NCBenchmark/code/train_binary_grim.py --data /home/suncz/work/embrys01/ATAC-seq/peaks/chicken/$i/${i}_peaks.narrowPeak_train_test_600.npz --model GRIM --linear_units 14600 --device cuda:3 --fasta /home/suncz/genome_index/bowtie/galgal5/galGal5.fa --outpath /home/suncz/work/part1/code/NCBenchmark/train_results/Embryo/chicken/$i --activate exp --seqlen 600 --seed $seed --batch 128 --lr 0.0001;done;done

##medaka
for seed in $(seq 10 10 110);do
for i in st15 st21  st28  st36 st24  st32  st40;do 
python /home/suncz/work/part1/code/NCBenchmark/code/train_binary_grim.py --data /home/suncz/work/embrys01/ATAC-seq/peaks/medaka/$i/${i}_peaks.narrowPeak_train_test_600.npz --model GRIM --linear_units 14600 --device cuda:0 --fasta /home/suncz/genome_index/bowtie/oryLat2/oryLat2.fa --outpath /home/suncz/work/part1/code/NCBenchmark/train_results/Embryo/medaka/$i --activate exp --seqlen 600 --seed $seed --batch 128 --lr 0.0001;done;done

##zebrafish
for seed in $(seq 10 10 110);do
for i in 1k-cell  256-cell  64-cell  dome  oblong;do
python /home/suncz/work/part1/code/NCBenchmark/code/train_binary_grim.py --data /home/suncz/work/embrys01/ATAC-seq/peaks/zebrafish/$i/${i}_peaks.narrowPeak_train_test_600.npz --model GRIM --linear_units 14600 --device cuda:1 --fasta /home/suncz/genome_index/bowtie/danRer10/danRer10.fa --outpath /home/suncz/work/part1/code/NCBenchmark/train_results/Embryo/zebrafish/$i --activate exp --seqlen 600 --seed $seed --batch 128 --lr 0.0001;done;done

##
```

<!-- for activate in relu;do for i in st32;do python /home/suncz/work/part1/code/NCBenchmark/code/train_binary_grim.py --data /home/suncz/work/embrys01/ATAC-seq/peaks/medaka/$i/${i}_peaks.narrowPeak_train_test_600.npz --model GRIM --linear_units 14600 --activate $activate --device cuda:1 --fasta /home/suncz/genome_index/bowtie/oryLat2/oryLat2.fa --outpath /home/suncz/work/part1/code/NCBenchmark/train_results/Embryo/medaka/$i --seqlen 600 --seed 40 --batch 32 --lr 0.001;done; done -->


<!-- for i in 2-cell;do 
python /home/suncz/work/part1/code/NCBenchmark/code/train_binary_grim.py --data /home/suncz/work/embrys01/ATAC-seq/peaks/human/$i/${i}_peaks.narrowPeak_train_test_600.npz --model GRIM --linear_units 14600 --activate relu --device cuda:0 --fasta /home/suncz/genome_index/hg19/hg19.fa --outpath /home/suncz/work/part1/code/NCBenchmark/train_results/Embryo/human/$i --seqlen 600 --seed 40 --batch 32 --lr 0.001;done -->

### 训练不同脊椎动物不同发育阶段的模型
```shell
#human
for i in 2-cell 8-cell icm hESC;do  
python /home/suncz/work/part1/code/NCBenchmark/code/train_binary_grim.py --data /home/suncz/work/embrys01/ATAC-seq/peaks/human/$i/${i}_peaks.narrowPeak_train_test_600.npz --model GRIM --linear_units 14600 --activate relu --device cuda:0 --fasta /home/suncz/genome_index/hg19/hg19.fa --outpath /home/suncz/work/part1/code/NCBenchmark/train_results/Embryo/human/$i --seqlen 600 --seed 40 --batch 256 --lr 0.0001;done

#mouse
for i in 2-cell  4-cell  8-cell  icm  mESC;do  
python /home/suncz/work/part1/code/NCBenchmark/code/train_binary_grim.py --data /home/suncz/work/embrys01/ATAC-seq/peaks/mouse/$i/${i}_peaks.narrowPeak_train_test_600.npz --model GRIM --linear_units 14600 --activate relu --device cuda:0 --fasta /home/suncz/genome_index/mm10/mm10.fa --outpath /home/suncz/work/part1/code/NCBenchmark/train_results/Embryo/mouse/$i --seqlen 600 --seed 40 --batch 256 --lr 0.0001;done

#bostau
for i in 2-cell  4-cell  8-cell  icm morula esc;do  
python /home/suncz/work/part1/code/NCBenchmark/code/train_binary_grim.py --data /home/suncz/work/embrys01/ATAC-seq/peaks/bostau/$i/${i}_peaks.narrowPeak_train_test_600.npz --model GRIM --linear_units 14600 --activate relu --device cuda:0 --fasta /home/suncz/genome_index/bowtie/bostau/bosTau9.fa --outpath /home/suncz/work/part1/code/NCBenchmark/train_results/Embryo/bostau/$i --seqlen 600 --seed 40 --batch 256 --lr 0.0001;done

#chicken
for i in HH11  HH16  HH19  HH24  HH28  HH32  HH38  HH6;do  
python /home/suncz/work/part1/code/NCBenchmark/code/train_binary_grim.py --data /home/suncz/work/embrys01/ATAC-seq/peaks/chicken/$i/${i}_peaks.narrowPeak_train_test_600.npz --model GRIM --linear_units 14600 --activate relu --device cuda:1 --fasta /home/suncz/genome_index/bowtie/galgal5/galGal5.fa --outpath /home/suncz/work/part1/code/NCBenchmark/train_results/Embryo/chicken/$i --seqlen 600 --seed 40 --batch 256 --lr 0.0001;done

#medaka
for i in st15 st21  st28  st36 st24  st32  st40;do  
python /home/suncz/work/part1/code/NCBenchmark/code/train_binary_grim.py --data /home/suncz/work/embrys01/ATAC-seq/peaks/medaka/$i/${i}_peaks.narrowPeak_train_test_600.npz --model GRIM --linear_units 14600 --activate relu --device cuda:2 --fasta /home/suncz/genome_index/bowtie/oryLat2/oryLat2.fa --outpath /home/suncz/work/part1/code/NCBenchmark/train_results/Embryo/medaka/$i --seqlen 600 --seed 40 --batch 256 --lr 0.0001;done

#zebrafish
for i in 1k-cell  256-cell  64-cell  dome  oblong;do  
python /home/suncz/work/part1/code/NCBenchmark/code/train_binary_grim.py --data /home/suncz/work/embrys01/ATAC-seq/peaks/zebrafish/$i/${i}_peaks.narrowPeak_train_test_600.npz --model GRIM --linear_units 14600 --activate relu --device cuda:3 --fasta /home/suncz/genome_index/bowtie/danRer10/danRer10.fa --outpath /home/suncz/work/part1/code/NCBenchmark/train_results/Embryo/zebrafish/$i --seqlen 600 --seed 40 --batch 256 --lr 0.0001;done




```