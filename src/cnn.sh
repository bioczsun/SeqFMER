##200bp
for activate in relu exp;do
for i in GM12878 IMR-90  K562  kidney  liver  T-cell;do
python /home/suncz/work/part1/code/NCBenchmark/code/train_binary.py --data /home/suncz/work/part1/code/NCBenchmark/data/$i/train_test_200.npz --model CNN --linear_units 9900 --activate $activate --device cuda:0 --fasta /home/suncz/genome_index/hg38/hg38.fa --outpath /home/suncz/work/part1/code/NCBenchmark/train_results/$i --seqlen 200 --seed 40 --batch 256 --lr 0.00001;done
done

##400bp
for activate in relu exp;do
for i in GM12878 IMR-90  K562  kidney  liver  T-cell;do
python /home/suncz/work/part1/code/NCBenchmark/code/train_binary.py --data /home/suncz/work/part1/code/NCBenchmark/data/$i/train_test_400.npz --model CNN --linear_units 19800 --activate $activate --device cuda:0 --fasta /home/suncz/genome_index/hg38/hg38.fa --outpath /home/suncz/work/part1/code/NCBenchmark/train_results/$i --seqlen 400 --seed 40 --batch 256 --lr 0.00001;done
done

##800bp
for activate in relu exp;do
for i in GM12878 IMR-90  K562  kidney  liver  T-cell;do
python /home/suncz/work/part1/code/NCBenchmark/code/train_binary.py --data /home/suncz/work/part1/code/NCBenchmark/data/$i/train_test_800.npz --model CNN --linear_units 39900 --activate $activate --device cuda:0 --fasta /home/suncz/genome_index/hg38/hg38.fa --outpath /home/suncz/work/part1/code/NCBenchmark/train_results/$i --seqlen 800 --seed 40 --batch 256 --lr 0.00001;done
done

##1000bp
for activate in relu exp;do
for i in GM12878 IMR-90  K562  kidney  liver  T-cell;do
python /home/suncz/work/part1/code/NCBenchmark/code/train_binary.py --data /home/suncz/work/part1/code/NCBenchmark/data/$i/train_test_1000.npz --model CNN --linear_units 49800 --activate $activate --device cuda:0 --fasta /home/suncz/genome_index/hg38/hg38.fa --outpath /home/suncz/work/part1/code/NCBenchmark/train_results/$i --seqlen 1000 --seed 40 --batch 256 --lr 0.00001;done
done