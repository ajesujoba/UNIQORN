#!/bin/bash
#SBATCH -p d5gpu
#SBATCH -t 48:00:00
# 300:00
#SBATCH --gres gpu:1
#SBATCH -o exN1.log

eval "$(conda shell.bash hook)"
# conda activate mpi
#gpustat --no-color
#python -m

python scorepath.py \
	--inputfile /GW/qa2/work/uniqorn/Crawl_Google_docs/LCQuAD_2/KGmode/getEntity/getSeedEnt/GitCode/train.json \
	--bertdir /GW/qa2/work/uniqorn/Crawl_Google_docs/LCQuAD_2/KGmode/hdt/BERT/model/bert_base_cased/ \
	--pathfile /GW/qa2/work/uniqorn/Crawl_Google_docs/LCQuAD_2/KGmode/getEntity/getSeedEnt/GitCode/WikiTriplesUniqorn/ \
	--outputfile /GW/qa2/work/uniqorn/Crawl_Google_docs/LCQuAD_2/KGmode/getEntity/getSeedEnt/GitCode/WikiTriplesUniqorn/
