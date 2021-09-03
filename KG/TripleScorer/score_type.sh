#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 7-00:00:00
# 300:00
#SBATCH --gres gpu:1
#SBATCH -o scoreType.log

eval "$(conda shell.bash hook)"
# conda activate mpi
#gpustat --no-color
#python -m

python scoreTypes.py \
	--inputfile /GW/qa2/work/uniqorn/Crawl_Google_docs/LCQuAD_2/KGmode/getEntity/getSeedEnt/GitCode/train.json \
	--triplefile /GW/qa2/work/uniqorn/Crawl_Google_docs/LCQuAD_2/KGmode/getEntity/getSeedEnt/GitCode/WikiTriplesUniqorn/ \
	--outputfile /GW/qa2/work/uniqorn/Crawl_Google_docs/LCQuAD_2/KGmode/getEntity/getSeedEnt/GitCode/ScoreTriples/ \
	--modeldir /GW/qa2/work/uniqorn/Crawl_Google_docs/LCQuAD_2/KGmode/getEntity/getSeedEnt/GitCode/BERT/model2021012803e5.bin
