# 0 and 10 represents the range pf questions in train.json from which you want to extract the +/- examples
python extract_kg_training.py /GW/qa2/work/uniqorn/Crawl_Google_docs/LCQuAD_2/LcQUAD2.0_data/train.json /GW/qa2/work/uniqorn/Crawl_Google_docs/LCQuAD_2/KGmode/hdt/WikiTriplesUniqorn2 /GW/qa2/work/uniqorn/Crawl_Google_docs/LCQuAD_2/KGmode/hdt/kgexample.csv 0 10

# negative_ex.csv is the file that has the posiive and negative examples needed for training 
python sample_negative.py kgexample.csv 5 negative_ex.csv

#extract the fields needed for training , this will generate a file posneg_train.csv
python selectcol.py
