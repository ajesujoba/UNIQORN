import transformers

DEVICE = "cuda"
MAX_LEN =  512
TRAIN_BATCH_SIZE = 50
VALID_BATCH_SIZE = 50
EPOCHS = 10
ACCUMULATION = 4

BERT_PATH =  "./BERT/"
#MODEL_PATH = "textcased.bin" #"model2021012803e5.bin"
#TRAINING_FILE = "/GW/qa2/work/uniqorn/Crawl_Google_docs/LCQuAD_2/similarity_data3.csv"
#TRAINING_FILE = "/GW/qa2/work/uniqorn/Crawl_Google_docs/LCQuAD_2/KGmode/hdt/BERTdata/traindataReady/posneg_train.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case = False)
