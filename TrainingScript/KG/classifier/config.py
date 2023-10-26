import transformers

DEVICE = "cuda"
MAX_LEN =  512
TRAIN_BATCH_SIZE = 50
VALID_BATCH_SIZE = 50
EPOCHS = 10
ACCUMULATION = 4

BERT_PATH =  "/GW/qa2/work/uniqorn/Crawl_Google_docs/LCQuAD_2/KGmode/hdt/BERT/model/bert_base_cased/"
MODEL_PATH = "model.bin"
TRAINING_FILE = "posneg_train.csv"
#TRAINING_FILE = "/GW/qa2/work/uniqorn/Crawl_Google_docs/LCQuAD_2/KGmode/hdt/BERTdata/traindataReady/posneg_train.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case = False)
