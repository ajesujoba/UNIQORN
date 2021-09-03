# this python file contains the functions needed to compute the question, triple similarity for UNIQORN
import BERT.config as config
import copy
import BERT.dataset as dataset
import torch
from collections import OrderedDict
from BERT.model import BERTBaseUncased
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def getBERTmodel(bertdir):
    MODEL = BERTBaseUncased()
    model_old = torch.load(bertdir, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in model_old.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
        # load params
    MODEL.load_state_dict(new_state_dict)
    MODEL.eval()
    return MODEL

def getsimilarity(q1,q2,model):
    return sentence_pair_prediction(q1,q2,model)

def sentence_pair_prediction(q1,q2,model):
    tokenizer = config.TOKENIZER
    max_len = config.MAX_LEN
    q1 = str(q1)
    q2 = str(q2)
    inputs = tokenizer.encode_plus(q1,q2,add_special_tokens = True, max_length=max_len, padding='longest',truncation=True)
    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]
    padding_length = max_len - len(ids)
    ids = ids + ([0] * padding_length)
    mask = mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([0] * padding_length)
    #unsqueeze to make a batch of 1
    ids = torch.tensor(ids,dtype=torch.long).unsqueeze(0)
    mask = torch.tensor(mask, dtype = torch.long).unsqueeze(0)
    token_type_ids = torch.tensor(token_type_ids, dtype = torch.long).unsqueeze(0)
    #'targets' : torch.tensor(self.target[item], dtype=torch.long)}
    ids = ids.to(device, dtype = torch.long)
    token_type_ids = token_type_ids.to(device, dtype = torch.long)
    mask = mask.to(device, dtype = torch.long)
    #targets = targets.to(device, dtype = torch.float)
    outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
    outputs =  torch.sigmoid(outputs).cpu().detach().numpy()
    return outputs[0][0]

