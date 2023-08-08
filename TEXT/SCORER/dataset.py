import config
import torch

class BERTDataset:
    def __init__(self, q1, q2, target):
        self.q1 = q1
        self.q2 = q2
        self.target = target
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN
    
    def __len__(self):
        return len(self.q1)
    
    def __getitem__(self, item):
        q1 = str(self.q1[item])
        q2 = str(self.q2[item])
        
        q1 =  " ".join(q1.split())
        q2 =  " ".join(q2.split())
        
        inputs = self.tokenizer.encode_plus(q1,q2,add_special_tokens = True, max_length=self.max_len, padding='longest',truncation=True)
        
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]
        
        padding_length = self.max_len - len(ids)
        ids = ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        
        return {'ids':torch.tensor(ids,dtype=torch.long), 'mask' : torch.tensor(mask, dtype = torch.long), 'token_type_ids' : torch.tensor(token_type_ids, dtype = torch.long), 'targets' : torch.tensor(self.target[item], dtype=torch.long)}
        
        
