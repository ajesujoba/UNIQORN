import config 
import dataset
import torch
import pandas as pd
from sklearn import model_selection
import engine
import numpy as np
from model import BERTBaseUncased
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn import metrics
import torch.nn as nn
import sys

def run():
    dfx = pd.read_csv(config.TRAINING_FILE).fillna("none")
    
    df_train, df_valid = model_selection.train_test_split(dfx, test_size=0.2, random_state=42, stratify=dfx.label.values)
    
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)
    
    #pass the q1,q2,target here . 
    train_dataset = dataset.BERTDataset(df_train.question.values,df_train.context.values,target = df_train.label.values)
    
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size = config.TRAIN_BATCH_SIZE, num_workers = 4)
    
    valid_dataset = dataset.BERTDataset(df_valid.question.values, df_valid.context.values,df_valid.label.values)
    
    valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size = config.VALID_BATCH_SIZE, num_workers = 1)
    
    device = torch.device("cuda")
    model = BERTBasecased()
    model.to(device)
    
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias','LayerNorm.bias','LayetNorm.weight']
    optimizer_parameters = [{'params' : [p for n,p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},{'params' : [p for n,p in param_optimizer if  any(nd in n for nd in no_decay)], 'weight_decay': 0.1}]
    
    num_train_steps = int(len(df_train)/config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters,  lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = num_train_steps)
    
    model = nn.DataParallel(model)

    best_accuracy = 0
    
    for epoch in range(config.EPOCHS):
        engine.train_fn(epoch, train_data_loader, model, optimizer, device, scheduler)
        outputs, targets = engine.eval_fn(valid_data_loader,model,device)
        
        #outputs = torch.sigmoid(outputs)
        #outputs2 = np.array(torch.round(outputs))

        outputs = np.array(outputs) >= 0.5
        accuracy = metrics.accuracy_score(targets,outputs)
        print(f"Epoch {epoch}, Validation accuracy: {accuracy}")
        
        # decide how you want to save the checkpoint
        if accuracy > best_accuracy:
            print("Saving Checkpoint!!!")
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_accuracy = accuracy
            
            
if __name__ == "__main__":
    print("Running")
    run()
    
                             
