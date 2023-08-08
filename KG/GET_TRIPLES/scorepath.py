#Get the score for each path using bert base cased
# append the path triples and score the similarity between the embedding representation with question
# Get triples from CLOCQ
import requests
import pickle
import time
import sys
import json
import re
import argparse
from transformers import BertTokenizer, BertModel
import torch
import string

DEVICE = "cuda"
MAX_LEN =  512
TRAIN_BATCH_SIZE = 50
VALID_BATCH_SIZE = 50
EPOCHS = 10
ACCUMULATION = 4

def get_bert_emb(text):
    inputs = tokenizer(text,max_length=MAX_LEN,truncation=True, return_tensors="pt")
    inputs.to(device)
    outputs = model(**inputs)
    text_emb = outputs[0][0]
    text_emb = text_emb[0]
    return text_emb

predicate_pattern   = re.compile('^P[0-9]*$')
entity_pattern      = re.compile('^Q[0-9]*$')



def getQuestion(qir):
    finalqid =  []
    questions = {}
    for line in open(qir,'r'):
        data = json.loads(line)
        if data['answers'] == None or type(data['answers']) == bool:
            continue
        
        finalqid.append(data['id'])
        questions[data['id']] = data['question']
    print("Got ", len(finalqid), " Questions!")
    return finalqid,questions


def main(argv):
    global  tagme_ent, aida_ent,questionids, done_spo
    global tokenizer, model, device

    questionfile = argv.inputfile
    pathfile = argv.pathfile
    outputdir = argv.outputfile
    bertdir = argv.bertdir
    
    BERT_PATH = argv.bertdir # "/GW/qa2/work/uniqorn/Crawl_Google_docs/LCQuAD_2/KGmode/hdt/BERT/model/bert_base_cased/"
    tokenizer = BertTokenizer.from_pretrained(BERT_PATH, do_lower_case = False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #if torch.cuda.is_available():
    #    device = "cuda:0"
    #else:
    #    device = "cpu"

    device = torch.device(device)
    
    model = BertModel.from_pretrained(BERT_PATH)
    model.to(device)
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

    questionfile = argv.inputfile
    pathfile = argv.pathfile
    outputdir = argv.outputfile 

    wrdr = outputdir+'/score.log'
    fxwd = open(wrdr,'w')
    
    questionids,questions =  getQuestion(questionfile)
    
    count = 0
    #with ope
    for ques in questionids:
        count = count + 1
        print(ques)
        triplelabel = {}
        questns = questions[ques]
        str1=pathfile+'/PathL_'+str(ques)+'.pkl'
        str2=outputdir+'/PathSc_'+str(ques)+'.pkl'
      
        with open(str1, 'rb') as handle:
            tripledict = pickle.load(handle)
        #print('trp dict = ',tripledict)
        triplelabel = {}
        if len(tripledict)<0:
            continue
        #get question embedding
        ques_emb = get_bert_emb(questns)
        

        for keys in tripledict.keys():
            paths = tripledict[keys]
            triplelabel[keys] = {}
            new_triple = []
            score = []
            ftriple = ''

            for path in paths:
                #print('path len = ',len(path))
                if len(path) == 1:
                    #get the path
                    pt1 = path[0]
                    flabel = ' '.join(pt1)
                    trip1 = ' #### '.join(pt1)
                    ftriple = (trip1)

                elif len(path) == 2:
                    #get the first item
                    pt1 = path[0]
                    lbl1 = ' '.join(pt1)
                    trip1 = ' #### '.join(pt1)
                    if lbl1[-1] not in string.punctuation:
                        lbl1 = lbl1+'.'

                    #get the second item
                    pt2 = path[1]
                    lbl2 = ' '.join(pt2)
                    trip2 = ' #### '.join(pt2)

                    flabel = lbl1 + ' '+ lbl2
                    ftriple = (trip1,trip2)
                else:
                    continue
                #score with bert
                context = get_bert_emb(flabel)
                simi = cos(ques_emb, context)
                new_triple.append(ftriple)
                score.append(simi.data.cpu().numpy().item())

            triplelabel[keys]['context'] = new_triple
            triplelabel[keys]['score'] = score
        fxwd.write(ques + "\n")
            
            #print(tripledict)


        with open(str2, 'wb') as handle:
            pickle.dump(triplelabel, handle)
    fxwd.close()


    


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Command Line argument for Triple Extraction from Seed Entities using CLOCQ')
    # Add an argument
    parser.add_argument('--inputfile', type=str, required=True,help='The json file containing the questions.')
    parser.add_argument('--bertdir', type=str, required=True,help='The BERT directory for the Case version.')
    parser.add_argument('--pathfile', type=str, required=True,help='The pkl file containing the Path Labels gotten from CLOCQ..')
    parser.add_argument('--outputfile', type=str, required=True,help='The directory to store triples as returned by CLOCQ.')
    # Parse the argument
    argv = parser.parse_args()
    main(argv)
                 

