#Extract triples in the format to be used by UNIQORN 
# s -> p -> o -> Q
# BERT is used to score the triple and they are written to file 
import torch
import pickle
import re
import kgextract as kex
import sys
import json
import os
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
predicate_pattern   = re.compile('^P[0-9]*$')
entity_pattern      = re.compile('^Q[0-9]*$')

def read_kg(dirs):
    verbfacts = []
    with open(dirs, 'rb') as fp:
        triples2 = pickle.load(fp)
    return triples2

def gettriplesforEntity(triples2, key):
    #alltriples =  [triples2[t] for t in triples2]
    verbfacts = []
    flattened_list = triples2[key]
    for xt in flattened_list:
        #get all label
        #broke = False
        t = [{'id':itm['id'],'label': itm['label']} if itm['label']!='' and itm['label']!=[''] else  {'id':itm['id'],'label': itm['id']} for itm in xt]
        tplabels = [x['label'] for  x in t]
        tpids = [x['id'] for  x in t if entity_pattern.match(x['id'].strip())]
        sp = t[0]['id']
        pr = t[1]['id']
        finallabel =[]
        for x in tplabels:
            if type(x) == list:
                possibleitem = x[0].strip().strip('"').strip()
                for item in x:
                    #possibleitem = x[0].strip('"')
                    if entity_pattern.match(item.strip()):
                        continue
                    elif predicate_pattern.match(item.strip()):
                        continue
                    else:
                        possibleitem = item.strip().strip('"').strip()
                        break
                finallabel.append(possibleitem)
            else:
                finallabel.append(x.strip().strip('"').strip())

        verbfacts.append(finallabel)
    return verbfacts

def getQuestion(qir):
    questns = {}
    for line in open(qir,'r'):
        data = json.loads(line)
        questns[data['id']] = {'text':data['question']}
    return questns


def main(argv):
    questionfile = argv.inputfile
    questionids =  getQuestion(questionfile)

    print("Loading BERT!!!")
    bertdir = argv.modeldir
    model = kex.getBERTmodel(bertdir)
    model.to(device)
    print("BERT Loaded!!!")
    
    factdir = argv.triplefile
    outputdir = argv.outputfile

    for qid in list(questionids.keys()):
        print(qid)
        #pathx = '/GW/qa2/work/uniqorn/Crawl_Google_docs/LCQuAD_2/KGmode/hdt/Experiments/elqExperiments/NEDExperiments/ELQ_TAGME/ScoreTriples/spo_'+qid+'.txt'
        ques = questionids[qid]['text']
        dirs = factdir+'/SPO_'+qid+'.pkl'
        try:
            kg = read_kg(dirs)
        except IOError:
            #continue
            pass
        if len(kg) <= 0:
            print("Text length not up content, Continue ....., actual length = ",len(kg))
            #continue
        f1 = open(outputdir+'/spo_'+qid+'.txt','w')
        for keys in kg:
            texts = gettriplesforEntity(kg, keys)
            for tiples in texts:
                tiples2 = ' ### '.join(tiples)
                tripletoscore = ' '.join(tiples)
                score = kex.getsimilarity(ques,tripletoscore,model)
                f1.write(str(score) + ' #### ' + tiples2)
                f1.write('\n')
        f1.close()
        
if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Command Line argument for Aliases extraction from triples.')
    # Add an argument
    parser.add_argument('--inputfile', type=str, required=True,help='The json file containing the questions.')
    parser.add_argument('--modeldir', type=str, required=True,help='BERT checkpoint to use for scoring')
    parser.add_argument('--triplefile', type=str, required=True,help='The pkl file containing the triples directly from CLOCQ')
    parser.add_argument('--outputfile', type=str, required=True,help='The directory to store the BERT scored triples for each question using CLOCQ.')
    # Parse the argument
    argv = parser.parse_args()
    main(argv)
