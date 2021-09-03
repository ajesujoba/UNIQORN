#read the entity Types [instance|occupation] from the directory as gotten from CLOCQ
#write it as a triple that can be used in UNIQORN sub | 0 | type of | 0 | obj
import requests
import time
import json
import pickle
import re
import kgextract as kex
import torch
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

predicate_pattern   = re.compile('^P[0-9]*$')
entity_pattern      = re.compile('^Q[0-9]*$')


def getQuestion(dirs):
    questns = {}
    for line in open(dirs,'r'):
        data = json.loads(line)
        questns[data['id']] = {'text':data['question']}
    return questns


def getallIDs(ids,x):
    label = ""
    #label= x[0].strip().strip('"').strip()
    if entity_pattern.match(ids):
        if type(x) == list:
            label= x[0].strip().strip('"').strip()
            if label == '' and len(x)>1:
                label = x[1].strip().strip('"').strip()
            for item in x:
                #label = x[0].strip('"').strip()
                if item.strip() == '':
                    continue
                if entity_pattern.match(item.strip()):
                    continue
                elif predicate_pattern.match(item.strip()):
                    continue
                else:
                    label = item.strip().strip('"').strip()
                    break
        else:
            label = x
        return label

    return None
'''
MAIN
'''
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

    for ques in  list(questionids.keys()):
        print(ques)
        question = questionids[ques]['text']
        rdirs = factdir + '/TYPE_'+str(ques)+'.json'
        wdirs = outputdir+'/TYPE_'+str(ques)+'.txt'
        with open(rdirs, 'r') as fp:
            typlist = json.load(fp)
        fs = open(wdirs,'w')
        newtriplefacts = []
        for k,v in list(typlist.items()):
            key = eval(k)
            keyx = list(key)
            if keyx[1] == '':
                keyx[1] = keyx[0]
            sub = keyx[1]
            hum = False
            pred = 'instance of'
            score = 0 
            for item in v:
                if item[0] == 'Q5' or item[1] == ['Q5', 'human']:
                    hum = True
                    continue
                else:
                    if hum == True:
                        pred = 'occupation'
                    else:
                        pred = 'instance of'

                #get the triple
                obj = getallIDs(key[0],item[1])
                verbal = sub.strip() + ' ' + pred.strip() + ' ' + obj.strip()
                #score the verbalized triple
                score = kex.getsimilarity(question,verbal,model)
                #write the
                trips = sub.strip() + ' ### '+ str(score) + ' ### '+pred.strip()+' ### ' + str(score) + ' ### '+obj.strip()
                fs.write(trips)
                fs.write('\n')
        fs.close()


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Command Line argument for Entity Types scoring')
    # Add an argument
    parser.add_argument('--inputfile', type=str, required=True,help='The json file containing the questions.')
    parser.add_argument('--modeldir', type=str, required=True,help='BERT checkpoint to use for scoring')
    parser.add_argument('--triplefile', type=str, required=True,help='The pkl file containing the types directly from CLOCQ')
    parser.add_argument('--outputfile', type=str, required=True,help='The directory to store the BERT scored Type triples for each question using CLOCQ.')
    # Parse the argument
    argv = parser.parse_args()
    main(argv)
