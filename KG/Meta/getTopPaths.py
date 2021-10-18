# Get triples from CLOCQ
# For each seed entity pair, get the top scoring paths to be included during answering

import requests
import pickle
import argparse
import time
import sys
import json
import re
#from networkx import json_graph
#import transformers
predicate_pattern   = re.compile('^P[0-9]*$')
entity_pattern      = re.compile('^Q[0-9]*$')
#clocq = CLOCQ()

def getItemLabels(item):
    tupl = []
    for it in item:
        its = it.strip('"').strip()
        if entity_pattern.match(its) or predicate_pattern.match(its):
            tupl.append(clocq.item_to_label(its))
        else:
            tupl.append(its)
    return tuple(tupl)




def getFastLabel(items):
    it1 = items[0]
    it2 = items[1]
    return (getItemLabels(it1),getItemLabels(it2))



def getQuestion(qir):
    finalqid =  []
    for line in open(qir,'r'):
        data = json.loads(line)
        if data['answers'] == None or type(data['answers']) == bool:
            continue
        finalqid.append(data['id'])
    print("Got ", len(finalqid), " Questions!")
    return finalqid

def prepro(text):
    return text.strip().strip('"').replace(':',' ').strip()

def main(argv):
    global  tagme_ent, aida_ent,questionids, done_spo

    questionfile = argv.inputfile
    questionids =  getQuestion(questionfile)
    #the seed entities
    pathfile = argv.pathfile
    outputdir = argv.outputfile 

    count = 0

    for ques in questionids:
        count = count + 1
        print(ques)
        triplelabel = {}
    
        str1=pathfile+'/PathSc_'+str(ques)+'.pkl'
        str2=outputdir+'/e2e_'+str(ques)+'.pkl'
      
        with open(str1, 'rb') as handle:
            tripledict = pickle.load(handle)
        #print('trp dict = ',tripledict)
        triplelabel = {}
        if len(tripledict)<0:
            continue

        for keys in tripledict.keys():
            #paths = tripledict[keys]
            #new_triple = []
            score = []
            ftriple = ''

            triples = tripledict[keys]['context']
            scores = tripledict[keys]['score']

            k1,k2 = keys
            newkey = frozenset((prepro(k1).lower()+':Entity',prepro(k2).lower()+':Entity'))
            #print(newkey)
            if len(newkey) <=1:
                continue
            if list(newkey)[0] == list(newkey)[1]:
                print("Id is same")
                continue;

            if newkey not in triplelabel:
                #get the maximum context scores
                maxi = max(scores)
                #Get the index of the context with the maximum scores
                max_ind = [i for i, j in enumerate(scores) if j == maxi]
                #get the path the maximum scores
                contxts = [triples[i] for i in max_ind]
                maxsco = [scores[i] for i in max_ind]
                triplelabel[newkey] = {}
                triplelabel[newkey]['context'] = contxts
                triplelabel[newkey]['score'] = maxsco
            else:
                continue

    
        with open(str2, 'wb') as handle:
            pickle.dump(triplelabel, handle)
    


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Command Line argument for Top Path selection for seed entity pairs.')
    # Add an argument
    parser.add_argument('--inputfile', type=str, required=True,help='The json file containing the questions.')
    parser.add_argument('--pathfile', type=str, required=True,help='The pkl file containing the scored paths from BERT.')
    parser.add_argument('--outputfile', type=str, required=True,help='The directory to store the paths')
    # Parse the argument
    argv = parser.parse_args()
    main(argv)

                
