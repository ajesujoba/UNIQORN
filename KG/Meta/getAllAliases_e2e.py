# Update the predicate aliases bank, we need to add the aliases from the seed connection paths

import requests
import time
import json
import pickle
import re
import sys
from networkx import json_graph
import argparse
from KnowledgeGraphInterfacePublic import CLOCQ

predicate_pattern   = re.compile('^P[0-9]*$')
entity_pattern      = re.compile('^Q[0-9]*$')

def getquestion(dirs):
    questns = {}
    #with open(dirs,'r') as f:
    #quesa = json.load(f)
    for line in open(dirs,'r'):
        data = json.loads(line)
        if data['answers'] == None or type(data['answers']) == bool:
            continue
        questns[data['id']] = {'text':data['question']}
    return questns


def getallIDs(item,clocq):
    ids = item
    if predicate_pattern.match(ids):
        label = clocq.item_to_label(item)
        label = label.strip().strip('"').strip().lower()
        return (ids,label)
    return None

def linearizetriple(triples):
    linetrip = []
    for item in triples:
        linetrip.extend(item)
        #print(item)
    return linetrip
'''
MAIN
'''
def main(argv):
    clocq = CLOCQ() 
    questionfile = argv.inputfile
    questionids =  getquestion(questionfile)

    pathfile = argv.pathfile
    outputdir = argv.outputfile 

    for ques in  list(questionids.keys()):
        print(ques)
        allaliases  = {}
        dirs = pathfile+'/Path_'+str(ques)+'.pkl'
        wdirs = outputdir+'/aliases_'+str(ques)+'.json'
        with open(dirs, 'rb') as fp:
            triplefacts = pickle.load(fp)

        with open(wdirs, 'rb') as fp:
            allaliases = json.load(fp)
            
        newtriplefacts = []
        for key in triplefacts:
            finaltriples = []
            for triplesx in triplefacts[key]:
                finaltriples.extend(linearizetriple(triplesx))
            newtriplefacts.extend(finaltriples)

        
        #currenttriple =  triplesx
        newtriplefacts = set(newtriplefacts)
        #print(newtriplefacts)
        factids = [getallIDs(item,clocq)  for item in newtriplefacts]
        #factids = [getallIDs(x,clocq)  for item in newtriplefacts for x in item ]
        Ids = list(set([x for x in factids if x is not None]))

        #print('prev size = ', len(allaliases))

        for eid in Ids:
            idx = eid[0]
            if eid[1] not in allaliases:
                #print(eid[1], ' not present!!')
                aliases = clocq.item_to_aliases(idx)
                allaliases[eid[1]] = aliases
        #print('New size = ', len(allaliases))
        with open(wdirs, 'w') as fp:
            json.dump(allaliases, fp) 


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Command Line argument for Triple Extraction from Seed Entities using CLOCQ')
    # Add an argument
    parser.add_argument('--inputfile', type=str, required=True,help='The json file containing the questions.')
    parser.add_argument('--pathfile', type=str, required=True,help='The pkl file containing the Seed entities as extracted by tagme and elq (merged).')
    parser.add_argument('--outputfile', type=str, required=True,help='The directory to store triples as returned by CLOCQ.')
    # Parse the argument
    argv = parser.parse_args()
    main(argv)
