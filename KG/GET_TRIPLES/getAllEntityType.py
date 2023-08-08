#get the Entity in the Context and get their Class as well as occupation if they are human
#used this for context from TAGME and AIDA
import requests
import time
import json
import pickle
import re
import argparse
from CLOCQInterfaceClient import CLOCQInterfaceClient

predicate_pattern   = re.compile('^P[0-9]*$')
entity_pattern      = re.compile('^Q[0-9]*$')

def getquestion(dirs):
    questns = {}
    for line in open(dirs,'r'):
        data = json.loads(line)
        questns[data['id']] = {'text':data['question']}
    return questns


def getallIDs(item):
    ids = item['id']
    x = item['label']
    label = ""
    #label= x[0].strip().strip('"').strip()
    if entity_pattern.match(ids):
        if type(x) == list:
            label= x[0].strip().strip('"').strip()
            for item in x:
                #label = x[0].strip('"').strip()
                if entity_pattern.match(item.strip()):
                    continue
                elif predicate_pattern.match(item.strip()):
                    continue
                else:
                    label = item.strip().strip('"').strip()
                    break
        else:
            label = x
        return (ids,label)

    return None
'''
MAIN
'''
def main(argv):
    #clocq = CLOCQ()
    clocq_host = "https://clocq.mpi-inf.mpg.de/api" # host for client
    clocq_port = "443" # port for client
    clocq = CLOCQInterfaceClient(host=clocq_host, port=clocq_port)
    
    ## load the question file
    questionfile = argv.inputfile
    questionids =  getquestion(questionfile)
    
    triplefile = argv.triplefile
    outputfile = argv.outputfile

    for ques in  list(questionids.keys()):
        print(ques)
        alltypes  = {}
        dirs = outputfile+'/SPO_'+str(ques)+'.pkl'
        wdirs = outputfile+'/TYPE_'+str(ques)+'.json'
        with open(dirs, 'rb') as fp:
            triplefacts = pickle.load(fp)
        newtriplefacts = []
        for key in triplefacts:
            finaltriples = []
            for triplesx in triplefacts[key]:
                currenttriple =  triplesx
                updatedtriple = [getallIDs(item)  for item in currenttriple ]
                newtriplefacts.extend(updatedtriple)

        Ids = list(set([x for x in newtriplefacts if x is not None]))

        for eid in Ids:
            idx = eid[0]
            types = clocq.get_types(idx)
            alltypes[str(eid)] = types
        with open(wdirs, 'w') as fp:
            json.dump(alltypes, fp)

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Command Line argument for Aliases extraction from triples.')
    # Add an argument
    parser.add_argument('--inputfile', type=str, required=True,help='The json file containing the questions.')
    parser.add_argument('--triplefile', type=str, required=True,help='The pkl file containing the triples directly from CLOCQ')
    parser.add_argument('--outputfile', type=str, required=True,help='The directory to store the seed entity names for each question using CLOCQ.')
    # Parse the argument
    argv = parser.parse_args()
    main(argv)
