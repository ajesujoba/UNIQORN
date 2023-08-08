# Get triples/facts for the seed entities extracted by the NED tool using CLOCQ
import requests
import pickle
import time
import argparse
import json
import os
from CLOCQInterfaceClient import CLOCQInterfaceClient


def getQuestion(qir):
    qid =  []
    for line in open(qir,'r'):
        data = json.loads(line)
        qid.append(data['id'])
    return qid


def main(argv):
    global  seed_ent, aida_ent,questionids, done_spo
    
    questionfile = argv.inputfile
    questionids =  getQuestion(questionfile)
    
    clocq_host = "https://clocq.mpi-inf.mpg.de/api" # host for client
    clocq_port = "443" # port for client
    clocq = CLOCQInterfaceClient(host=clocq_host, port=clocq_port)
    
    #clocq = CLOCQ()
    count = 0

    #the seed entities
    seedfile = argv.seedfile
    outputdir = argv.outputfile 
    
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    with open(seedfile, 'rb') as f:
        seed_ent = pickle.load(f)
    for ques in questionids:
        count = count + 1
        print(count)
        str1=outputdir+'/SPO_'+str(ques)+'.pkl'
        tripledict = {}
        #print(seed_ent[ques])

        for tup in seed_ent[ques]:
            if tup[0] == None:
                continue
            if tup[0] == 'http://www.wikidata.org/entity/' or  tup[0] == 'http://www.wikidata.org/entity' or tup[0] == 'http://www.wikidata.org/entity/null':
                continue
            id2=tup[0].lstrip('http://www.wikidata.org/entity/')

            ngbs = clocq.get_neighborhood(id2, True)
            tripledict[id2] = ngbs

        #write tripledict to pickle

        with open(str1, 'wb') as handle:
            pickle.dump(tripledict, handle)
 

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Command Line argument for Triple Extraction from Seed Entities using CLOCQ')
    # Add an argument
    parser.add_argument('--inputfile', type=str, required=True,help='The json file containing the questions.')
    parser.add_argument('--seedfile', type=str, required=True,help='The pkl file containing the Seed entities as extracted by tagme and elq (merged).')
    parser.add_argument('--outputfile', type=str, required=True,help='The directory to store triples as returned by CLOCQ.')
    # Parse the argument
    argv = parser.parse_args()
    main(argv)
