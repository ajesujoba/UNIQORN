#Get path labels 
#Get the surface form  of the Wikidata response of the seed paths
# Get triples from CLOCQ
import requests
import pickle
import time
import argparse
import sys
import json
import re
#from KnowledgeGraphInterfacePublic import CLOCQ
from CLOCQInterfaceClient import CLOCQInterfaceClient

predicate_pattern   = re.compile('^P[0-9]*$')
entity_pattern      = re.compile('^Q[0-9]*$')


def getItemLabels(item,clocq):
    tupl = []
    for it in item:
        its = it.strip('"').strip()
        if entity_pattern.match(its) or predicate_pattern.match(its):
            tupl.append(clocq.get_labels(its)[0])
        else:
            tupl.append(its)
    return tuple(tupl)


def getFastLabel(items,clocq):
    it1 = items[0]
    it2 = items[1]
    return (getItemLabels(it1,clocq),getItemLabels(it2,clocq))



def getQuestion(qir):
    finalqid =  []
    for line in open(qir,'r'):
        data = json.loads(line)
        if data['answers'] == None or type(data['answers']) == bool:
            continue
            
        finalqid.append(data['id'])
    print("Got ", len(finalqid), " Questions!")
    return finalqid



def main(argv):
    global  tagme_ent, aida_ent,questionids, done_spo
    
    questionfile = argv.inputfile
    questionids =  getQuestion(questionfile)
    
    #clocq = CLOCQ()
    clocq_host = "https://clocq.mpi-inf.mpg.de/api" # host for client
    clocq_port = "443" # port for client
    clocq = CLOCQInterfaceClient(host=clocq_host, port=clocq_port)
    count = 0
    
    #the seed entities
    pathfile = argv.pathfile
    outputdir = argv.outputfile 
    
    for ques in questionids:
        count = count + 1
        print(count)
        str1=argv.pathfile+'/Path_'+str(ques)+'.pkl'
        str2=outputdir+'/PathL_'+str(ques)+'.pkl'
        with open(str1, 'rb') as handle:
            tripledict = pickle.load(handle)
        #print('trp dict = ',tripledict)
        triplelabel = {}
        for keys in tripledict:
            #print(keys)
            ites0 = keys[0]
            ites1 = keys[1]
            lblid = (clocq.get_labels(ites0)[0],clocq.get_labels(ites1)[0])

            #print(lblid)
            allPaths = tripledict[keys]
            allitlabels = []
            if allPaths == []:
                continue
            for paths in allPaths:
                #print(paths)
                if len(paths)==1:
                    #lblid = clocq.item_to_label(ites[0]) + "####" + clocq.item_to_label(ites[1])
                    art_label = [tuple(set(tuple([getItemLabels(paths[0],clocq)])))]
                    #print('single art = ', art_label)
                    allitlabels.extend(art_label)
                    #allitlabels.extend(set(tuple([getItemLabels(paths[0],clocq)])))
                elif len(paths)==2:
                    cart_label = set(tuple([getFastLabel(paths,clocq)]))
                    #print(cart_label)
                    allitlabels.extend(cart_label)
            triplelabel[lblid] = allitlabels


        with open(str2, 'wb') as handle:
            pickle.dump(triplelabel, handle)


    


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
