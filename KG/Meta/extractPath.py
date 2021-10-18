# Get connection between two seed entities 
# Get the connections/paths between the different seed entities. The triples are returned if the seed entities are in 1-hop 
import requests
import pickle
import argparse
import time
import sys
import json
import re
from KnowledgeGraphInterfacePublic import CLOCQ

predicate_pattern   = re.compile('^P[0-9]*$')
entity_pattern      = re.compile('^Q[0-9]*$')

def getItemLabels(item,clocq):
    tupl = []
    for it in item:
        its = it.strip('"').strip()
        if entity_pattern.match(its) or predicate_pattern.match(its):
            tupl.append(clocq.item_to_label(its))
        else:
            tupl.append(its)
    return tuple(tupl)




def getFastLabel(items,clocq):
    it1 = items[0]
    it2 = items[1]
    return ((getItemLabels(it1,clocq),getItemLabels(it2,clocq)))



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
    
    clocq = CLOCQ()
    count = 0

    #the seed entities
    seedfile = argv.seedfile
    outputdir = argv.outputfile 

    with open(seedfile, 'rb') as f:
        tagme_ent = pickle.load(f)
    for ques in questionids:
        count = count + 1
        print(count)
        str1=outputdir+'/Path_'+str(ques)+'.pkl'
        tripledict = {}
        triplelabel = {}
        #print(tagme_ent[ques])
        entityid = []

        for tup in tagme_ent[ques]:
            if tup[0] == None:
                continue
            if tup[0] == 'http://www.wikidata.org/entity/' or  tup[0] == 'http://www.wikidata.org/entity' or tup[0] == 'http://www.wikidata.org/entity/null':
                continue
            id2=tup[0].lstrip('http://www.wikidata.org/entity/')
            entityid.append(id2)


        result = [set((entityid[p1], entityid[p2])) for p1 in range(len(entityid)) for p2 in range(p1+1,len(entityid))]
        for res in result:
            ites = tuple(res)
            #lblid = clocq.item_to_label(ites[0]) + "####" + clocq.item_to_label(ites[1])
            connects = clocq.connectivity_check(ites[0],ites[1])
            if connects <= 0:
                continue
            paths = clocq.connect(ites[0],ites[1])
            if paths==[[[], []]] or paths == [[[], []], [[], []]]:
                continue
            #print(ites)
            #print(paths)
            allitlabels = []
            allitids = []
            cart_prod = tuple()

            if connects == 1:
                if len(paths) == 1:
                    path0 = paths[0]
                    if isinstance(path0, str):
                        allitids.extend([tuple([paths])])
                        continue
                    else:
                        for patho in path0:
                            if patho == []:
                                continue
                            elif isinstance(patho, str):
                                if entity_pattern.match(patho):
                                    ## it only has just one triple
                                    allitids.extend([tuple([path0])])
                                    break
                                else:
                                    ## if it has many paths
                                    if isinstance(patho[0], list):
                                        allitids.extend([tuple(patho)])
                                    else:
                                        allitids.extend([tuple([patho])])
                elif len(paths) > 1:
                    if isinstance(paths[0], list):
                        for itemp in paths:
                            allitids.extend([tuple([itemp])])

                    elif isinstance(paths[0], str):
                        allitids.extend([tuple([patho])])
            elif connects == 0.5:
                #print('We have length of 1!!!!!!!',paths)
                for path in paths:
                    if len(path)==1:
                        path0 = path[0]
                        allitids.extend(set(tuple([path0])))
                        #lblid = clocq.item_to_label(ites[0]) + "####" + clocq.item_to_label(ites[1])
                        #allitlabels.extend(set(tuple([getItemLabels(path0,clocq)])))
                        print('We have length of 1!!!!!!!!')
                        continue
                    elif len(path)==2:
                        path0 = path[0]#[list(item) for item in set(tuple(row) for row in path[0])]
                        path1 = path[1]#[list(item) for item in set(tuple(row) for row in path[1])]
                        if path0 == [] or path1 ==[]:
                            continue
                        cart_prod = tuple([(a,b) for a in path0 for b in path1])
                        #cart_label =tuple([getFastLabel(item,clocq) for item in cart_prod])
                        #tripledict[ites] = cart_prod
                        allitids.extend(cart_prod)
            tripledict[ites] = allitids

        with open(str1, 'wb') as handle:
            pickle.dump(tripledict, handle)

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Command Line argument for Path Extraction from Seed Entities using CLOCQ')
    # Add an argument
    parser.add_argument('--inputfile', type=str, required=True,help='The json file containing the questions.')
    parser.add_argument('--seedfile', type=str, required=True,help='The pkl file containing the Seed entities as extracted by tagme and elq (merged).')
    parser.add_argument('--outputfile', type=str, required=True,help='The directory to store triples as returned by CLOCQ.')
    # Parse the argument
    argv = parser.parse_args()
    main(argv)
                 
