#This code accepts a json file containing sets of {question ids, question} and returns a dictionary containing the entities identified from those question by TAGME API
# Format of the input file
#   {"id": "train_7769", "question": "Which is the EAGLE id of Hadrian?", "answers": ["dates/lod/129"]}

import requests
import pickle
import time
import json
import csv
import operator
import sys
import multiprocessing
from multiprocessing import Pool
import argparse

Wiki_Threshold=0.0
map1={}	
question={}	
	
def get_wikidata_id(link1):
    link2=link1.split()
    link=link2[0]
    for i in range(1,len(link2)):
        link+='_'+link2[i]
    url = 'https://query.wikidata.org/sparql'
    query = """
    prefix schema: <http://schema.org/>
    SELECT * WHERE {
    <https://en.wikipedia.org/wiki/"""+link+"""> schema:about ?item .
    }
    """
    ct=0
    ids=set()
    while ct<=3:
        try:
            r = requests.get(url, params = {'format': 'json', 'query': query})
            data = r.json()
            #print data
            ids=set()
            for item in data['results']['bindings']:
                ids.add(item['item']['value'])
            return ids	
        except:
            ct+=1
            time.sleep(10)
    return ids	


def get_response(ques,qid,tagme_ent):
    tagme_ent[qid]={}
    tagme_ent[qid]['ques']=ques
    tagme_ent[qid]['spot']=[]
    req_string='https://tagme.d4science.org/tagme/tag?lang=en&include_abstract=true&include_categories=true&gcube-token=9dc5f6c0-3040-411b-9687-75ca53249072-843339462&text='+ques
    try:
        r = requests.get(req_string)
        wiki=r.json()
        #print (wiki)
        annotations=wiki['annotations']
        #print ("Annotations ",wiki,annotations)
        flag=0
        for doc in annotations:
            if doc['rho']>=Wiki_Threshold:
                tagme_ent[qid]['spot'].append((doc['spot'],doc['title'],doc['rho']))
                #print ("Wiki added ",doc['rho'],doc['spot'], doc['dbpedia_categories'])
                #print "Spot, title ", doc['spot'],doc['title']	
                flag=1
    except:
        print("TAGME Problem ", ques)
    return tagme_ent
	

def get_seed_entities_multiprocess(args):
    global question
    proc_id=args[0][0]
    ques_id_lower=args[0][1]
    ques_id_upper=args[0][2]
    tagme_ent=args[0][3]
    for ques_id in range(ques_id_lower,ques_id_upper):
        qid=map1[ques_id]
        ques=question[qid]['text']
        tagme_ent=get_response(ques,qid,tagme_ent)
        tagme_ent[qid]['wikidata']=[]
        for link in tagme_ent[qid]['spot']:
            #print (link)
            wikidata_id=get_wikidata_id(link[1])
            #print (wikidata_id)
            for wiki_id in wikidata_id:
                tagme_ent[qid]['wikidata'].append((wiki_id,link[1]))	
                print("Done for ",qid,len(tagme_ent))
    return tagme_ent

def getquestion(dirs):
    questns = {}
    for line in open(dirs,'r'):
        data = json.loads(line)
        questns[data['id']] = {'text':data['question']}
    return questns


def main(argvs):
    tagme_ent={}
    global question; 
    
    filejson =  argvs.inputfile;
    outputfile = argvs.outputfile;
    question=getquestion(filejson)#pickle.load(open('LCQUAD2.0_Questions_FULL','rb'))
    ct=0
    for qid in question:
        ct+=1
        map1[ct]=qid
    jobs = []

    cores=argvs.core
    loww=0
    chunksize = int((len(question)-loww)/cores)
    
    splits = []
    for i in range(cores):
        splits.append(loww+1+((i)*chunksize))
    splits.append(len(question)+1)
    
    args = []
    for i in range(cores):
        a=[]
        arguments = (i, splits[i], splits[i+1],tagme_ent)
        a.append(arguments)
        args.append(a)
    
    p = Pool(cores)
    tag_list = p.map(get_seed_entities_multiprocess, args)
    p.close()
    p.join()	
    #x=get_seed_entities_multiprocess(5,6)
    
    tag_ent={}
    for it in tag_list:
        for qid in it:
            tag_ent[qid]=it[qid]
    print (len(tag_ent),len(tag_list))
    pickle.dump(tag_ent,open(outputfile,'wb'))


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Command Line argument for the Seed Entity extraction!')
    # Add an argument
    parser.add_argument('--inputfile', type=str, required=True,help='The json file containing the questions.')
    parser.add_argument('--core', type=int, default=8,help='The numbers of cores to use for preprocessing.')
    parser.add_argument('--outputfile', type=str, required=True,help='The name of the pickle file to save the dictionary containing the entities.')
    # Parse the argument
    argvs = parser.parse_args()
    main(argvs)	



		
