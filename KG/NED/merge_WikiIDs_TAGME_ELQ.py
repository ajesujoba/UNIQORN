import argparse
import pickle 
import json

def get_ids_elq(x):
    ids={}
    for item in x['ent']:
        ids[item[1]]={}
        ids[item[1]]['name']=item[0]
        ids[item[1]]['spot']=''
    for item in x['match']:
        if item[1] in ids:
            ids[item[1]]['spot']=item[0]
        else:
            ids[item[1]]={}
            ids[item[1]]['name']=''
            ids[item[1]]['spot']=item[0]
    return ids				


def get_ids_tagme(x):
    ids={}
    spot={}
    for item in x['spot']:
        spot[item[1]]=item[0]
    for item in x['wikidata']:
        ids[item[0]]={}
        if item[1] in spot:
            ids[item[0]]['spot']=spot[item[1]]
            ids[item[0]]['name']=item[1]
        else:
            ids[item[0]]['name']=item[1]
            ids[item[0]]['spot']=''
    return ids				

def getQuestion(qir,qid):
    questns = {}
    for line in open(qir,'r'):
        data = json.loads(line)
        qid.append(data['id'])
    return qid


def main(argv):

    tagmefile = argv.tagmefile
    tagme=pickle.load(open(tagmefile,'rb'))
    elqfile = argv.elqfile
    elq=pickle.load(open(elqfile,'rb'))
    
    print (len(tagme), len(elq))

    questnfile = argv.inputfile
    question = getQuestion(questnfile,[])

    outputfile = argv.outputfile
    
    ct=0;ct1=0;ct2=0;ct3=0;ct4=0
    tagme_elq={}
    for qid in question:
        if qid in elq and qid in tagme:
            ids1=get_ids_elq(elq[qid])
            ids2=get_ids_tagme(tagme[qid])
            tagme_elq[qid]=[]
            for ids in ids1:
                if ids in ids2:
                    if ids1[ids]['spot'].lower()==ids2[ids]['spot'].lower() and ids1[ids]['name'].lower()==ids2[ids]['name'].lower():
                        tagme_elq[qid].append((ids, ids1[ids]['name'], ids1[ids]['spot']))
                        ct1+=1
                    else:
                        if ids1[ids]['spot'].lower()==ids2[ids]['spot'].lower() and ids1[ids]['name'].lower()!=ids2[ids]['name'].lower():
                            tagme_elq[qid].append((ids, ids1[ids]['name']+' | '+ids2[ids]['name'], ids1[ids]['spot']))
                            ct2+=1
                        else:
                            if ids1[ids]['spot'].lower()!=ids2[ids]['spot'].lower() and ids1[ids]['name'].lower()==ids2[ids]['name'].lower():
                                tagme_elq[qid].append((ids, ids1[ids]['name'], ids1[ids]['spot']+' | '+ids2[ids]['spot']))
                                ct3+=1
                            else:
                                tagme_elq[qid].append((ids, ids1[ids]['name']+' | '+ids2[ids]['name'], ids1[ids]['spot']+' | '+ids2[ids]['spot']))
                                ct4+=1
                else:
                    tagme_elq[qid].append((ids, ids1[ids]['name'], ids1[ids]['spot']))
            for ids in ids2:
                if ids not in ids1:
                    tagme_elq[qid].append((ids, ids2[ids]['name'], ids2[ids]['spot']))
        ct+=1
        if ct%100==0:
            print ("Done ",ct, len(tagme_elq))
    pickle.dump(tagme_elq,open(outputfile,'wb'))			
    #print (ct1,ct2,ct3,ct4)

if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description='Command Line argument for the Seed Entity extraction with ELQ!')
    # Add an argument
    parser.add_argument('--inputfile', type=str, required=True,help='The json file containing the questions.')
    parser.add_argument('--tagmefile', type=str, required=True,help='The pkl file containing the Seed entities as extracted by tagme.')
    parser.add_argument('--elqfile', type=str, required=True,help='The json file containing the seed entities as extracted by elq.')
    parser.add_argument('--outputfile', type=str, required=True,help='The directory to store the merge entities from TAGME and ELQ')
    # Parse the argument
    argv = parser.parse_args()
    main(argv)  
