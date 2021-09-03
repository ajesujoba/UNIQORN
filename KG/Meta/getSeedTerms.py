#Get the NERD entity names for each questions
import re
import pickle
import json
import argparse
from KnowledgeGraphInterfacePublic import CLOCQ

entity_pattern = re.compile('^http://www.wikidata.org/entity/Q[0-9]*$')
entity_pattern2 = re.compile('^https://www.wikidata.org/entity/Q[0-9]*$')

def getquestion(dirs):
    questns = {}
    for line in open(dirs,'r'):
        data = json.loads(line)
        questns[data['id']] = {'text':data['question']}
    return questns

def main(argv):
    clocq = CLOCQ()

    #the seed entities
    seedfile = argv.seedfile
    with open(seedfile, 'rb') as f:
        seed_ent = pickle.load(f)

    questionfile = argv.inputfile
        
    questionids =  getquestion(questionfile)

    outputfile = argv.outputfile
    
    for ques in questionids:
        entityname = []
        if ques in seed_ent:
            pass
            #print(seed_ent[ques])
        for tup in seed_ent[ques]:
            entid=tup[0].strip()#.lower()
            #print(entid)
            if entity_pattern.match(entid):
                entQID = entid.lstrip('http://www.wikidata.org/entity/')
                #print(entQID)
                try:
                    text = clocq.item_to_label(entQID.strip()).lower()
                except json.decoder.JSONDecodeError:
                    text = entQID.lower()
                #print(text)
                entityname.append(text)
            elif entity_pattern2.match(entid):
                entQID = entid.lstrip('https://www.wikidata.org/entity/')
                try:
                    text = clocq.item_to_label(entQID.strip()).lower()
                except json.decoder.JSONDecodeError:
                    text = entQID.lower()
                print(text)
                entityname.append(text)
        #write entitname to json
        nameout = outputfile+'/entity_'+ques+'.txt'
        with open(nameout, 'w') as filehandle:
            json.dump(entityname, filehandle)
    print('Extraction of entities!!!')


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Command Line argument for Triple Extraction from Seed Entity names using CLOCQ')
    # Add an argument
    parser.add_argument('--inputfile', type=str, required=True,help='The json file containing the questions.')
    parser.add_argument('--seedfile', type=str, required=True,help='The pkl file containing the Seed entities as extracted by tagme and elq (merged).')
    parser.add_argument('--outputfile', type=str, required=True,help='The directory to store the seed entity names for each question using CLOCQ.')
    # Parse the argument
    argv = parser.parse_args()
    main(argv)
