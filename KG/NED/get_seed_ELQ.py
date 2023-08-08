#Get the Seed Entities from ELQ
#This code accepts a json file containing sets of {question ids, question} and returns a dictionary containing the entities identified from those question by TAGME API
# Format of the input file
#   {"id": "train_7769", "question": "Which is the EAGLE id of Hadrian?", "answers": ["dates/lod/129"]}

import re
import elq.main_dense as main_dense
import argparse
import json
import pickle
from wikidata.client import Client
from urllib.error import HTTPError

#ELQ parameters
#models_path = "models/" # the path where you stored the ELQ models
models_path = "/GW/qa6/work/UNIQORN/Pipeline_Soumajit/KG/BLINK-main/models/"

config = {
    "interactive": False,
    "biencoder_model": models_path+"elq_wiki_large.bin",
    "biencoder_config": models_path+"elq_large_params.txt",
    "cand_token_ids_path": models_path+"entity_token_ids_128.t7",
    "entity_catalogue": models_path+"entity.jsonl",
    "entity_encoding": models_path+"all_entities_large.t7",
    "output_path": "logs/", # logging directory
    "faiss_index": "hnsw",
    "index_path": models_path+"faiss_hnsw_index.pkl",
    "num_cand_mentions": 10,
    "num_cand_entities": 10,
    "threshold_type": "joint",
    "threshold": -4.5,
}

client = Client()
entity_pattern      = re.compile('^Q[0-9]*$')

def getEntityname(eid):
    #entity = client.get(eid, load=True)
    try:
        entity = client.get(eid, load=True)
        return str(entity.label)
    except AssertionError:
        print(eid,' causing error')
        pass
    except KeyError:
        print(eid,' causing error')
    except HTTPError:
        print(eid,' causing error')
    return eid #str(entity.label)

def getQuestion(dirs,qid):
    for line in open(dirs,'r'):
        data = json.loads(line)
        qid.append({'id':data['id'],'text':data['question'].lower()})
    return qid

def main(argv):
    #load the ELQ model
    id2wikidata = json.load(open(models_path + "id2wikidata.json"))
    args = argparse.Namespace(**config)
    models = main_dense.load_models(args, logger=None)

    #load the data
    filejson =  argv.inputfile;
    data_to_link = getQuestion(filejson,[])

    outputfile = argv.outputfile;

    print(len(data_to_link))
    globaldict = {}
    batchsize = 1
    for i in range(0, len(data_to_link), batchsize):
        predictions = main_dense.run(args, None, *models, test_data=data_to_link[i:i+batchsize])
        predictions = {prediction['id']: {'ent':[(getEntityname(id2wikidata.get(prediction['pred_triples'][idx][0])), 'http://www.wikidata.org/entity/'+id2wikidata.get(prediction['pred_triples'][idx][0]), 'http://www.wikidata.org/entity/'+id2wikidata.get(prediction['pred_triples'][idx][0]), '' ,round(prediction['scores'][idx],1)) for idx in range(len(prediction['pred_triples'])) if id2wikidata.get(prediction['pred_triples'][idx][0])!=None and entity_pattern.match(id2wikidata.get(prediction['pred_triples'][idx][0])) ], 'question': prediction['text'], 'match': [(prediction['pred_tuples_string'][idx][1], 'http://www.wikidata.org/entity/'+id2wikidata.get(prediction['pred_triples'][idx][0]),round(prediction['scores'][idx],1)) for idx in range(len(prediction['pred_triples'])) if id2wikidata.get(prediction['pred_triples'][idx][0]) != None and entity_pattern.match(id2wikidata.get(prediction['pred_triples'][idx][0])) ]} for prediction in predictions }
        
        globaldict = {**globaldict,**predictions}

    pickle.dump(globaldict,open(outputfile,'wb'))

if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description='Command Line argument for the Seed Entity extraction with ELQ!')
    # Add an argument
    parser.add_argument('--inputfile', type=str, required=True,help='The json file containing the questions.')
    parser.add_argument('--outputfile', type=str, required=True,help='The name of the pickle file to save the dictionary containing the entities.')
    # Parse the argument
    argv = parser.parse_args()
    main(argv)
