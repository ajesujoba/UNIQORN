import sys
from generate_graphs_from_triples_WITH_HETER import call_main_GRAPH
#from get_GST_from_QKG_with_Frozenset_RANK_DOCS_KG import call_main_GST
import pickle
import time
import multiprocessing
from multiprocessing import Pool
from multiprocessing import Queue
import json
import os
import numpy as NP

import gensim
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize

punc = '''!()-[]{};:'"\, <>./?@#$%^&*_~'''

def tokenize_questions(quest):
    text_tokens = word_tokenize(quest.lower())
    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words('english') and not word in punc]
    return tokens_without_sw


def getQuestion(qir,range1,range2):
    qid =  []
    questns = {}
    done = []
    #done = [line.strip() for line in open('/GW/qa2/work/uniqorn/Crawl_Google_docs/LCQuAD_2/newHeterogen/BERT_HYB_cased/code/HeterAlignments2/done')]
    for line in open(qir,'r'):
        data = json.loads(line)
        if data['answers'] == None or type(data['answers']) == bool:
            #with open("/GW/qa2/work/uniqorn/Crawl_Google_docs/LCQuAD_2/LcQUAD2.0_data/split/num3.txt",'a') as f:
            #f.write(data['id'])
            #f.write("\n")
            continue
        if data['id'] in done:
            continue
        qid.append(data['id'])
        questns[data['id']] = {'text':data['question']}
    #finalqid  = qid[startindext:endindex]
    #print("Got ", len(qid) , " Questions, but returning  ", len(finalqid), " Questions!")
    return qid[range1:range2], questns


def main(argv):
    global config
    global gdict,logdir, logfile, complogdir, uncomplogdir
    global questionids, thequestions
    global queue, compqueue,uncompqueue
    global task, results, writespo
    global writedir, writegraph, writeanswer, writecorner
    data = sys.argv[1]
    logfile = sys.argv[2]
    range1 = int(sys.argv[3])
    range2 = int(sys.argv[4])
    queue = Queue()
    uncompqueue = Queue()
    compqueue = Queue()
    #main method
    filejson = "./../Files/"+data
    #filejson = "/GW/qa2/work/uniqorn/Crawl_Google_docs/LCQuAD_2/KGmode/hdt/Experiments/Philipp/"+data

    #get config
    config = {}
    #read config file
    stream=open("./config_createTableSd.yml", 'r')
    for line in stream:
        if not line.startswith('#') and len(line)>1:
            #print 'line ',line
            line=line.split(':')
            config[line[0]]=line[1].strip()
    print ("Configurations -->", config)

    questionids, thequestions =  getQuestion(filejson,range1,range2)
    print("Got ", len(questionids), " questions")

    gdict = {}
    createTable = True if int(config['createtable']) == 1 else False
    if createTable:
        embedding=config['embedding'] #GLOVE or WORD2VEC
        if embedding=='WORD2VEC':
            model1 = gensim.models.KeyedVectors.load_word2vec_format('./EMBED/GoogleNews-vectors-negative300.bin.gz', binary=True)
            word_vectors = model1.wv
            gdict=word_vectors
        elif embedding == 'WIKI':
            gdict = gensim.models.KeyedVectors.load_word2vec_format('./EMBED/enwiki_20180420_100d.txt',binary=False)
        else:
            glove_file='./../files/glove.6B/glove.6B.300d.txt'
            fg=open(glove_file,'r')
            for line in fg:
                line=(line.strip()).split()
                vec=[]
                for i in range(1,len(line)):
                    vec.append(float(line[i]))
                    gdict[line[0]]=vec
    ##call function process
    jobs = []
    ques_id_lower=0
    ques_id_upper=2
    proc_id=0
    args=[]
    cores=10#0
    loww=0

    chunksize = int((len(questionids)-loww)/cores)
    splits = []
    for i in range(cores):
        splits.append(loww+1+((i)*chunksize))
    splits.append(len(questionids)+1)

    args = []
    for i in range(cores):
        a=[]
        #arguments = (i, splits[i], splits[i+1], model)
        arguments = (i, splits[i], splits[i+1])
        a.append(arguments)
        args.append(a)
    time1=time.time()
    
    p = Pool(cores)
    res_list = p.map(call_process, args)

def call_process(args):
    proc_id=args[0][0]
    ques_id_lower=args[0][1]
    ques_id_upper=args[0][2]
    topcontext=0

    for q in range(ques_id_lower, ques_id_upper):
        try:
            qid = questionids[q-1]
            print(qid)
            questn = thequestions[qid]['text']
            kgquestn_tokens = tokenize_questions(questn)
            
            
            kgspo_dir = config['read_KGfiles']+'ScoreTriples/spo_'+qid+'.txt'
            kgtype_dir = config['read_KGfiles']+'ScoreTriples/TYPE_'+qid+'.txt'
            kgcontextscore_dir = config['read_KGfiles']+'ScoreTriples/tripscore_'+qid+'.txt'
            kgentityname= config['read_KGfiles']+'ScoreTriples/entity_'+qid+'.txt'
            kgaliases = config['read_KGfiles']+'ScoreTriples/aliases_'+qid+'.json'
            kgpath_file = config['read_KGfiles']+'ScoreTriples/e2e_'+qid+'.pkl'
            #cornerstone_file = writecorner+'/LcQUAD_'+qid+'.json'

            #TEXT file
            textspo_dir = config['read_TEXTfiles']+'_SPO2_'+config['benchmark']+'/LcQUAD_'+qid+'.txt'
            textcoocur_dir = config['read_TEXTfiles']+'_COOCCUR2_'+config['benchmark']+'/LcQUAD_cooccur_'+qid+'.txt'
            textcontextscore_dir = config['read_TEXTfiles']+'_CONTEXTSCORE_'+config['benchmark']+'/LcQUAD_snscore_'+qid+'.txt'
            textqt_dir = config['read_TEXTfiles']+'_QT_'+config['benchmark']+'/LcQUAD_'+qid
            texthearst_dir = config['read_TEXTfiles']+'_hearst_'+config['benchmark']+'/LcQUAD2_'+qid+'.json'

            print(texthearst_dir)

            #call_main_GRAPH(qid,kgspo_dir,kgtype_dir,kgcontextscore_dir ,kgentityname, kgquestn_tokens, kgaliases, textspo_dir, textcoocur_dir, textcontextscore_dir,textqt_dir,texthearst_dir,  config,   topcontext,gdict )
            call_main_GRAPH(qid,kgspo_dir,kgtype_dir,kgcontextscore_dir,kgentityname, kgquestn_tokens, kgpath_file, kgaliases, textspo_dir, textcoocur_dir, textcontextscore_dir,textqt_dir,texthearst_dir,  config,   topcontext,gdict)
            #break
        except  FileNotFoundError as err:
            print("OS error: {0}".format(err))
            print("There was an error ")




if __name__ == "__main__":
    main(sys.argv)
