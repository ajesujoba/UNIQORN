import sys
from generate_graphs_from_triples import call_main_GRAPH
from get_GST_from_QKG_KG import call_main_GST
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
import utils
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize

punc = '''!()-[]{};:'"\, <>./?@#$%^&*_~'''


def writer(pqueue):
    #for a in iter(pqueue.get, None):
    while not queue.empty():
        with open(logdir,'a') as f:
            a = pqueue.get()
            #print(a)
            f.write(str(a))
            f.write("\n")

def writecomplete(compqueue):
    #for a in iter(pqueue.get, None):
    while not compqueue.empty():
        with open(complogdir,'a') as f:
            a = compqueue.get()
            #print(a)
            f.write(str(a))
            f.write("\n")
def writeuncomplete(uncompqueue):
    #for a in iter(pqueue.get, None):
    while not uncompqueue.empty():
        with open(uncomplogdir,'a') as f:
            a = uncompqueue.get()
            #print(a)
            f.write(str(a))
            f.write("\n")


def tokenize_questions(quest):
    text_tokens = word_tokenize(quest.lower())
    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words('english') and not word in punc]
    return tokens_without_sw

def getQid(qir):
    qids = []
    for line in open(qir,'r'):
        data = json.loads(line)
        qids.append((data['id'],data['question']))
    return qids

def getQuestion(qir):
    qid =  []
    questns = {}
    for line in open(qir,'r'):
        data = json.loads(line)
        if data['answers'] == None or type(data['answers']) == bool:
            #with open("/GW/qa2/work/uniqorn/Crawl_Google_docs/LCQuAD_2/LcQUAD2.0_data/split/num3.txt",'a') as f:
            #f.write(data['id'])
            #f.write("\n")
            #continue
            pass
        qid.append(data['id'])
        questns[data['id']] = {'text':data['question']}
    #finalqid  = qid[startindext:endindex]
    #print("Got ", len(qid) , " Questions, but returning  ", len(finalqid), " Questions!")
    return qid, questns


def getquestion(dirs):
    qid = []
    questns = {}
    with open(dirs,'r') as f:
        quesa = json.load(f)
        #for line in open(dirs,'r'):
        #data = json.loads(line)
    for data in quesa:
        qid.append(data['id'])
        questns[data['id']] = {'text':data['question']}
    return qid, questns

def createdirs(taskname):
    #make directory
    writedir = config['write_dir']+taskname
    if not os.path.exists(writedir):
        os.mkdir(writedir)
    writecorner = writedir+'/_corner_'+taskname
    if not os.path.exists(writecorner):
        os.mkdir(writecorner)
    writeanswer = writedir+'/_ANSWER_'+taskname
    if not os.path.exists(writeanswer):
        os.mkdir(writeanswer)
    #create directory for logging questions without GST
    writegstlog = writeanswer+'/log'
    if not os.path.exists(writegstlog):
        os.mkdir(writegstlog)
    writegraph = writedir+'/_XG_'+taskname
    if not os.path.exists(writegraph):
        os.mkdir(writegraph)
    print("Created all directories")
    return writedir, writegraph, writeanswer,  writecorner

def getstoreddir():
    #get the textkg durectory
    writespo = config['kgcorpus']
    #writeqt = config['kgcorpus']+'_QT_'+config['benchmark']
    return writespo


def main(argv):
    global config
    global gdict,logdir, logfile, complogdir, uncomplogdir
    global questionids, thequestions
    global queue, compqueue,uncompqueue
    global task, results, writespo
    global writedir, writegraph, writeanswer, writecorner 
    
    logfile = sys.argv[2]
    queue = Queue()
    uncompqueue = Queue()
    compqueue = Queue()
    #main method
    config = vars(argv)
    filejson = config['inputfile']
    logfile = config['logfile']
    nocores = config['core']

    questionids, thequestions =  getQuestion(filejson)
    print("Got ", len(questionids), " questions")
    writespo = getstoreddir()

    gdict = {}
    
    '''
    embedding=config['embedding'] #GLOVE or WORD2VEC
    if embedding=='WORD2VEC':
        model1 = gensim.models.KeyedVectors.load_word2vec_format('/GW/qa/work/quest/Word2Vec/GoogleNews-vectors-negative300.bin.gz', binary=True)
        word_vectors = model1.wv
        gdict=word_vectors
    else:
        glove_file='./../files/glove.6B/glove.6B.300d.txt'
        fg=open(glove_file,'r')
        for line in fg:
            line=(line.strip()).split()
            vec=[]
            for i in range(1,len(line)):
                vec.append(float(line[i]))
                gdict[line[0]]=vec
    '''
    jobs = []
    ques_id_lower=0
    ques_id_upper=2
    proc_id=0
    args=[]
    cores=nocores
    loww=0

    topk = config['topk']
    gst_threshold  = [config['n_GST']]
    candidates = [([[topk]],ks) for ks in gst_threshold]
     

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
    for grid in candidates:
        newargs = [a+grid[0]+[grid[1]] for a in args]
        taskname  = "_".join(str(x) for x in grid[0][0])+"_"+str(grid[1])
        writedir, writegraph, writeanswer, writecorner = createdirs(taskname)
        #p = Pool(cores)
        #res_list = p.map(call_process, newargs)
        #f = open(writedir+'/_'+taskname+'_'+logfile, "a")
        logdir = writedir+'/'+logfile
        complogdir = writedir+'/'+logfile+'_completed'
        uncomplogdir = writedir+'/'+logfile+'_uncompleted'
        p = Pool(cores)
        res_list = p.map(call_process, newargs)
        #p = Pool.Process(call_process,args)
        #p.start()
        p.close()
        p.join()
        print(taskname)
    time2=time.time()


def call_process(args):
    #Type_Alignment_flag = int(config['Type_Alignment'])
    #Predicate_Alignment_flag = int(config['Predicate_Alignment'])
    #Entity_Alignment_flag = int(config['Entity_Alignment'])
    stempred = True if int(config['stem_pred']) == 1 else False
    degenerate = True if int(config['degenerate']) == 1 else False
    verbose = int(config['verbose'])
    proc_id=args[0][0]
    ques_id_lower=args[0][1]
    ques_id_upper=args[0][2]
    topcontext = args[1][0]
    no_GST = args[2]
    
    for q in range(ques_id_lower, ques_id_upper):
        t1 = time.time()
        tend = time.time()
        compltedd = True
        try: 
            qid = questionids[q-1]
            print(qid)
            questn = thequestions[qid]['text']
            questn_tokens = tokenize_questions(questn)
            #gt = thequestions[qid]['GT']
            #for q in qids[430:]:
            spo_dir = writespo+'/spo_'+qid+'.txt'
            type_dir = writespo+'/TYPE_'+qid+'.txt'
            entityname= writespo+'/entity_'+qid+'.txt'
            aliases = writespo+'/aliases_'+qid+'.json'
            cornerstone_file = writecorner+'/LcQUAD_'+qid+'.json'
            #qtfile = writeqt+'/LcQUAD_'+qid
            graph_file = writegraph+'/LcQUAD_'+qid+'.json'
            path_file = writespo+'/e2e_'+qid+'.pkl'
            qtype = {}
            gt = []
            answer_path=writeanswer
            answer_list_file = answer_path+'/LcQUAD_Answer_list_'+qid
            #CORNER_TAGMEAIDA/ XG_TAGMEAIDA/
            #terms file is the file containing the NED term from AIDA/TAGME/CLOCQ
            #entity instance and occupation file
            #q_ent, cornerstones =  call_main_GRAPH(spo_file, terms, f2,QKG_file, cornerstone_file,gdict,prune,verbose,gt,config,h1,h2,node_weight_KG)
            #cornerstones =  call_main_GRAPH(spo_dir,type_dir,cornerstone_file, graph_file, entityname, questn_tokens, config)
            call_main_GRAPH(spo_dir,type_dir,cornerstone_file, graph_file,path_file, entityname, questn_tokens, aliases, config,topcontext=topcontext,stempred=stempred,degeneratex=degenerate) 
        
            
            #print("It returns ",cornerstones)
            verbose=int(config['verbose']) #verbose
            #no_GST=int(config['n_GST']) #Number of GSTs
            h2 = 0
            t1 = time.time()
            print(graph_file)
            call_main_GST(graph_file, cornerstone_file, qtype, answer_list_file, no_GST, gdict, verbose,gt,config,h2)
             
            #graph_node,graph_edge,GST_match_flag, candidate_match_flag1,  top_match_flag1, candidate_match_flag2, top_match_flag2, candidate_match_flag3, top_match_flag3, candidate_match_flag4, top_match_flag4, candidate_match_flag5, top_match_flag5 =call_main_GST(graph_file, cornerstone_file, qtype, answer_list_file, no_GST, gdict, verbose,gt,config,h2)

            queue.put(qid)
            writer(queue)
            compltedd = True
            tend = time.time()
            #return 
        except FileNotFoundError as err:
            compltedd = False
            tend = time.time()
            print("OS error: {0}".format(err))
            print("There was an error ")
            queue.put(qid)
            writer(queue)

        if compltedd == True:
            compqueue.put(qid+','+str(tend-t1))
            writecomplete(compqueue)
        else:
            uncompqueue.put(qid+','+str(tend-t1))
            writeuncomplete(uncompqueue)
            continue



        



if __name__ == "__main__":
    argvs = utils.getarguments()
    main(argvs)
