import sys
from generate_graphs_from_triples_WITH_HETER import call_main_GRAPH
from get_GST_from_QKG_with_Frozenset_RANK_DOCS_KG2 import call_main_GST
#from get_GST_from_QKG_with_Frozenset_RANK_DOCS import call_main_GST
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

punc = '''!()-–[]{};:'"\, <>./?@#$%^&*_~`'''

def tokenize_questions(quest):
    text_tokens = word_tokenize(quest.lower())
    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words('english') and not word in punc]
    return tokens_without_sw
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

def getQuestion(qir,range1,range2):
    qid =  []
    questns = {}
    done = set()
    #pickle.load(open("/GW/qa2/work/uniqorn/Crawl_Google_docs/LCQuAD_2/UNIQORN2/KGTEXT/ExperimentResults/TestSet/0.8_0.7_0.7/saved_logs/questions_done_21_4","rb"))
    print("Already completed questions: ",len(done))
    for line in open(qir,'r'):
        data = json.loads(line)
        if data['answers'] == None or type(data['answers']) == bool:
            #with open("/GW/qa2/work/uniqorn/Crawl_Google_docs/LCQuAD_2/LcQUAD2.0_data/split/num3.txt",'a') as f:
            #f.write(data['id'])
            #f.write("\n")
            #continue
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
    stream=open("./config_test.py", 'r')
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
    ##call function process
    jobs = []
    ques_id_lower=0
    ques_id_upper=2
    proc_id=0
    args=[]
    cores=10#8
    loww=0


    parameters = [0.5,0.6,0.7,0.8,0.9,1.0]
    parameters = [1.0,0.9,0.8,0.7,0.6,0.5]
    #from itertools import product
    #candidates = [i for i in permutations(parameters,2)]
    #candidates = [ [e,p,p] for e,p in product(parameters, repeat=2)]

    candidates = [[0.8,0.7,0.7]]
    

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
        newargs = [a+grid for a in args]
        taskname  = "_".join(str(x) for x in grid)
        writedir, writegraph, writeanswer, writecorner = createdirs(taskname)
        logdir = writedir+'/'+logfile
        complogdir = writedir+'/'+logfile+'_completed'
        uncomplogdir = writedir+'/'+logfile+'_uncompleted'
        
        p = Pool(cores)
        res_list = p.map(call_process, newargs)
        #return


def createdirs(taskname):
    #make directory
    if not os.path.exists(config['write_dir']):
        os.mkdir(config['write_dir'])
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
def call_process(args):
    proc_id=args[0][0]
    ques_id_lower=args[0][1]
    ques_id_upper=args[0][2]
    topcontext=5#10
    #print(args)
    #return 

    et = args[1]
    pt = args[2]
    tt = args[3]

    for q in range(ques_id_lower, ques_id_upper):
        t1 = time.time()
        tend = time.time()
        compltedd = True
        try:
            qid = questionids[q-1]
            print(qid)
            questn = thequestions[qid]['text']
            kgquestn_tokens = tokenize_questions(questn)
            
            #KG file 
            kgspo_dir = config['read_KGfiles']+'/ScoreTriples/spo_'+qid+'.txt'
            kgtype_dir = config['read_KGfiles']+'/ScoreTriples/TYPE_'+qid+'.txt'
            kgcontextscore_dir = config['read_KGfiles']+'/ScoreTriples/tripscore_'+qid+'.txt'
            kgentityname= config['read_KGfiles']+'/ScoreTriples/entity_'+qid+'.txt'
            kgaliases = config['read_KGfiles']+'/ScoreTriples/aliases_'+qid+'.json'
            kgpath_file = config['read_KGfiles']+'/ScoreTriples/e2e_'+qid+'.pickle'
            
            #finalXG and answer file 
            finalxg_dir = writegraph+'/Lcquad_'+qid+'.p'
            cornerstone_file = writecorner + '/Lcquad_'+qid+'.json'
            answer_list_file = writeanswer+'/LcQUAD_Answer_list_'+qid


            #TEXT file
            textspo_dir = config['read_TEXTfiles']+'_SPO2_'+config['benchmark2']+'/LcQUAD_'+qid+'.txt'
            textcoocur_dir = config['read_TEXTfiles']+'_COOCCUR2_'+config['benchmark2']+'/LcQUAD_cooccur_'+qid+'.txt'
            textcontextscore_dir = config['read_TEXTfiles']+'_CONTEXTSCORE_'+config['benchmark2']+'/LcQUAD_snscore_'+qid+'.txt'
            textqt_dir = config['read_TEXTfiles']+'_QT_'+config['benchmark2']+'/LcQUAD_'+qid
            texthearst_dir = config['read_TEXTfiles']+'_hearst_'+config['benchmark2']+'/LcQUAD2_'+qid+'.json'
            #call_main_GRAPH(qid,kgspo_dir,kgtype_dir,kgcontextscore_dir,kgentityname, kgquestn_tokens, kgaliases, textspo_dir, textcoocur_dir, textcontextscore_dir,textqt_dir,texthearst_dir,  config,   topcontext,gdict,finalxg=finalxg_dir , finalcorner=cornerstone_file, et=et, pt=pt, tt=tt,we =  writeanswer+'/LcQUAD_'+qid )
            call_main_GRAPH(qid,kgspo_dir,kgtype_dir,kgcontextscore_dir,kgentityname, kgquestn_tokens,kgpath_file, kgaliases, textspo_dir, textcoocur_dir, textcontextscore_dir,textqt_dir,texthearst_dir,  config,   topcontext,gdict,finalxg=finalxg_dir , finalcorner=cornerstone_file, et=et, pt=pt, tt=tt,we =  writeanswer+'/LcQUAD_'+qid )

            #print("It returns ",cornerstones)
            verbose=int(config['verbose']) #verbose
            no_GST=int(config['n_GST']) #Number of GSTs
            h2 = 0
            h4 = 0.05
            t1 = time.time()
            print(finalxg_dir)
            qtype = {}
            gt = []
            
            #graph_node,graph_edge,GST_match_flag, candidate_match_flag1,  top_match_flag1, candidate_match_flag2, top_match_flag2, candidate_match_flag3, top_match_flag3, candidate_match_flag4, top_match_flag4, candidate_match_flag5, top_match_flag5 =
            call_main_GST(finalxg_dir, cornerstone_file, qtype, answer_list_file, no_GST, gdict, verbose,gt,config,h2)
            
            queue.put(qid)
            writer(queue)
            compltedd = True
            tend = time.time() 
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
            #print(qid+','+str(tend-t1))
            #print('Writing completed to file .................................................... &***********************************************8')
            #write to completed File, questionid and the time it took
        else:
            uncompqueue.put(qid+','+str(tend-t1))
            writeuncomplete(uncompqueue)
            continue




if __name__ == "__main__":
    main(sys.argv)
