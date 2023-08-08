#this answer question given that there exist a directed context graph where we need to compute alignment, XG ,  question tokens
#python3 answer_fromgraph.py train_6k12.json allsponoalignment103011 ceres 120
#filename, nameoffoldertostoreanswer, nameoftask(tologprogress), time restriction
#filejson in the main method adds to the filename 
import os
import time
import networkx as nx
import sys
import json
import pickle
#sys.path.insert(1, '/GW/qa2/work/uniqorn/Crawl_Google_docs/LCQuAD_2')
#import kgextraction as kex
from statistics import mean 
import gensim
import generate_graphs_from_triples4 as ggtp
from get_GST_from_QKG_with_Frozenset_RANK_DOCS import call_main_GST
import multiprocessing
from multiprocessing import Pool
from multiprocessing import Queue

import signal
from contextlib import contextmanager
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize

punc = '''!()-–[]{};:'"\, <>./?@#$%^&*_~`'''
#time the process
@contextmanager
def timeout(time):
    # Register a function to raise a TimeoutError on the signal.
    signal.signal(signal.SIGALRM, raise_timeout)
    # Schedule the signal to be sent after ``time``.
    signal.alarm(time)
    try:
        yield
    except TimeoutError:
        pass
    finally:
        # Unregister the signal so it won't be triggered
        # if the timeout is not reached.
        signal.signal(signal.SIGALRM, signal.SIG_IGN)

def raise_timeout(signum, frame):
    raise TimeoutError

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

def get_text_file(dirs):
    datas = []
    for line in open(dirs,'r'):
        data = json.loads(line)
        datas.append(data)
    return datas

def getQid(qir):
    qids = []
    for line in open(qir,'r'):
        data = json.loads(line)
        qids.append((data['id'],data['question']))
    return qids
def writList(dir,list12):
    with open (dir,"w")as fp:
        for line in list12:
            fp.write(line+"\n")

def main(argv):
    global config
    global dirt,verbose
    global questionids,compqueue,uncompqueue, uncomplogdir, complogdir, writecontextscores
    global thequestions, timegst, writecorner, writedir, writegraph, writeanswer, logdir, logfile
    global writecontext,  writespo, writeqt, writecooccur, writeanswer,writehearst, writealign, writedirectedXG
    global gdict
    global queue
    queue = Queue()
    uncompqueue = Queue()
    compqueue = Queue()
    starttime = time.time()

    #global modelo #BERT model

    #global qno_to_id
    data = sys.argv[1]
    #taskname = sys.argv[2]
    logfile = sys.argv[2]
    #timegst = int(sys.argv[3])
    start = int(sys.argv[3])
    end  = int(sys.argv[4])

    #prevfile = '/GW/qa2/work/uniqorn/Crawl_Google_docs/LCQuAD_2/GST/results/'+ argv[4] 

    
    
    #main method
    #filejson = "/GW/qa2/work/uniqorn/Crawl_Google_docs/LCQuAD_2/code/restdata/"+data
    filejson = "./../Files/"+data
    
    #get config
    config = {}
    #read config file 
    stream=open("./config_test.yml", 'r')
    for line in stream:
        if not line.startswith('#') and len(line)>1:
            #print 'line ',line
            line=line.split(':')
            config[line[0]]=line[1].strip()
    print ("Configurations -->", config)
    #return 
    writespo, writeqt, writecooccur, writehearst, writealign,writedirectedXG,writecontextscores = getstoreddir()  
    
    questionids, thequestions =  getQuestion(filejson,start,end)
    print("Got ", len(questionids), " questions")
    ct = 0
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
    gdict = {}
    
    parameters = [1.0,0.9,0.8,0.7,0.6,0.5]
    #parameters = [0.5,0.6,0.7,0.8,0.9,1.0]
    #from itertools import product
    ##candidates = [i for i in permutations(parameters,2)]
    #done = []
    #candidates = [ [e,p,p] for e,p in product(parameters, repeat=2) if [e,p,p] not in done]
    candidates = [[0.5,0.9,0.9]]

    #print(candidates)
    #return
    jobs = []
    ques_id_lower=0
    ques_id_upper=2
    proc_id=0
    args=[]
    cores=6
    loww=0
    chunksize = int((len(questionids)-loww)/cores)

    splits = []

    for i in range(cores):
        splits.append(loww+1+((i)*chunksize))
    splits.append(len(questionids)+1)
    args = []
    for i in range(cores):
        a=[]
        arguments = (i, splits[i], splits[i+1])
        a.append(arguments)
        args.append(a)
    
    time1 = time.time()
    #p = Pool(cores)
    for grid in candidates:
        #p = Pool(cores)
        pstart = time.time()
        newargs = [a+grid for a in args]
        #print(newargs)
        taskname  = "_".join(str(x) for x in grid)
        writedir, writegraph, writeanswer, writecorner = createdirs(taskname)
        print('writecorner = ',writecorner)
        f = open(writedir+'/_'+taskname+'_'+logfile, "a")
        logdir = writedir+'/'+logfile
        complogdir = writedir+'/'+logfile+'_completed'
        uncomplogdir = writedir+'/'+logfile+'_uncompleted'
        
        p = Pool(cores) 
        res_list = p.map(call_process, newargs)
        #p = Pool.Process(call_process,args)
        #p.start()
        p.close()
        p.join()
        pend = time.time()
        f.write(str(pend - pstart))
        f.write("\n")

    time2=time.time()

    print("++++++++++++++++++++++++>>>>>>>>>>>>>>>>>>>>",res_list)
    #print(mrr_result(1, res_list,thequestions))
    endtime = time.time()
    print(starttime, endtime, starttime-endtime)
    
    #call_process([[0,1,len(questionids),model]])

    #call_process(writecontext,  writespo, writeqt, writecorner, writecooccur, writegraph)



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

def getstoreddir():
    #get the textkg durectory
    
    writespo = config['write_textkg']+'_SPO2_'+config['benchmark']
    writeqt = config['write_textkg']+'_QT_'+config['benchmark']
    #writecorner = config['write_textkg']+'_CORNER_'+config['benchmark']
    writecooccur = config['write_textkg']+'_COOCCUR2_'+config['benchmark']
    writecontextscores = config['write_textkg']+'_CONTEXTSCORE_'+config['benchmark']
    writehearst = config['write_textkg']+'_hearst_'+config['benchmark']
    writealign = config['write_textkg']+'_align_'+config['benchmark']
    writedirectedXG = config['write_textkg']+'_DXG_'+config['benchmark']
    
    return writespo, writeqt, writecooccur, writehearst, writealign, writedirectedXG,writecontextscores



#Type_Alignment_flag = int(config['Type_Alignment'])
#Predicate_Alignment_flag = int(config['Predicate_Alignment'])
#Entity_Alignment_flag = int(config['Entity_Alignment'])


def getQuestion(qir,start, end):
    question = {}
    qid =  []
    #done = pickle.load(open("./ExperimentResults/UniqornTest/0.5_0.9_0.9/saved_logs/questions_done","rb"))
    #prevfile = '/GW/qa2/work/uniqorn/Crawl_Google_docs/LCQuAD_2/UNIQORN/TEXT/typedcorner/ExperimentResults/UniqornTest/0.5_0.9_0.9/done'
    #done = [line.strip().split(',')[0] for line in open(prevfile, 'r')] 
    for line in open(qir,'r'):
        data = json.loads(line)
        if data['answers'] == None or type(data['answers']) == bool:
            #with open("/GW/qa2/work/uniqorn/Crawl_Google_docs/LCQuAD_2/LcQUAD2.0_data/split/num3.txt",'a') as f:
            #f.write(data['id'])
            #f.write("\n")
            continue
        #if data['id'] in done:
        #    continue
        #else:
        #print("data not processed", data['id'])

        question[data['id']]={}
        qid.append(data['id'])
        question[data['id']]['text']=data['question']
        #print(data['answers'])
        if data['answers'] == None:
            question[data['id']]['GT']=[None]
            continue

        if type(data['answers']) == bool:
            question[data['id']]['GT']=[data['answers']]
            continue

        if type(data['answers']) == list and type(data['answers'][0]) ==  list:
            question[data['id']]['GT']=list(set([y for x in data['answers'] for y in x]))
            continue

        question[data['id']]['GT']=list(data['answers'])
        
    return qid[start:end], question


def call_process(args):
    res_list=[]
    Type_Alignment_flag = int(config['Type_Alignment'])
    Predicate_Alignment_flag = int(config['Predicate_Alignment'])
    Entity_Alignment_flag = int(config['Entity_Alignment'])
    verbose = int(config['verbose'])
    Add_Type_flag = int(config['Add_Type'])
    degenerate = int(config['Degenerate'])
    addcoccur  =  True if config['Add_Cooccur']=='1' else False
    proc_id=args[0][0]
    ques_id_lower=args[0][1]
    ques_id_upper=args[0][2]
    #print("args = ",args)
    et = args[1]
    pt = args[2]
    tt = args[3]
    topcontext = 5
    #print(args)
    #print(et,pt,tt)

    #model = args[0][3]
    for q in range(ques_id_lower, ques_id_upper):
        qid = questionids[q-1] 
        questn = thequestions[qid]['text']
        gt = thequestions[qid]['GT']
        #for q in qids[430:]:
        #dirtloc = dirt+qid+".txt"
        #dirtfin = finaldir+qid+".txt"
        print(qid)
        finalhearstdict = []
        wholehearst = []
        hearstdicts = dict()
        unique_SPO_dict = dict()
        unique_SPO_dict2 = dict()
        compltedd = False
        #with timeout(timegst):
        t1 = time.time()
        tend = time.time()
        try:
            #data_file = get_text_file(dirtloc)
        
            questn_token = []
            #wholehearst = []
            justnp = []
            doc = 0
            doc_title = qid+".txt"
            #read the SPOs and COOCCUR
            f22=open(writespo+'/LcQUAD_'+qid+'.txt','r')
            f33=open(writecooccur+'/LcQUAD_cooccur_'+qid+'.txt','r')
            f44=open(writecontextscores+'/LcQUAD_snscore_'+qid+'.txt','r')
            contextscoDicts = {}

            if topcontext>0:
                unique_SPO_dict,topsnips,_ = ggtp.getSPOtopk(f22,f33,f44,topcontext,addcocor=addcoccur)
            else:
                unique_SPO_dict = ggtp.getSPO(f22,f33,addcocor=addcoccur)



            t11 = time.time()


            qtfile = writeqt+'/LcQUAD_'+qid
            #questn_tokens = pickle.load(open(qtfile,'rb'))
            questn_tokens = tokenize_questions(questn)

            cornerstone_file = writecorner+'/LcQUAD_'+qid
            #allentpred = []
            #allentpred.extend(list(unique_SPO_dict.keys()))
            #allwords = list(set([j for i in allentpred for j in i])) #allwords includes the predicates and entities to be used to find the connerstone
            print("question tokens == === === = ", questn_tokens)
            #print('all connerstone words ==== > ',allwords)
            
            #connerstone = [words for words in allwords if set(kex.tokenize_text(kex.parsetextspacy(words))).intersection(set(questn_tokens))  ]
            #connerstone = pickle.load(open(cornerstone_file,'rb'))
            #write the cornerstone
            #pickle.dump(connerstone,open(cornerstone_file,'wb'))

            #print("connerstone =======>>> ", connerstone)
            G,cornerdict, predcornerdict = ggtp.build_graph_from_triple_edges2(unique_SPO_dict,questn_tokens)
            #print("hearst =======>>>> ",final_hearst)
            G = ggtp.update_edge_weight(G)

            if Add_Type_flag == 1:
                hearst_file  = writehearst+'/LcQUAD2_'+qid+'.json'
                final_hearst = json.load(open(hearst_file,'r'))['hearst']
                G = ggtp.add_type_edges(G,final_hearst,topsnips)
            else:
                Type_Alignment_flag=0
            #read directed XG

            if degenerate == 1:
                #print("Setting weight to 1 ")
                G = ggtp.degenerate(G)


            '''
            if Type_Alignment_flag == 1 or Predicate_Alignment_flag == 1:
                #Read Glove embeddings
                g_pred,g_ent,g_type,g_ques=ggtp.read_glove(G,[],gdict)
            '''
            #print("\n\nJoba Size of the graph directed",len(G.nodes()),len(G.edges()))

             
            if Predicate_Alignment_flag == 1:
                #Add relation alignment edges from glove embeddings
                print ("\n\nAdding predicate alignment edges\n\n")
                predlookup  = pickle.load(open(writealign+'/LcQUAD_'+qid+'predicate.pickle','rb'))
                #print (" Pred lookup = ", type(predlookup), ' empty?? ',bool(predlookup))
                #print(predlookup)
                G = ggtp.add_predicate_alignment_edges2(G,predlookup,pt)
        
            if Type_Alignment_flag==1:
                print ("\n\nAdding type alignment edges\n\n")
                typelookup  = pickle.load(open(writealign+'/LcQUAD_'+qid+'type.pickle','rb'))
                G = ggtp.add_type_alignment_edges2(G,typelookup,tt)

            if Entity_Alignment_flag==1:
                print ("\n\nAdding entity alignment edges\n\n")
                entitylookup  = pickle.load(open(writealign+'/LcQUAD_'+qid+'entity.pickle','rb'))
                G = ggtp.add_entity_alignment_edges2(G,entitylookup,et)
            
                if verbose:
                    print("\n\nSize of the graph directed",len(G.nodes()),len(G.edges()))
            
            #visualize_graph(G)
            #G=G2.to_undirected() #make QKG Undirected
            
            G=ggtp.directed_to_undirected(G)       
            #print ("Size of the graph ",len(G.nodes()),len(G.edges()))
            if len(G.nodes())>0:
                G=max(nx.connected_component_subgraphs(G), key=len)
                print ("\n\nSize of the graph ",len(G.nodes()),len(G.edges()))


            G = ggtp.add_nodes_weights(G,cornerdict, predcornerdict,meanval=False)
            corner = {}
            corner = ggtp.getcornerstone(G,questn_tokens)
            #print("corner = ", corner)
            print("cornerstone_file = ", cornerstone_file)
            pickle.dump(corner,open(cornerstone_file,'wb'))
            
        
            #cornerstone_file = writecorner+'/LcQUAD_'+qid
            #write the cornerstone
            #print("Writ the corner")
            #writList(cornerstone_file+'.txt',corner)
            #pickle.dump(corner,open(cornerstone_file,'wb'))

            
            #write the graph
            QKG_file = writegraph+'/LcQUAD_'+qid+'.p'
            with open(QKG_file, 'wb') as f:
                nx.write_gpickle(G,f)
        

            QKG_match_flag=0
            for gt1 in gt:
                if type(gt1) == bool or gt1 == None:
                    QKG_match_flag = 0
                    break
                for n in G.nodes():
                    nn=n.split(':')
                    if len(nn)>0:
                        if nn[0]==(gt1.lower()):
                            QKG_match_flag=1
                    else:
                        if n==(gt1.lower()):
                            QKG_match_flag=1


            
            #call main GST here 
            qtype = ""
            h4 = 0.05
            gt = thequestions[qid]['GT']
            #no_GST = 10
            verbose=int(config['verbose']) #verbose
            no_GST=int(config['n_GST']) #Number of GSTs
            answer_path=writeanswer
            answer_list_file=answer_path+'/LcQUAD_Answer_list_'+str(qid)
            t2=time.time()
            nogstpath = answer_path+'/'+str(qid)+'_GST'
            
            call_main_GST(QKG_file, cornerstone_file, qtype, answer_list_file, no_GST, gdict, verbose,gt,config,h4,0,qid,nogstpath)
            

            #graph_node,graph_edge,GST_set,GST_match_flag, candidate_match_flag1,candidate_match_flag_top1, top_match_flag1, candidate_match_flag2,candidate_match_flag_top2, top_match_flag2, candidate_match_flag3,candidate_match_flag_top3, top_match_flag3, candidate_match_flag4,candidate_match_flag_top4, top_match_flag4, candidate_match_flag5,candidate_match_flag_top5, top_match_flag5 =call_main_GST(QKG_file, cornerstone_file, qtype, answer_list_file, no_GST, gdict, verbose,gt,config,h4,0,qid,nogstpath)
            
            '''
            print("graph nodes",graph_node)
            try:
                str1=answer_list_file+'_node_wt'
                answer_list1=pickle.load(open(str1,'rb'))
            except:
                print ("No answer pickle")
                answer_list1=[]
            try:
                str1=answer_list_file+'_tree_cost'
                answer_list2=pickle.load(open(str1,'rb'))
            except:
                print ("No answer pickle")
                answer_list2=[]
            try:
                str1=answer_list_file+'_tree_count'
                answer_list3=pickle.load(open(str1,'rb'))
            except:
                print ("No answer pickle")
                answer_list3=[]
            try:
                str1=answer_list_file+'_corner_dist'
                answer_list4=pickle.load(open(str1,'rb'))
            except:
                print ("No answer pickle")
                answer_list4=[]
            try:
                str1=answer_list_file+'_corner_dist_wt'
                answer_list5=pickle.load(open(str1,'rb'))
            except:
                print ("No answer pickle")
                answer_list5=[]
            t3 = time.time()
            context_match_flag = 1
            print ("DONE GST Algorithm...",proc_id,qid,context_match_flag,QKG_match_flag,GST_match_flag, candidate_match_flag1, top_match_flag1, candidate_match_flag2, top_match_flag2, candidate_match_flag3, top_match_flag3, candidate_match_flag4, top_match_flag4, candidate_match_flag5, top_match_flag5, candidate_match_flag_top1,candidate_match_flag_top2,candidate_match_flag_top3,candidate_match_flag_top4,candidate_match_flag_top5,t2-t1,t3-t2,t3-t1,t11-t1)
            #res_list = []
            #res_list.append((qid,answer_list1,answer_list2,answer_list3,answer_list4,answer_list5,graph_node,graph_edge,context_match_flag,QKG_match_flag, GST_match_flag, candidate_match_flag1, top_match_flag1,candidate_match_flag2, top_match_flag2,candidate_match_flag3, top_match_flag3,candidate_match_flag4, top_match_flag4,candidate_match_flag5, top_match_flag5,t2-t1,t3-t2))#, GST_set))
            '''

            res_list.append(qid)
            #print(res_list)
        
            #break

            
            doc += 1
            queue.put(qid)
            writer(queue)
            compltedd = True
            tend = time.time()
        except  FileNotFoundError as err:#TypeError as err:# as err: # IOError as err:
            compltedd = False
            tend = time.time()
            print("OS error: {0}".format(err))
            print("There was an error ")
            queue.put(qid)
            writer(queue)
            #continue
        
        if compltedd == True:
            compqueue.put(qid+','+str(tend-t1))
            writecomplete(compqueue)
            #write to completed File, questionid and the time it took
        else:
            uncompqueue.put(qid+','+str(tend-t1))
            writeuncomplete(uncompqueue)
            continue
            #write to uncompleted file
        
    
    
    
    return res_list

def rank_list_answers(list_tuples):
    val = []
    ranklist = []
    rank = 0
    skip = 0
    prev = None
    res = []
    for l in list_tuples:
        val.append(l[1])
        if l[1] == prev:
            skip += 1
        else:
            rank += skip + 1
            skip = 0
        res.append( (l[0], l[1], rank) )
        prev = l[1]
    size = len(set(val))
    return res, size

def group_rank(ranked_list, size):
    newrank = x = [[] for i in range(size)]
    #print(newrank)
    prev = 0
    cnt = 0
    index = 0
    for x in ranked_list:
        #print(x)
        if prev==0:
            index = 0
            newrank[index].append(x)
            prev = x[2]
            continue
        elif prev == x[2]:
            newrank[index].append(x)
            prev = x[2]
            continue
        elif  x[2] > prev:
            index += 1
            newrank[index].append(x)
            prev = x[2]
            continue
    #print(index)
    return newrank

def gettopk(newranks, rank=5):
    #print(newranks)
    answeroptions = []
    if rank == 1:
        answeroptions.extend(newranks[0])
    elif rank > 1:
        count = 0
        for x in newranks:
            lenrank = len(x)
            if count < rank:
                answeroptions.extend(x)
                count += lenrank
    return answeroptions

def mrr_result(rank_type, res_list,question):
    sum_mrr1=[]
    sum_mrr3=[]
    sum_mrr5=[]
    
    if rank_type==1:
        rank_str='_node_wt.txt'
        aid=1
        can_flag_id=11
        top_flag_id=12
    if rank_type==2:
        rank_str='_tree_cost.txt'
        aid=2
        can_flag_id=13
        top_flag_id=14
    if rank_type==3:
        rank_str='_tree_count.txt'
        aid=3
        can_flag_id=15
        top_flag_id=16
    if rank_type==4:
        rank_str='_corner_dist.txt'
        aid=4
        can_flag_id=17
        top_flag_id=18
    if rank_type==5:
        rank_str='_corner_dist_wt.txt'
        aid=5
        can_flag_id=19
        top_flag_id=20
    for res_list1 in res_list:#range(0,proc_id):
        res_list1=sorted(res_list1,key=lambda x:x[0])
        for tuple1 in res_list1:
            qid=tuple1[0]
            answer_list=tuple1[aid]

            mrr1=get_metric(question[qid]['GT'],answer_list,1)
            mrr3=get_metric(question[qid]['GT'],answer_list,3)
            mrr5=get_metric(question[qid]['GT'],answer_list,5)
            
            sum_mrr1.append(mrr1)
            sum_mrr3.append(mrr3)
            sum_mrr5.append(mrr5)
    return float(sum(sum_mrr1)),float(sum(sum_mrr3)),float(sum(sum_mrr5))


def write_results(rank_type,res_list,benchmark,corpora,question,time1,time2,result_file,error_file,h1,h2,h3,h4):
    sum_mrr1=[]
    sum_mrr3=[]
    sum_mrr5=[]
    sum_node=[]
    sum_edge=[]
    time_qkg=[]
    time_gst=[]
    #print res_list
    result_file_last=(result_file.split('/'))[2]
    hpath='0_'+(str(h3).split('.'))[1]+'_0_'+(str(h4).split('.'))[1]+'_0_'+(str(h1).split('.'))[1]+'_0_'+(str(h2).split('.'))[1]
    result_file_path='./results/312_'+str(config['n_GST'])+'_'+hpath
    try:
        os.mkdir(result_file_path)
    except OSError:
        print ("Creation of the directory %s failed" % result_file_path)
    else:
        print ("Successfully created the directory %s " % result_file_path)
    result_file=result_file_path+'/'+result_file_last
    print ("Updated Result File ",result_file)
    if rank_type==1:
        rank_str='_node_wt.txt'
        aid=1
        can_flag_id=11
        top_flag_id=12
    if rank_type==2:
        rank_str='_tree_cost.txt'
        aid=2
        can_flag_id=13
        top_flag_id=14
    if rank_type==3:
        rank_str='_tree_count.txt'
        aid=3
        can_flag_id=15
        top_flag_id=16
    if rank_type==4:
        rank_str='_corner_dist.txt'
        aid=4
        can_flag_id=17
        top_flag_id=18
    if rank_type==5:
        rank_str='_corner_dist_wt.txt'
        aid=5
        can_flag_id=19
        top_flag_id=20
    
    fname=result_file+rank_str
    resf=open(fname,'w')
    error_flags={}
    errors=[]
    for res_list1 in res_list:#range(0,proc_id):
        res_list1=sorted(res_list1,key=lambda x:x[0])
        for tuple1 in res_list1:
            qid=tuple1[0]
            answer_list=tuple1[aid]
            graph_node=tuple1[6]
            graph_edge=tuple1[7]
            context_match_flag=tuple1[8]
            QKG_match_flag=tuple1[9]
            GST_match_flag=tuple1[10]
            candidate_match_flag=tuple1[can_flag_id]
            top_match_flag=tuple1[top_flag_id]
            time_qkg.append(tuple1[21])
            time_gst.append(tuple1[22])
            error_flags[qid]=(context_match_flag,QKG_match_flag, GST_match_flag, candidate_match_flag, top_match_flag)
            errors.append((context_match_flag,QKG_match_flag, GST_match_flag, candidate_match_flag, top_match_flag))
            
            resf.write("\n\n====================================================================>")
            str1="\n\nResults for Question ->"+str(qid)+' '+question[qid]['text']
            resf.write(str1.encode('utf-8'))
            str1="\nGround truth answers ->"
            for item in question[qid]['GT']:
                str1+='\n'+item
            resf.write(str1.encode('utf-8'))
                
            resf.write("\n\nQUEST All Answers -->\n")
            for i in range(0,len(answer_list)):
                resf.write(str(i)+' '+str(answer_list[i][0].encode('utf-8'))+' '+str(answer_list[i][2]))
                resf.write('\n')
                resf.write("\n\nQUEST Error Reason -->"+str(context_match_flag)+' '+str(QKG_match_flag)+' '+str(GST_match_flag)+' '+str(candidate_match_flag)+' '+str(top_match_flag)+'\n')
                #if len(answer_list)>0 and len(answer_list[0])>1:
                #       resf.write("\n\n Supporting Facts for top 1 -->"+str(answer_list[0][1]))
                resf.write("\n\nQUEST Undirected QKG size -->"+str(graph_node)+' '+str(graph_edge)+'\n')
                resf.write("\n\nQUEST Time taken -->"+str(tuple1[21])+' '+str(tuple1[22])+' '+str(tuple1[21]+tuple1[22])+'\n')
                sum_node.append(graph_node)
                sum_edge.append(graph_edge)
                
                mrr1=get_metric(question[qid]['GT'],answer_list,1)
                mrr3=get_metric(question[qid]['GT'],answer_list,3)
                mrr5=get_metric(question[qid]['GT'],answer_list,5)
                    
                resf.write("\n\nFor Answer, MRR@1, MRR@3, MRR@5 "+str(mrr1)+' '+str(mrr3)+' '+str(mrr5))
                #resf.write("\nFor Supporting Sentences, EM, Precision, Recall, F1 for top 1 "+str(sem1)+' '+str(spr1)+' '+str(sre1)+' '+str(sf1))
                #resf.write("\nJointly, EM, Precision, Recall, F1 for top 1 "+str(jem1)+' '+str(jpr1)+' '+str(jre1)+' '+str(jf1))
                #print "Time taken for this question -->",t2-t1
                resf.write("\n====================================================================>\n\n")
                sum_mrr1.append(mrr1)
                sum_mrr3.append(mrr3)
                sum_mrr5.append(mrr5)


        time3=time.time()

        #print "Full processed , half processed ",len(sum_em1),qcount-len(sum_em1)
        resf.write("\n----------------------ALL QUESTIONS--------------------------------")
        resf.write("\n\nAVERAGE MRR@1, MRR@3, MRR@5 - "+str(float(sum(sum_mrr1))/float(len(sum_mrr1)))+' '+str(float(sum(sum_mrr3))/float(len(sum_mrr3)))+' '+str(float(sum(sum_mrr5))/float(len(sum_mrr5))))
        resf.write("\n\nAVERAGE Undirected QKG size - "+str(float(sum(sum_node))/float(len(sum_node)))+' '+str(float(sum(sum_edge))/float(len(sum_edge))))
        resf.write("\nOverall Processing Time taken - Total, Avg "+str(time2-time1)+' '+str(float(time2-time1)/float(len(sum_mrr1))))
        resf.write("\nOverall Printing Time taken - Total, Avg "+str(time3-time2))

        resf.write("Partwise time for QKG "+str(float(sum(time_qkg))/float(len(time_qkg)))+'\n')
        resf.write("Partwise time for GST "+str(float(sum(time_gst))/float(len(time_gst)))+'\n')

        pname=error_file+rank_str#'./results/errors/'+corpora+'_Error_Analysis_flags_Default_type_Relaxed'+rank_str
        pickle.dump(error_flags,open(pname,'w'))


        resf.write("\n\n-----------------------Error Analysis-----------------------------")
        sum_er1=0
        sum_er2=0
        sum_er3=0
        sum_er4=0
        sum_er5=0
        for tup in errors:
            sum_er1+=tup[0]
            sum_er2+=tup[1]
            sum_er3+=tup[2]
            sum_er4+=tup[3]
            sum_er5+=tup[4]
        resf.write('\n'+str(float(sum_er1)/float(len(errors)))+' '+str(float(sum_er2)/float(len(errors)))+' '+str(float(sum_er3)/float(len(errors)))+' '+str(float(sum_er4)/float(len(errors)))+' '+str(float(sum_er5)/float(len(errors))))
        print (sum_er1,sum_er2,sum_er3,sum_er4,sum_er5,len(question),len(errors))
        for qid in error_flags:
            tuple1=error_flags[qid]
            context_match_flag=tuple1[0]
            QKG_match_flag=tuple1[1]
            GST_match_flag=tuple1[2]
            candidate_match_flag=tuple1[3]
            top_match_flag=tuple1[4]
            #typ=tuple1[5]
            
            if context_match_flag==0 and QKG_match_flag==0 and GST_match_flag==0 and candidate_match_flag==0 and top_match_flag==0:
                p1+=1
            else:
                if context_match_flag==1 and QKG_match_flag==0 and GST_match_flag==0 and candidate_match_flag==0 and top_match_flag==0:
                    p2+=1
                else:
                    if context_match_flag==1 and QKG_match_flag==1 and GST_match_flag==0 and candidate_match_flag==0 and top_match_flag==0:
                        p3+=1
                    else:
                        if context_match_flag==1 and QKG_match_flag==1 and GST_match_flag==1 and candidate_match_flag==0 and top_match_flag==0:
                            p4+=1
                        else:
                            if context_match_flag==1 and QKG_match_flag==1 and GST_match_flag==1 and candidate_match_flag==1 and top_match_flag==0:
                                p5+=1
                            else:
                                if context_match_flag==1 and QKG_match_flag==1 and GST_match_flag==1 and candidate_match_flag==1 and top_match_flag==1:
                                    p6+=1
                                else:
                                    p7+=1
        print (p1,p2,p3,p4,p5,p6,p7,p1+p2+p3+p4+p5+p6+p7)
        a_len=p1+p2+p3+p4+p5+p6+p7
        
        resf.write('\n'+str(float(p1)/float(a_len))+' '+str(float(p2)/float(a_len))+' '+str(float(p3)/float(a_len))+' '+str(float(p4)/float(a_len))+' '+str(float(p5)/float(a_len))+' '+str(float(p6)/float(a_len))+' '+str(float(p7)/float(a_len)))
        
        resf.close()


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude) 
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def metric_max_prec_over_ground_truths(prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = precision_score(prediction, str(ground_truth))
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def precision_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    return precision

def get_metric(GT1,answer_list,topk):
    if GT1[0]==None or type(GT1[0]) == bool:
        return 0
    GT=process_ground_truth(GT1)
    print ("\nLowered ground truth --> ", GT)
    
    rank=0
    answer_list1=[]
    for j in range(0,len(answer_list)):
        if j>0:
            if answer_list[j][2] < answer_list[j-1][2]:
                rank+=1
                #answer_list[j]=tuple(list(answer_list[j]).append(rank))
            answer_list1.append((answer_list[j][0],answer_list[j][1],answer_list[j][2],rank))
    answer_list=answer_list1
    
    mrr=0.0
    flag=0
    if topk==5:
        topk=len(answer_list)
    for i in range(0,len(answer_list)):
        if answer_list[i][3]<=topk-1: #rank 1
            ans1=answer_list[i][0].split('|')
            for s in ans1:
                s=s.split(':')[0]
                s=s.strip()
                s=s.strip(',')
                #s=s.encode('utf-8')
                for GTT in GT:
                    if GTT==s: #Order preserving
                        mrr+=1.0/float(answer_list[i][3]+1)
                        flag=1
                        break
                if flag==1:
                    break
            if flag==1:
                break
        if flag==1:
            break
    return mrr


def get_metric(GT1,answer_list,topk):
    if GT1[0]==None or type(GT1[0]) == bool:
        return 0
    GT=process_ground_truth(GT1)
    print ("\nLowered ground truth --> ", GT)
    
    rank=0
    answer_list1=[]
    for j in range(0,len(answer_list)):
        if j>0:
            if answer_list[j][2] < answer_list[j-1][2]:
                rank+=1
            #answer_list[j]=tuple(list(answer_list[j]).append(rank))
            answer_list1.append((answer_list[j][0],answer_list[j][1],answer_list[j][2],rank))
    answer_list=answer_list1

    print(answer_list)
    
    mrr=0.0
    flag=0
    if topk==5:
        topk=len(answer_list)
    for i in range(0,len(answer_list)):
        if answer_list[i][3]<=topk-1: #rank 1
            ans1=answer_list[i][0].split('|')
            for s in ans1:
                s=s.split(':')[0]
                s=s.strip()
                s=s.strip(',')
                #s=s.encode('utf-8')
                for GTT in GT:
                    if GTT==s: #Order preserving
                        mrr+=1.0/float(answer_list[i][3]+1)
                        flag=1
                        break
                if flag==1:
                    break
            if flag==1:
                break
        if flag==1:
            break
    return mrr


           
def replace_symbols(s):
        s=s.replace('(',' ')
        s=s.replace(')',' ')
        s=s.replace('[',' ')
        s=s.replace(']',' ')
        s=s.replace('{',' ')
        s=s.replace('}',' ')
        s=s.replace('|',' ')
        s=s.replace('"',' ')
        s=s.strip()
        return s

def process_ground_truth(GT1):
        GT=set()
        '''
        for GT11 in GT1:
                GT2=set(GT11.split('|'))
                for s in GT2:
                        s=s.strip()
                        s=s.strip(',')
                        s=replace_symbols(s)
                        GT.add(s.lower())
        '''
        for s in GT1:
                s=s.strip()
                s=s.strip(',')
                s=replace_symbols(s)
                GT.add(s.lower())
        return GT

if __name__ == "__main__":
    main(sys.argv)
