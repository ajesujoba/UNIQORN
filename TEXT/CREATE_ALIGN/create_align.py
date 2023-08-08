#this create the alignment table for ETP
import os
import time
import networkx as nx
import sys
import json
import pickle
# import kgextraction as kex
from statistics import mean 
import gensim
import generate_graphs_from_triples3 as ggtp
import multiprocessing
from multiprocessing import Pool
from multiprocessing import Queue
def writer(pqueue):
    #for a in iter(pqueue.get, None):
    while not queue.empty():
        with open(logdir,'a') as f:
            a = pqueue.get()
            #print(a)
            f.write(str(a))
            f.write("\n")



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
    global questionids
    global thequestions, logdir, timegst
    global writecontext,  writespo, writeqt, writecorner, writecooccur, writegraph, writeanswer,writehearst, writealign
    global gdict
    global queue
    queue = Queue()

    #global modelo #BERT model

    #global qno_to_id
    data = sys.argv[1]
    logfile = sys.argv[2]
    start = int(sys.argv[3])
    end  = int(sys.argv[4])
    
    
    #main method
    filejson = "./../Files/"+data
    
    #get config
    config = {}
    #read config file 
    stream=open("./configtable.yml", 'r')
    for line in stream:
        if not line.startswith('#') and len(line)>1:
            #print 'line ',line
            line=line.split(':')
            config[line[0]]=line[1].strip()
    print ("Configurations -->", config)
    
    writespo, writeqt, writecorner, writecooccur, writehearst, writealign = getstoreddir()  
    #create directory #
    #writedir, writegraph, writeanswer = createdirs(taskname)
    #print(writedir, writegraph, writeanswer)
    logdir1 = writealign+'/log/'#+logfile 
    if not os.path.exists(logdir1):
        os.mkdir(logdir1)
    logdir = writealign+'/log/'+logfile
    llf=open(logdir,"w")
    llf.close()
        
    questionids, thequestions =  getQuestion(filejson,start,end)
    print("Got ", len(questionids), " questions")
    ct = 0
    
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
    
    time1=time.time()
    p = Pool(cores)
    res_list = p.map(call_process, args)
    #p = Pool.Process(call_process,args)
    #p.start()
    p.close()
    p.join()
    time2=time.time()

    print("++++++++++++++++++++++++>>>>>>>>>>>>>>>>>>>>",res_list)
    #print(mrr_result(1, res_list,thequestions))
    
    #call_process([[0,1,len(questionids),model]])


    #call_process(writecontext,  writespo, writeqt, writecorner, writecooccur, writegraph)



def createdirs(taskname):
    #make directory
    writedir = config['write_dir']+taskname
    if not os.path.exists(writedir):
        os.mkdir(writedir)


    writeanswer = writedir+'/_ANSWER_'+taskname
    if not os.path.exists(writeanswer):
        os.mkdir(writeanswer)

    writegraph = writedir+'/_XG_'+taskname
    if not os.path.exists(writegraph):
        os.mkdir(writegraph)


    print("Created all directories")

    return writedir, writegraph, writeanswer

def getstoreddir():
    #get the textkg durectory
    
    writespo = config['write_textkg']+'_SPO2_'+config['benchmark']
    writeqt = config['write_textkg']+'_QT_'+config['benchmark']
    writecorner = config['write_textkg']+'_CORNER_'+config['benchmark']
    writecooccur = config['write_textkg']+'_COOCCUR2_'+config['benchmark']
    writehearst = config['write_textkg']+'_hearst_'+config['benchmark']
    writealign = config['write_textkg']+'_align_'+config['benchmark']
    
    return writespo, writeqt, writecorner, writecooccur, writehearst, writealign



#Type_Alignment_flag = int(config['Type_Alignment'])
#Predicate_Alignment_flag = int(config['Predicate_Alignment'])
#Entity_Alignment_flag = int(config['Entity_Alignment'])


def getQuestion(qir,start,end):
    question = {}
    qid =  []
    done = []
    #done  =[line.strip() for line in open('/GW/qa2/work/uniqorn/Crawl_Google_docs/LCQuAD_2/newHeterogen/BERT_TEXT_cased/model/TEXT/files/_align_LCQUAD_TEXT/log/'+prev,'r')]
    for line in open(qir,'r'):
        data = json.loads(line)
        if data['answers'] == None or type(data['answers']) == bool:
            #with open("/GW/qa2/work/uniqorn/Crawl_Google_docs/LCQuAD_2/LcQUAD2.0_data/split/num3.txt",'a') as f:
            #f.write(data['id'])
            #f.write("\n")
            continue
        #if data['id'] in done:
        #continue
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
    qid2 = qid[start:end]
    qid = [item for item in qid2 if item not in done]
        
    return qid, question


def call_process(args):
    res_list=[]
    Type_Alignment_flag = int(config['Type_Alignment'])
    Predicate_Alignment_flag = int(config['Predicate_Alignment'])
    Entity_Alignment_flag = int(config['Entity_Alignment'])
    verbose = int(config['verbose'])
    proc_id=args[0][0]
    ques_id_lower=args[0][1]
    ques_id_upper=args[0][2]
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
        try:
            #data_file = get_text_file(dirtloc)
        
            questn_token = []
            #wholehearst = []
            justnp = []
            doc = 0
            doc_title = qid+".txt"
            t1 = time.time()
            f22=open(writespo+'/LcQUAD_'+qid+'.txt','r')
            f33=open(writecooccur+'/LcQUAD_cooccur_'+qid+'.txt','r')

            for line in f22:
                triple=line.strip().split(' | ')
                doc_id=triple[0]
                doc_title=triple[1]
                sent_id=triple[2]
                n1=triple[3]
                d1=float(triple[4])
                n2=triple[5]
                d2=float(triple[6])
                n3=triple[7]
                
                if (n1,n2,n3) not in unique_SPO_dict:
                    unique_SPO_dict[(n1,n2,n3)]={}
                    unique_SPO_dict[(n1,n2,n3)]['d1']=[]
                    unique_SPO_dict[(n1,n2,n3)]['d2']=[]
                    unique_SPO_dict[(n1,n2,n3)]['doc_id']=[]
                    unique_SPO_dict[(n1,n2,n3)]['doc_title']=[]
                    unique_SPO_dict[(n1,n2,n3)]['sent_id']=[]
                unique_SPO_dict[(n1,n2,n3)]['d1'].append(d1)
                unique_SPO_dict[(n1,n2,n3)]['d2'].append(d2)
                unique_SPO_dict[(n1,n2,n3)]['doc_id'].append(doc_id)
                unique_SPO_dict[(n1,n2,n3)]['doc_title'].append(doc_title)
                unique_SPO_dict[(n1,n2,n3)]['sent_id'].append(sent_id)
                


            for line in f33:
                triple=line.strip().split(' | ')
                doc_id=triple[0]
                doc_title=triple[1]
                sent_id=triple[2]
                n1=triple[3]
                d1=float(triple[4])
                n2=triple[5]
                d2=float(triple[6])
                n3=triple[7]
                
                if (n1,n2,n3) not in unique_SPO_dict:
                    unique_SPO_dict[(n1,n2,n3)]={}
                    unique_SPO_dict[(n1,n2,n3)]['d1']=[]
                    unique_SPO_dict[(n1,n2,n3)]['d2']=[]
                    unique_SPO_dict[(n1,n2,n3)]['doc_id']=[] 
                    unique_SPO_dict[(n1,n2,n3)]['doc_title']=[]
                    unique_SPO_dict[(n1,n2,n3)]['sent_id']=[]
                unique_SPO_dict[(n1,n2,n3)]['d1'].append(d1)
                unique_SPO_dict[(n1,n2,n3)]['d2'].append(d2)
                unique_SPO_dict[(n1,n2,n3)]['doc_id'].append(doc_id)
                unique_SPO_dict[(n1,n2,n3)]['doc_title'].append(doc_title)
                unique_SPO_dict[(n1,n2,n3)]['sent_id'].append(sent_id)
                

            t11 = time.time()
            qtfile = writeqt+'/LcQUAD_'+qid
            #questn_tokens = pickle.load(open(qtfile,'rb'))

            cornerstone_file = writecorner+'/LcQUAD_'+qid
            allentpred = []
            allentpred.extend(list(unique_SPO_dict.keys()))
            allwords = list(set([j for i in allentpred for j in i])) #allwords includes the predicates and entities to be used to find the connerstone
            #print("question tokens == === === = ", questn_tokens)
            #print('all connerstone words ==== > ',allwords)
            
            connerstone = [] #words for words in allwords if set(kex.tokenize_text(kex.parsetextspacy(words))).intersection(set(questn_tokens))  ]
            #connerstone = pickle.load(open(cornerstone_file,'rb'))
            #write the cornerstone
            #pickle.dump(connerstone,open(cornerstone_file,'wb'))

            #print("connerstone =======>>> ", connerstone)
            hearst_file  = writehearst+'/LcQUAD2_'+qid+'.json'
            final_hearst = json.load(open(hearst_file,'r'))['hearst']
            G,_,_ = ggtp.build_graph_from_triple_edges2(unique_SPO_dict,connerstone)
            #print("hearst =======>>>> ",final_hearst)
            print("Type  in = ", type(G))
            #hearst_file  = writehearst+'/LcQUAD_'+qid+'.json'
            #final_hearst = json.load(open(hearst_file,'r'))['hearst']
            G = ggtp.add_type_edges(G,final_hearst)# ,topsnips)
            # G = ggtp.add_type_edges(G,final_hearst)

            
            if Type_Alignment_flag == 1 or Predicate_Alignment_flag == 1:
                #Read Glove embeddings
                g_pred,g_ent,g_type,g_ques=ggtp.read_glove(G,[],gdict,option=config['embedding'])

             
            if Predicate_Alignment_flag == 1:
                #Add relation alignment edges from glove embeddings
                print ("\n\nAdding predicate alignment edges\n\n")
                G,xp=ggtp.add_predicate_alignment_edges(G,g_pred,gdict)
                with open(writealign+'/LcQUAD_'+qid+'predicate.pickle', 'wb') as handle:
                    pickle.dump(xp, handle)
        
            if Type_Alignment_flag==1:
                print ("\n\nAdding type alignment edges\n\n")
                G,xt=ggtp.add_type_alignment_edges(G,g_type,gdict)
                with open(writealign+'/LcQUAD_'+qid+'type.pickle', 'wb') as handle:
                    pickle.dump(xt, handle)


            if Entity_Alignment_flag==1:
                print ("\n\nAdding entity alignment edges\n\n")
                G,xe=ggtp.add_entity_alignment_edges(G,"",questn_token)
                with open(writealign+'/LcQUAD_'+qid+'entity.pickle', 'wb') as handle:
                    pickle.dump(xe, handle)
            
                if verbose:
                    print("\n\nSize of the graph directed",len(G.nodes()),len(G.edges()))
            
            res_list.append(qid)

            
            doc += 1
            queue.put(qid)
            writer(queue)
        except IOError as err:# as err: # IOError as err:
            print("OS error: {0}".format(err))
            print("There was an error ")
            queue.put(qid)
            writer(queue)
            continue
    return res_list


if __name__ == "__main__":
    main(sys.argv)
