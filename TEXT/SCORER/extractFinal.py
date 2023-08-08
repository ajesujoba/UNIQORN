import os
import time
import networkx as nx
import sys
import json
import pickle
import kgextraction as kex
from statistics import mean 
import gensim
#import GST.generate_graphs_from_triples2 as ggtp
#from GST.get_GST_from_QKG_with_Frozenset_RANK_DOCS import call_main_GST
import multiprocessing 
from multiprocessing import Pool
from multiprocessing import Queue
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#import torch.multiprocessing as Pool
#from GST.generate_graphs_from_triples2 import build_graph_from_triple_edges2
#from GST.generate_graphs_from_triples2 import add_type_edges
#from GST.generate_graphs_from_triples2 import visualize_graph

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

#num2.txt was used for training data
def writer(pqueue):
    #for a in iter(pqueue.get, None):
    while not queue.empty():
        with open(logdir,'a') as f:
            a = pqueue.get()
            #print(a)
            f.write(str(a))
            f.write("\n")

def main(argv):
    global queue
    queue = Queue()
    global config
    global dirt,verbose
    global questionids
    global thequestions,logdir
    global writecontext,  writespo, writeqt, writecorner, writecooccur, writegraph, writeanswer,writehearst, writealign
    global gdict, writeDXG, writecontextscore, logf
    global model
    #global modelo #BERT model

    #global qno_to_id
    data = sys.argv[1]
    logf = sys.argv[2]
    #main method
    
    filejson = "./../Files/"+data
    dirt = "./../Files/"
    logdir = "./../Files/"+logf

    #get configuration file    
    config = {}
    #read config file 
    stream=open("./configFinal.yml", 'r')
    for line in stream:
        if not line.startswith('#') and len(line)>1:
            #print 'line ',line
            line=line.split(':')
            config[line[0]]=line[1].strip()
    print ("Configurations -->", config)

    #create directory #
    writecontext,  writecontextscore, writespo, writeqt, writecorner, writecooccur, writegraph, writeanswer, writehearst, writealign,writeDXG = createdirs()
    #qids =  getQid(filejson)
    #print("Got ", len(qids), " questions")
    questionids, thequestions =  getQuestion(filejson)
    print("Got ", len(questionids), " questions")
    ct = 0
    model = kex.getBERTmodel()
    model.to(device)
    print("BERT Loaded")
    #model.share_memory()
    #split the questions to cores
    call_process(1, 7)

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
    
    jobs = []
    ques_id_lower=0
    ques_id_upper=2
    proc_id=0
    args=[]
    cores=1
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
    multiprocessing.set_start_method('spawn',force=True)
    p = Pool(cores)
    res_list = p.map(call_process, args)
    #p = Pool.Process(call_process,args)
    #p.start()
    p.close()
    p.join()
    time2=time.time()
    
    #call_process([[0,1,len(questionids),model]])


    #call_process(writecontext,  writespo, writeqt, writecorner, writecooccur, writegraph)
    '''



def createdirs():
    #make directory
    writecontext = config['write_context']+'_CONTEXT_'+config['benchmark']
    if not os.path.exists(writecontext):
        os.mkdir(writecontext)
    writecontextscore = config['write_context']+'_CONTEXTSCORE_'+config['benchmark']
    if not os.path.exists(writecontextscore):
        os.mkdir(writecontextscore)


    writespo = config['write_spo']+'_SPO2_'+config['benchmark']
    if not os.path.exists(writespo):
        os.mkdir(writespo)
    writeqt = config['write_questionterms']+'_QT_'+config['benchmark']
    if not os.path.exists(writeqt):
        os.mkdir(writeqt)
    writecorner = config['write_connerstone']+'_CORNER_'+config['benchmark']
    if not os.path.exists(writecorner):
        os.mkdir(writecorner)
    writecooccur = config['write_cooccur']+'_COOCCUR2_'+config['benchmark']
    if not os.path.exists(writecooccur):
        os.mkdir(writecooccur)
    #writegraph = '/GW/qa2/work/uniqorn/Crawl_Google_docs/LCQuAD_2/GST/results/allsponoalignment20/_XG_allsponoalignment20'
    writegraph = config['write_graph']+'_XG_'+config['benchmark']
    if not os.path.exists(writegraph):
        os.mkdir(writegraph)
    writehearst = config['write_hearst']+'_hearst_'+config['benchmark']
    if not os.path.exists(writehearst):
        os.mkdir(writehearst)
    writealign = config['write_align']+'_align_'+config['benchmark']
    if not os.path.exists(writealign):
        os.mkdir(writealign)


    writeanswer = config['write_answer']+'_ANSWER_'+config['benchmark']
    if not os.path.exists(writeanswer):
        os.mkdir(writeanswer)

    writeDXG = config['write_spo']+'_DXG_'+config['benchmark']
    if not os.path.exists(writeDXG):
        os.mkdir(writeDXG)


    print("Created all directories")

    return writecontext, writecontextscore, writespo, writeqt, writecorner, writecooccur, writegraph, writeanswer, writehearst, writealign, writeDXG



#Type_Alignment_flag = int(config['Type_Alignment'])
#Predicate_Alignment_flag = int(config['Predicate_Alignment'])
#Entity_Alignment_flag = int(config['Entity_Alignment'])


def getQuestion(qir):
    question = {}
    qid =  []
    for line in open(qir,'r'):
        data = json.loads(line)
        if data['answers'] == None or type(data['answers']) == bool:
            with open("testboolnull.txt",'a+') as f:
                f.write(data['id'])
                f.write("\n")

            continue

        question[data['id']]={}
        qid.append(data['id'])

        question[data['id']]['text']=data['question']
        #print(data['answers'])
        if data['answers'] == None:
            continue
            question[data['id']]['GT']=set([None])
            continue

        if type(data['answers']) == bool:
            continue
            question[data['id']]['GT']=set([data['answers']])
            continue

        if type(data['answers']) == list and type(data['answers'][0]) ==  list:
            question[data['id']]['GT']=set([y for x in data['answers'] for y in x])
            continue

        question[data['id']]['GT']=set(data['answers'])
        
    return qid, question


def call_process(ques_id_lower, ques_id_upper):
    Type_Alignment_flag = int(config['Type_Alignment'])
    Predicate_Alignment_flag = int(config['Predicate_Alignment'])
    Entity_Alignment_flag = int(config['Entity_Alignment'])
    #verbose = int(config['verbose'])
    #proc_id=args[0][0]
    #ques_id_lower=args[0][1]
    #ques_id_upper=args[0][2]
    #model = args[0][3]
    for q in range(ques_id_lower, ques_id_upper):
        qid = questionids[q-1] 
        questn = thequestions[qid]['text']
        gt = thequestions[qid]['GT']
        #for q in qids[430:]:
        dirtloc = dirt+qid+".json"
        #dirtfin = finaldir+qid+".txt"
        print(qid)
        finalhearstdict = []
        wholehearst = []
        wholedocsd = []
        hearstdicts = dict()
        unique_SPO_dict = dict()
        setsnip = set()
        try:
            data_file = get_text_file(dirtloc)
        
            questn_token = []
            #wholehearst = []
            justnp = []
            doc = 0
            doc_title = qid+".txt"
            f22=open(writespo+'/LcQUAD_'+qid+'.txt','w')
            f33=open(writecooccur+'/LcQUAD_cooccur_'+qid+'.txt','w')
            f44 = open(writecontextscore+'/LcQUAD_snscore_'+qid+'.txt','w')
            for line in data_file:
                #factsdictslm = dict()
                textss = line['text']
                #if the text has no content
                if textss == "" or textss == None: continue
                
                print(" texts lens = ",len(textss))
                tetlen = len(textss)
                if tetlen < 9000000:
                    #process like that 
                    final = [textss]
                elif tetlen>=9000000 and tetlen<=20000000:
                    n = tetlen//2
                    final = [textss[i * n:(i + 1) * n].strip() for i in range((len(textss) + n - 1) // n )]
                elif tetlen>20000000 and tetlen<=100000000:
                    n = tetlen//10
                    final = [textss[i * n:(i + 1) * n].strip() for i in range((len(textss) + n - 1) // n )]
                elif tetlen>100000000 :
                    n = tetlen//20
                    final = [textss[i * n:(i + 1) * n].strip() for i in range((len(textss) + n - 1) // n )]
                snippet = 0    
                #whole triple if file in divided
                wholetriple = []
                #wholehearst = []
                wholecooccur = []
                wholescore = []
                factsdicts = dict()
                cooccurdicts = dict()
                #hearstdicts = dict()
                t1=time.time()
                for texts in final:                                  
                    if texts=="" or texts==None: continue
                    #score = kex.getsimilarity(questn,,model)
                    # do truecasing of the input text
                    strs = kex.truecase(kex.parsetextspacy(texts))
                    #coreference resolution, here we use noisy coref
                    strs = (kex.pronoun_coref(kex.parsetextspacy(strs)))
                    #strs = (kex.corefer(kex.parsetextspacy(strs)))
                    
                    #
                    text_doc = kex.parsetextspacy(strs.strip())
                    questn_doc = kex.parsetextspacy(questn.strip())
                    #get tokenized version of the text - paragraphs
                    text_tokens = kex.tokenize_text(text_doc)
                    #get tokenized question without 
                    questn_tokens = kex.process_questions(questn_doc)
                    #print(text_tokens)
                    #print(questn_tokens)
                    #extract all entitiy span and predicate span from the text
                    entity_indexes_span, predicates_indexes_span,np = kex.extractallEPs(text_doc)
                    justnp.extend([kex.getspantext(indexpair,text_doc).lower() for indexpair in np])
                     
                    #print("predicates index span =============>>>", predicates_indexes_span)
                    spanvalue = 50
                     
                    #get the question tokens in the passage
                    question_index = list(filter(None,kex.questionWordIndex(text_tokens, questn_tokens))) #get the index of the question word in the 
                    flattened_list = [y for x in question_index for y in x]
                    #print(flattened_list)
                    indexes = [ (max(0,x-spanvalue-1),x+spanvalue+1) for x in flattened_list]
                    #print(indexes)
                    indexes = sorted(indexes, key=lambda tup: tup[0])
                    if len(indexes) <= 0 : continue
                    indexes = list(kex.merge(indexes))
                    #print(indexes)
                    #from all the indexes extract triples
                    #hearst  =  [kex.getHearstfromtext(kex.getspantext(indexpair,text_doc)) for indexpair in indexes]
                    #triples = [kex.getthetriples(indexpair, text_doc, entity_indexes_span, predicates_indexes_span) for indexpair in indexes]
                    triples = [(i+snippet, kex.getthetriples(indexpair, text_doc, entity_indexes_span, predicates_indexes_span)) for i,indexpair in enumerate(indexes)]
                    #print('length of triples of list = ',len(triples) )
                    #triples = [j for i in triples for j in i]
                    #print(triples)
                    #get hearst for each spa
                    ##########################hearst = [kex.getHearstfromtext(texts)]
                    #final_hearst = [y for x in hearst for y in x]
                    #hearst  =  [kex.getHearstfromtext(kex.getspantext(indexpair,text_doc)) for indexpair in indexes]
                    #get score for each span
                    score = [kex.getsimilarity(questn,kex.getspantext(indexpair,text_doc),model) for indexpair in indexes]
                    #get co-occurence triples from each span
                    #cooccur = kex.co_occur(text_doc,entity_indexes_span, predicates_indexes_span)
                    cooccur = [(j+snippet, kex.co_occur(text_doc,kex.getEntityWithinSpan(indexpair, entity_indexes_span), kex.getPredicatesWithinSpan(indexpair,predicates_indexes_span))) for j,indexpair in enumerate(indexes)]
                    #hearster = [[hst[0], hst[1], str(score[i])]  for i,hearstos in enumerate(hearst) for hst in hearstos]
                    docsd = [{'docid':doc,'snippetid':i+snippet,'snippet':kex.getspantext(indexpair,text_doc),'score':str(score[i])} for i,indexpair in enumerate(indexes)]

                    
                    snippet += len(indexes)
                    wholetriple.extend(triples)
                    wholescore.extend(score)
                    wholecooccur.extend(cooccur)
                    #wholehearst.extend(hearster)
                    wholedocsd.extend(docsd)
                #print("The length of fact = ", len(wholetriple), " length of hearst = ", len(wholehearst), " length of score = ", len(wholescore))
                #print(wholetriple)
                
                #print(wholetriple)
                for i in range(len(wholetriple)):
                    #print(wholetriple[i])
                    if wholetriple[i][1] == []:
                        continue
                    #print(wholetriple[i])
                    factsdicts = kex.createdict3(wholetriple[i],wholescore[i],factsdicts)

                #for i in range(len(wholetriple)):
                #factsdicts = kex.createdict2(wholetriple[i],wholescore[i],factsdicts)
                #get the mean of score for triples
                #print("facts dicts = ", factsdicts)
                finalfactdict = list(set((k, max(v)) for k, v in factsdicts.items()))

                #print('final tuple of nodes and scores ===== > ', finalfactdict)

                for i in range(len(wholecooccur)):
                    #print(wholecooccur[i])
                    if wholecooccur[i][1] == []:
                        continue
                    cooccurdicts = kex.createdict3(wholecooccur[i],wholescore[i],cooccurdicts)
                finalcooccurdict = list(set((k, max(v)) for k, v in cooccurdicts.items()))

                #print('final tuple of nodes and scores ===== > ', finalfactdict)


                #final_hearst = [y for x in wholehearst for y in x]

                #for i in range(len(wholehearst)):
                #    hearstdicts = kex.createdict2(wholehearst[i],wholescore[i],hearstdicts)
                #finalhearstdict = list(set((k, mean(v)) for k, v in hearstdicts.items()))
                #print(hearstdicts)



                for n in finalfactdict:
                    #print(n)
                    
                    if (n[0][1][0].strip()=='' or n[0][1][1].strip()=='' or n[0][1][2].strip()=="" ):
                        continue
                    '''if n[0][1] not in unique_SPO_dict:
                        unique_SPO_dict[n[0][1]]={}
                        unique_SPO_dict[n[0][1]]['d1']=[]
                        unique_SPO_dict[n[0][1]]['d2']=[]
                        unique_SPO_dict[n[0][1]]['doc_id']=[]
                        unique_SPO_dict[n[0][1]]['doc_title']=[]
                        unique_SPO_dict[n[0][1]]['sent_id']=[]
                    unique_SPO_dict[n[0][1]]['d1'].append(n[1])
                    unique_SPO_dict[n[0][1]]['d2'].append(n[1])
                    unique_SPO_dict[n[0]]['doc_id'].append('doc_'+str(doc))
                    unique_SPO_dict[n[0]]['doc_title'].append(doc_title)
                    unique_SPO_dict[n[0]]['sent_id'].append('sent_'+str(doc))'''
                    setsnip.add('doc_'+str(doc) + '|' +'snipet_'+str(n[0][0])+ '|' + str(n[1]))
                    s1='doc_'+str(doc) +' | '+doc_title+' | '+'snipet_'+str(n[0][0])+' | '+n[0][1][0]+' | '+str(n[1])+' | '+n[0][1][1]+' | '+str(n[1])+' | '+n[0][1][2]
                    f22.write(s1)
                    f22.write('\n')

                for n in finalcooccurdict:
                    #print("********************************************************************************************************************************")
                    #print(n)
                    if (n[0][1][0].strip()=='' or n[0][1][1].strip()=='' or n[0][1][2].strip()=="" ):
                        continue
                    '''
                    if n[0] not in unique_SPO_dict and (n[0][0].strip()!="" and n[0][1].strip()!="" and n[0][2].strip()!="" ):
                        unique_SPO_dict[n[0]]={}
                        unique_SPO_dict[n[0]]['d1']=[]
                        unique_SPO_dict[n[0]]['d2']=[]
                        unique_SPO_dict[n[0]]['doc_id']=[]
                        unique_SPO_dict[n[0]]['doc_title']=[]
                        unique_SPO_dict[n[0]]['sent_id']=[]
                    unique_SPO_dict[n[0]]['d1'].append(n[1])
                    unique_SPO_dict[n[0]]['d2'].append(n[1])
                    unique_SPO_dict[n[0]]['doc_id'].append('doc_'+str(doc))
                    unique_SPO_dict[n[0]]['doc_title'].append(doc_title)
                    unique_SPO_dict[n[0]]['sent_id'].append('sent_'+str(doc))
                    '''
                    setsnip.add('doc_'+str(doc) + '|' +'snipet_'+str(n[0][0])+ '|' + str(n[1]))
                    s1='doc_'+str(doc) +' | '+doc_title+' | '+'snipet_'+str(n[0][0])+' | '+n[0][1][0]+' | '+str(n[1])+' | '+n[0][1][1]+' | '+str(n[1])+' | '+n[0][1][2]
                    f33.write(s1)
                    f33.write('\n')

                #write the set to file
                for xsx in list(setsnip):
                    f44.write(xsx)
                    f44.write('\n')


                doc += 1
            for line in wholedocsd:
                #print(line)
                docid = line['docid']
                sid = line['snippetid']
                text = line['snippet']
                score = float(line['score'])
                hearst  =  kex.getHearstfromtext(text)
                wholehearst.extend([[item[0],item[1],'doc_'+str(docid),'snipet_'+str(sid),score] for item in hearst if item != []])

            #print(wholehearst)
            datahe = {"hearst": wholehearst}
            with open(writehearst+'/LcQUAD2_'+qid+'.json', 'w') as jsonfile:
                json.dump(datahe, jsonfile)
            #print('Docs 0 eueb = ',wholedocsd)
            with open(writehearst+'/LcQUAD2_snipet'+qid+'.json', 'w') as files:
                #json.dump(str(wholedocsd), files)
                for item in wholedocsd:
                    files.write(json.dumps(item))
                    files.write('\n')
                #print("Unique SQO === >>> ", unique_SPO_dict)
                #print(dirtloc)
                
                #write facts

                #f22=open(writespo+'/LcQUAD_'+qid+'.txt','w')

                '''for n in finalfactdict:
                    if (n[0][0].strip()=='' or n[0][1].strip()=='' or n[0][2].strip()=="" ):
                        continue
                    #print(n)
                    s1='doc_'+str(doc) +' | '+doc_title+' | '+'sent_'+str(doc)+' | '+n[0][0]+' | '+str(n[1])+' | '+n[0][1]+' | '+str(n[1])+' | '+n[0][2]
                    f22.write(s1)
                    f22.write('\n')
                for n in finalcooccurdict:
                    if (n[0][0].strip()=='' or n[0][1].strip()=='' or n[0][2].strip()=="" ):
                        continue
                    #print(n)
                    s1='doc_'+str(doc) +' | '+doc_title+' | '+'sent_'+str(doc)+' | '+n[0][0]+' | '+str(n[1])+' | '+n[0][1]+' | '+str(n[1])+' | '+n[0][2]
                    f33.write(s1)
                    f33.write('\n')
                '''
            f22.close()
            f33.close()
            f44.close()
            queue.put(qid)
            writer(queue)
            #return 
            
        except IOError as err: # IOError as err:
            print("OS error: {0}".format(err))
            queue.put(qid)
            ##chronicler(a)
            writer(queue)
            continue


if __name__ == "__main__":
    main(sys.argv)
