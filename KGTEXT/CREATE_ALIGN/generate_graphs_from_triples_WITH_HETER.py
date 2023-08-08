import os
import networkx as nx
import sys
import matplotlib.pyplot as plt
import math
import numpy as np
#import hearstPatterns
import nltk
import pickle
import requests
import json
from collections import defaultdict
#from hearstPatterns.hearstPatterns import HearstPatterns
#from hearstPatterns import HearstPatterns
from nltk.corpus import stopwords
sw = stopwords.words("english")
from statistics import mean

verbose=0
MAX_MATCH=0
threshold_align = 0.5

aux_list=set(['be','am','being','been','is','are','was','were','has','have','had','having','do','does','did','done','will','would','shall','should','can','could','dare','may','might', 'must','need','ought'])

stop_list=set(['a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'aren\'t', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can\'t', 'cannot', 'could', 'couldn\'t', 'did', 'didn\'t', 'do', 'does', 'doesn\'t', 'doing', 'don\'t', 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn\'t', 'has', 'hasn\'t', 'have', 'haven\'t', 'having', 'he', 'he\'d', 'he\'ll', 'he\'s', 'her', 'here', 'here\'s', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'how\'s', 'i', 'i\'d', 'i\'ll', 'i\'m', 'i\'ve', 'if', 'in', 'into', 'is', 'isn\'t', 'it', 'it\'s', 'its', 'itself', 'let\'s', 'me', 'more', 'most', 'mustn\'t', 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours        ourselves', 'out', 'over', 'own', 'same', 'shan\'t', 'she', 'she\'d', 'she\'ll', 'she\'s', 'should', 'shouldn\'t', 'so', 'some', 'such', 'than', 'that', 'that\'s', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'there\'s', 'these', 'they', 'they\'d', 'they\'ll', 'they\'re', 'they\'ve', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', 'wasn\'t', 'we', 'we\'d', 'we\'ll', 'we\'re', 'we\'ve', 'were', 'weren\'t', 'what', 'what\'s', 'when', 'when\'s', 'where', 'where\'s', 'which', 'while', 'who', 'who\'s', 'whom', 'why', 'why\'s', 'with', 'won\'t', 'would', 'wouldn\'t', 'you', 'you\'d', 'you\'ll', 'you\'re', 'you\'ve', 'your', 'yours', 'yourself', 'yourselves'])

def splittext(line):
    spl = line.split('|')
    spl[2] = float(spl[2])
    fspl = ((spl[0],spl[1]),spl[2])
    return fspl


def rank_and_select_triples(lsto,topcontext=0):
    if topcontext == 0:
        return [x[0] for x  in  lsto]
    else:
        if topcontext > len(lsto):
            return lsto
        contextlist1 = sorted(lsto,key=lambda x:x[1],reverse=True)
        rankcontextlist,lenx = rank_list_answers(contextlist1)
        groupedcontext = group_rank(rankcontextlist, lenx)
        thetopcontexts = gettopk(groupedcontext, rank=topcontext)
        thetopcontextsx = [x[0] for x  in  thetopcontexts]
        return thetopcontextsx

def select_triples2(lsto,topcontext=0):
    if topcontext == 0:
        return lsto 
    else:
        if topcontext > len(lsto):
            return lsto
        lsto2 = [(x,float(x.split('####')[0].strip())) for x  in  lsto]
        contextlist1 = sorted(lsto2,key=lambda x:x[1],reverse=True)
        rankcontextlist,lenx = rank_list_answers(contextlist1)
        groupedcontext = group_rank(rankcontextlist, lenx)
        thetopcontexts = gettopk(groupedcontext, rank=topcontext)
        thetopcontextsx = [x[0] for x  in  thetopcontexts]
        return thetopcontextsx
    return lsto

def select_triples(lsto,topl):
    newtrip =  [line.split('#####')[1].strip()  for line in lsto if (line.split('#####')[0].strip(),line.split('#####')[0].strip()) in topl]
    #print('New KG triples  = ',newtrip)
    return newtrip


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


def group_rank(ranked_list,size):
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
    return newrank

def addtoGraph(G,nodeweightavg,questn_tokens,predalias_tokens,path_file,entity_names,stempred,entcornerdict,pred_count,type_count):
    #(G,nodeweightavg,questn_tokens,path_file,entity_names,stempred,entcornerdict,pred_count,type_count)

    allnodes = list(G.nodes)
    allentities = [item+':Entity' for item in entity_names]

    entityinG = list(set(allnodes).intersection(set(allentities)))

    if len(entityinG) <=1 :
        return G, nodeweightavg

    if len(entityinG) > 1:
        #get the combination, get the paths and add them to the current graph
        result = [frozenset((entityinG[p1], entityinG[p2])) for p1 in range(len(entityinG)) for p2 in range(p1+1,len(entityinG))]

        #read the path file and check the combinations that are present
        try:
            allpaths = pickle.load(open(path_file,'rb'))
        except FileNotFoundError:
            return G, nodeweightavg
        if len(allpaths) <=0:
            return G, nodeweightavg

        triples = []
        for seed in result:
            if seed in allpaths:
                #get the context and weight
                facts = allpaths[seed]['context']
                scores = allpaths[seed]['score']

                if isinstance(facts[0], str):
                    #check the number of item in there
                    #convert to unque dicts
                    for i,j in enumerate(facts):
                        triples.append((facts[i],scores[i]))
                        print('add single path!!!!!!!!')
                elif isinstance(facts[0], tuple):
                    for i,j in enumerate(facts):
                        if len(facts[i]) < 2:
                            continue
                        triples.append((facts[i][0],scores[i]))
                        triples.append((facts[i][1],scores[i]))
                        print('adding multiple paths!!!!!!')
        if len(triples) <=0:
            return G, nodeweightavg

    #convert the triple to the format needed for a graph
    unique_SPO_dict = formatTriple(triples)

    #add the new paths to the graph G
    G, nodeweightavg = build_graph_from_triple_edges_KG2(G,unique_SPO_dict,questn_tokens,predalias_tokens,entity_names,stempred,entcornerdict,pred_count,type_count)

    #return the graph
    return G, nodeweightavg
def formatTriple(triples):
    #print('Formating the triple!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    unique_SPO_dict={}
    first_flag=0
    allfacts = []
    for x,y in triples:
        xy = x.split('####')
        xp = [itp.strip() for itp in xy]
        score = y
        if len (xp) == 3:
            facts = getmainFact(xp,score)
        elif len (xp) > 3:
            facts = getmainFact(xp,score)
            facts.extend(getQualifiers(xp,score))
        allfacts.extend(facts)
    #print(allfacts)

    for line in allfacts:
        #sent[s_id][0]+' | '+sent[s_id][1]+' | '+sent[s_id][2]+' | '+s.encode('utf-8')+' | '+str(d1)+' | '+p.encode('utf-8')+' | '+str(d2)+' | '+o.encode('utf-8')
        #if verbose:
        #       print line
        #l=line.strip().split(' | ')
        triple=line.strip().split(' ### ')
        #print('tple == ', triple)
        #triple_list.append((l[0],l[1],l[2],l[3],l[4],l[5],l[6],l[7]))
        if triple[0]!='Qualifier':
            first_flag=1
            score_n1=0.0
            score_n3=0.0
            matched_n1=''
            matched_n3=''
            #n1_id=triple[0]
            #print(triple)
            n1=triple[0].strip().strip('"').replace(':',' ').strip()
            try:
                score_n1=float(triple[1].strip())
                score_n3=float(triple[3].strip())
            except ValueError:
                continue
            n2=triple[2].strip().strip('"').replace(':',' ').strip()
            n3=triple[4].strip().strip('"').replace(':',' ').strip()

            if (n1,n2,n3) not in unique_SPO_dict:
                unique_SPO_dict[(n1,n2,n3)]={}
            unique_SPO_dict[(n1,n2,n3)]['score_n1']=score_n1
            unique_SPO_dict[(n1,n2,n3)]['score_n3']=score_n3
            unique_SPO_dict[(n1,n2,n3)]['matched_n1']=matched_n1
            unique_SPO_dict[(n1,n2,n3)]['matched_n3']=matched_n3
        else:
            if first_flag == 0:
                continue
            score_qual=0.0
            try:
                score_qual = float(triple[1].strip())
            except ValueError:
                continue
            matched_qual=''
            qual1=triple[2].strip().strip('"').replace(':',' ').strip()
            #qual2_id=triple[3].decode('utf-8')
            qual2=triple[3].strip().strip('"').replace(':',' ').strip()
            try:
                if 'qualifier' not in unique_SPO_dict[(n1,n2,n3)]:
                    unique_SPO_dict[(n1,n2,n3)]['qualifier']=[]
            except KeyError:
                continue
            unique_SPO_dict[(n1,n2,n3)]['qualifier'].append((qual1,qual2))
            if 'matched_qual' not in unique_SPO_dict[(n1,n2,n3)]:
                unique_SPO_dict[(n1,n2,n3)]['matched_qual']=[]
            unique_SPO_dict[(n1,n2,n3)]['matched_qual'].append(matched_qual)
            if 'score_qual' not in unique_SPO_dict[(n1,n2,n3)]:
                unique_SPO_dict[(n1,n2,n3)]['score_qual']=[]
            unique_SPO_dict[(n1,n2,n3)]['score_qual'].append(score_qual)
    return unique_SPO_dict
def call_main_GRAPH_KG(spofile,topl,typefile, entity_dir, questn_tokens, path_file, pred_aliases_dir, config,topcontext=0,stempred=False,degeneratex=False,pred_count={}):
    try:
        f11=open(spofile,'r')
        #f22=open(typefile,'r')
        scoretriplesx = list(set([line.strip() for line in f11 if '### instance of ###' not in line and '### occupation ###' not in line]))
        with open(entity_dir, 'r') as filehandle:
            entity_names = json.load(filehandle)
        with open(pred_aliases_dir, 'r') as filehandle:
            aliases = json.load(filehandle)
    except FileNotFoundError as err:
        G = nx.DiGraph()
        corner = {}
        return G,corner, []

    #get the highest scoring triples from the KG
    scoretriples = select_triples(scoretriplesx,topl)
    #if KG is not part of the top triples return an empty directed graph and an empty cornerstone
    #print('Topl LLLL = ',topl)
    #print('The length of the triples here is = ',len(scoretriples))
    if len(scoretriples)<=0:
        G = nx.DiGraph()
        corner = {}
        return G,corner,scoretriples
    print('The length of the triples = ',len(scoretriples))



    #with open(entity_dir, 'r') as filehandle:
    #entity_names = json.load(filehandle)
    #print("enity names = ", entity_names)

    #with open(pred_aliases_dir, 'r') as filehandle:
    #aliases = json.load(filehandle)

    predalias_tokens = {al.lower():mergealiases(listtext) for al,listtext in aliases.items()}

    #print("predicate aliases = ",predalias_tokens)
    Add_Type_flag = int(config['Add_Type'])



    #pick top k triples here

    ##get all the facts into a list
    allfacts = []
    for triples in scoretriples:
        tpsplit = triples.split('####')
        score = tpsplit[0]
        xp = tpsplit[1].split('###')
        if len (xp) == 3:
            facts = getmainFact(xp,score)
        elif len (xp) > 3:
            facts = getmainFact(xp,score)
            facts.extend(getQualifiers(xp,score))
        allfacts.extend(facts)
    #print('actual = ',allfacts)
    #print('actual triple length = ', len(allfacts))

    if Add_Type_flag == 1:
        f22=open(typefile,'r')
        #get all the types and append to all facts (occupatio and instance)
        for line in f22:
            allfacts.append(line.strip())
        #print('actual = ',allfacts[20:50])


    unique_SPO_dict={}
    first_flag=0

    for line in allfacts:
        #sent[s_id][0]+' | '+sent[s_id][1]+' | '+sent[s_id][2]+' | '+s.encode('utf-8')+' | '+str(d1)+' | '+p.encode('utf-8')+' | '+str(d2)+' | '+o.encode('utf-8')
        #if verbose:
        #       print line
        #l=line.strip().split(' | ')
        triple=line.strip().split(' ### ')
        #print('The lenth of tripple = ',len(triple), triple)

        #triple_list.append((l[0],l[1],l[2],l[3],l[4],l[5],l[6],l[7]))
        if triple[0]!='Qualifier':
            first_flag=1
            score_n1=0.0
            score_n3=0.0
            matched_n1=''
            matched_n3=''
            #n1_id=triple[0]
            #print(triple)
            n1=triple[0].strip().strip('"').replace(':',' ').strip()
            try: 
                score_n1=float(triple[1].strip())
                score_n3=float(triple[3].strip())
            except ValueError:
                continue
            n2=triple[2].strip().strip('"').replace(':',' ').strip()
            n3=triple[4].strip().strip('"').replace(':',' ').strip()

            if (n1,n2,n3) not in unique_SPO_dict:
                unique_SPO_dict[(n1,n2,n3)]={}
            unique_SPO_dict[(n1,n2,n3)]['score_n1']=score_n1
            unique_SPO_dict[(n1,n2,n3)]['score_n3']=score_n3
            unique_SPO_dict[(n1,n2,n3)]['matched_n1']=matched_n1
            unique_SPO_dict[(n1,n2,n3)]['matched_n3']=matched_n3
        else:
            if first_flag == 0:
                continue
            score_qual=0.0
            try:
                score_qual = float(triple[1].strip())
            except ValueError:
                continue
            matched_qual=''
            qual1=triple[2].strip().strip('"').replace(':',' ').strip()
            #qual2_id=triple[3].decode('utf-8')
            qual2=triple[3].strip().strip('"').replace(':',' ').strip()
            try:
                if 'qualifier' not in unique_SPO_dict[(n1,n2,n3)]:
                    unique_SPO_dict[(n1,n2,n3)]['qualifier']=[]
            except KeyError:
                continue
            unique_SPO_dict[(n1,n2,n3)]['qualifier'].append((qual1,qual2))
            if 'matched_qual' not in unique_SPO_dict[(n1,n2,n3)]:
                unique_SPO_dict[(n1,n2,n3)]['matched_qual']=[]
            unique_SPO_dict[(n1,n2,n3)]['matched_qual'].append(matched_qual)
            if 'score_qual' not in unique_SPO_dict[(n1,n2,n3)]:
                unique_SPO_dict[(n1,n2,n3)]['score_qual']=[]
            unique_SPO_dict[(n1,n2,n3)]['score_qual'].append(score_qual)
        #print(unique_SPO_dict)
    #Question vectors and node vectors are built inside
    #G=build_graph_from_triple_edges(unique_SPO_dict,q_ent,type_qent,gdict,mention_dict,Cornerstone_Matching)
    G, nodeweightavg,entcornerdict,pred_count,type_count = build_graph_from_triple_edges_KG(unique_SPO_dict,questn_tokens,predalias_tokens,entity_names,stempred,pred_count=pred_count)
    if config['connectseed'] == '1':
        G,nodeweightavg = addtoGraph(G,nodeweightavg,questn_tokens,predalias_tokens,path_file,entity_names,stempred,entcornerdict,pred_count,type_count)
    G = update_edge_weight(G)
    #if degeneratex == True:
    #print("Setting weight to 1 ")
    #G = degenerate(G)
    #print([G[n11][n22]['wlist'] for (n11,n22) in G.edges()])
    #G, nodeweightavg = build_graph_from_triple_edges(unique_SPO_dict)
    print("The graph has ", len(G.nodes()), ' nodes and ', len(G.edges()), ' edges ')
    #G=G2.to_undirected() #make QKG Undirected
    #G=directed_to_undirected(G)
    #print("The graph has ", len(G.nodes()), ' nodes and ', len(G.edges()), ' edges ')
    #if len(G.nodes())>0:
    #G=max(nx.connected_component_subgraphs(G), key=len)
    #print("The graph has ", len(G.nodes()), ' nodes and ', len(G.edges()), ' edges ')
    #print(list(G.edges))
    #G = add_nodes_weights(G,entdict, predcornerdict,meanval=False)
    #get the corner stone
    nodeweights = {}
    nodeweights = dict((k, v) for k, v in nodeweightavg.items() if k in G.nodes())
    #if the nodes are still present in the graph after removal and they have a score above the threshold, they are cornerstone
    #corner = dict((k, v) for k, v in nodeweightavg.items() if v >= cornerthreshold and k in G.nodes())
    G = add_nodes_weights2(G,nodeweights)
    
    corner = getcornerstone_KG(G,questn_tokens,entity_names, predalias_tokens,stempred)
    #nx.write_gpickle(G,graph_file)
    #pickle.dump(corner,open(cornerstone_file,'wb'))
    return G,corner,scoretriples

def degenerate(G):
    #print("is it direted = ", nx.is_directed(G))
    for u,v,d in G.edges(data=True):
        d['wlist'] = [0.0]
        d['weight'] = 0
    return G

#merge predicate aliases so that I can use them for collecting predicate cornerstone
def mergealiases(textlist):
    text = ' '.join(textlist)
    #print(text)
    finaltok = [w for w in text.lower().split() if not w in sw]
    return set(finaltok)

def stemword(item):
    return porter.stem(item)

def getcornerstone_KG(G,questn_tokens, entitynames,predalias_tokens,stempred):
    corner = {}
    count = 0
    ques_corner = set(questn_tokens)
    for  n in G.nodes():
        nn2 = n.split(':')
        
        if G.node[n]['source']!='KG':
            continue

        if not (nn2[1]=='Entity' or nn2[1]=='Type' or nn2[1]=='Predicate'):
            continue
        #print(' the nodes are -------> ', n)
        if nn2[1]=='Entity' or nn2[1]=='Type':
            #if the entity node is in list of Entity nodes
            if nn2[0] in entitynames:
                corner[n] = nn2[0]
        if nn2[1]=='Predicate':
            if stempred == True:
                predicates_aliases = set([stemword(items) for items in predalias_tokens[nn2[0]]])
                wordsplit = set([stemword(items) for items in nn2[0].split()])
                if bool(wordsplit.intersection(ques_corner)) or bool(ques_corner.intersection(set(predicates_aliases))):
                    inter1 = list(wordsplit.intersection(ques_corner))
                    inter2 = list(ques_corner.intersection(set(predicates_aliases)))
                    corners = ' | '.join(list(set(inter1+inter2)))
                    corner[n] = corners
            else:
                if nn2[0] not in predalias_tokens:
                    continue
                predicates_aliases = predalias_tokens[nn2[0]]
                wordsplit = set(nn2[0].split())
                if bool(wordsplit.intersection(ques_corner)) or bool(ques_corner.intersection(set(predicates_aliases))):
                    inter1 = list(wordsplit.intersection(ques_corner))
                    inter2 = list(ques_corner.intersection(set(predicates_aliases)))
                    corners = ' | '.join(list(set(inter1+inter2)))
                    corner[n] = corners
    return corner

def getcornerstone_TEXT(G,questn_tokens):
    corner = {}
    count = 0
    for  n in G.nodes():
        if G.node[n]['source']!='TEXT':
            continue
        nn2 = n.split(':')
        if not (nn2[1]=='Entity' or nn2[1]=='Predicate'):
            continue
        #print(' the nodes are -------> ', n)
        nsplit = nn2[0]
        #wordsplit = set(n.split())
        wordsplit = set(nsplit.split())
        if bool(wordsplit.intersection(set(questn_tokens))):
            #print(n, ' is a cornerstone')
            count = count + 1
            inter = list(wordsplit.intersection(set(questn_tokens)))
            corner[n] = inter[-1]
    return corner


def add_nodes_weights2(G,cornerdict):
    for n1 in G.nodes():
        if n1 in cornerdict:
            val = cornerdict[n1]
            G.node[n1]['weight']=round(val,2)
    return G

def getmainFact(xp, score=0.0):
    mainfact = xp[0:3]
    mainfact.insert(1, str(score))
    mainfact.insert(3, str(score))
    return [' ### '.join(mainfact)]

def getQualifiers(xp, score=0.0):
    #return ['Qualifier | '+' | '.join([xp[i],xp[i+1]]) for i in range(3,len(xp)-1) if i%2 != 0 ]
    return ['Qualifier ### '+str(score)+' ###'+' ### '.join([xp[i],xp[i+1]]) for i in range(3,len(xp)-1) if i%2 != 0 ]

def add_edge_triple_KG(G,n1,n2,d):
    wlist1=[d]
    if (n1,n2) in G.edges():
        for (n11,n22) in G.edges():
            if (n2.split(':')[1]=='Predicate' and n22.split(':')[1]=='Predicate' and n2.split(':')[0]==n22.split(':')[0] and n1==n11) or (n1.split(':')[1]=='Predicate' and n11.split(':')[1]=='Predicate' and n1.split(':')[0]==n11.split(':')[0] and n2==n22):
                data=G.get_edge_data(n11,n22)
                #d=data['weight']
                wlist=data['wlist']
                wlist.extend(wlist1)
                G.add_edge(n1,n2, weight=wlist, did=['kg'], dtitle=['kg'], sid=['kg'], wlist=wlist, etype='Triple', source='KG')
    else:
        G.add_edge(n1,n2, weight=wlist1, did=['kg'], dtitle=['kg'], sid=['kg'], wlist=wlist1, etype='Triple', source='KG')
    return G


def update_edge_weight(G):
    for (n11,n22) in G.edges():
        if (n22.split(':')[1]=='Predicate') or (n11.split(':')[1]=='Predicate'):
            weights = G[n11][n22]['wlist']
            avgweights = mean(weights)
            G[n11][n22]['weight'] = avgweights
            #G[n11][n22]['wlist'] = [avgweights]
    return G

def build_graph_from_triple_edges_KG2(G,unique_SPO_dict,questn_token, predalias_tokens, entity_names,stempred,entcornerdict,pred_count,type_count):
    c = 0
    print("Connecting the nodes meeehhhnnnnnn ..........................................")
    if stempred == True:
       Cornerstone = set([stemword(qts) for qts in questn_token])

    else:
        Cornerstone = set(questn_token)
    print("Question tokens = ", Cornerstone)
    #print("Unique SPOs cornerstones ",len(unique_SPO_dict))
    for (nn1,nn2,nn3),v in unique_SPO_dict.items():
        n1=nn1.lower()
        n2=nn2.lower()
        n3=nn3.lower()
        #print(nn1,'->',nn2,'->',nn3)
        if n2 in aux_list:
            continue
        d1,d2 = v['score_n1'],v['score_n1']
        #print(d1,d2)

        n1split = set(n1.split())
        n2split = set(n2.split())
        n3split = set(n3.split())
        if stempred == True:
            n2splitstem = set([stemword(qts) for qts in n2split])
        if n2=='typeo of': #Type edge
            if n1.lower()+":Entity" not in G.nodes():
                continue
            if n3 not in type_count:
                type_count[n3]=1
            else:
                #print('Adding Types')
                type_count[n3]=type_count[n3]+1
                n11=n1.lower()+":Entity"
                n33=n3.lower()+":Type:"+str(type_count[n3])
                if n11 not in G.nodes():
                    G.add_node(n11,weight=unique_SPO_dict[(nn1,nn2,nn3)]['score_n1'],did=[], dtitle=[], sid=[], matched=unique_SPO_dict[(nn1,nn2,nn3)]['matched_n1'], source='KG')
                G.add_node(n33,weight=0.0, did=[], dtitle=[], sid=[], matched='', source='KG')
                G.add_edge(n11,n33, weight=0.0, did=[], dtitle=[], sid=[],  wlist=[0.0], etype='Type', source='KG')
        elif n2=='instance of' or n2=='occupation':
            n11=n1.lower()+":Entity"
            if n11 not in G.nodes():
                #print(n11,' not in Graphs')
                #if the subject entity in the triple is not in the current grapb, then continue
                continue
            #print(n11, ' is in the Graph')
            if n2 not in pred_count:
                pred_count[n2]=1
            else:
                pred_count[n2]=pred_count[n2]+1
            n11=n1.lower()+":Entity"
            n22=n2.lower()+":Predicate:"+str(pred_count[n2])
            n33=n3.lower()+":Type"

            #get the weights for each node that is an anchor
            if n1 in entity_names:
                if n11 not in entcornerdict:
                    entcornerdict[n11] = []
                entcornerdict[n11].append(d1)
            if n3 in entity_names:
                if n33 not in entcornerdict:
                    entcornerdict[n33] = []
                entcornerdict[n33].append(d2)
            if stempred == True:
                predalisstem = set([stemword(items) for items in predalias_tokens[n2.lower()]])
                if n2splitstem.intersection(Cornerstone) or Cornerstone.intersection(predalisstem):
                    if n22 not in entcornerdict:
                        entcornerdict[n22] = []
                    entcornerdict[n22].append(d2)
            else:
                if n2split.intersection(Cornerstone) or Cornerstone.intersection(predalias_tokens[n2.lower()]):
                    if n22 not in entcornerdict:
                        entcornerdict[n22] = []
                    entcornerdict[n22].append(d2)

            if n11 not in G.nodes():
                G.add_node(n11,weight=0.0, did=['kg'], dtitle=['kg'], sid=['kg'], matched=unique_SPO_dict[(nn1,nn2,nn3)]['matched_n1'], source='KG')
            G.add_node(n22,weight=0.0, did=['kg'], dtitle=['kg'], sid=['kg'], matched='', source='KG')
            if n33 not in G.nodes():
                G.add_node(n33,weight=0.0,did=['kg'], dtitle=['kg'], sid=['kg'], matched=unique_SPO_dict[(nn1,nn2,nn3)]['matched_n3'], source='KG')
            #G.add_edge(n11,n22, weight=0.0, wlist=[0.0], etype='Triple')
            #G.add_edge(n22,n33, weight=0.0, wlist=[0.0], etype='Triple')
            add_edge_triple_KG(G,n11,n22,d1)
            add_edge_triple_KG(G,n22,n33,d1)

        else:
            if n2 not in pred_count:
                pred_count[n2]=1
            else:
                pred_count[n2]=pred_count[n2]+1
            n11=n1.lower()+":Entity"
            n22=n2.lower()+":Predicate:"+str(pred_count[n2])
            n33=n3.lower()+":Entity"
            #get the weights for each node that is an anchor
            if n1 in entity_names:
                if n11 not in entcornerdict:
                    entcornerdict[n11] = []
                entcornerdict[n11].append(d1)
            if n3 in entity_names:
                if n33 not in entcornerdict:
                    entcornerdict[n33] = []
                entcornerdict[n33].append(d2)
            if stempred == True:
                predalisstem = set([stemword(items) for items in predalias_tokens[n2.lower()]])
                if n2splitstem.intersection(Cornerstone) or Cornerstone.intersection(predalisstem):
                    if n22 not in entcornerdict:
                        entcornerdict[n22] = []
                    entcornerdict[n22].append(d2)
            else:
                if n2split.intersection(Cornerstone): # or Cornerstone.intersection(predalias_tokens[n2.lower()]):
                    if n22 not in entcornerdict:
                        entcornerdict[n22] = []
                    entcornerdict[n22].append(d2)
                elif n2.lower() in predalias_tokens:
                    if Cornerstone.intersection(predalias_tokens[n2.lower()]):
                        if n22 not in entcornerdict:
                            entcornerdict[n22] = []
                        entcornerdict[n22].append(d2)


            if n11 not in G.nodes():
                G.add_node(n11,weight=0.0,did=['kg'], dtitle=['kg'], sid=['kg'], matched=unique_SPO_dict[(nn1,nn2,nn3)]['matched_n1'], source='KG')
            G.add_node(n22,weight=0.0,did=['kg'], dtitle=['kg'], sid=['kg'], matched='', source='KG')
            if n33 not in G.nodes():
                G.add_node(n33,weight=0.0,did=['kg'], dtitle=['kg'], sid=['kg'], matched=unique_SPO_dict[(nn1,nn2,nn3)]['matched_n3'], source='KG')
            #G.add_edge(n11,n22, weight=0.0, wlist=[0.0], etype='Triple')
            #G.add_edge(n22,n33, weight=0.0, wlist=[0.0], etype='Triple')
            add_edge_triple_KG(G,n11,n22,d1)
            add_edge_triple_KG(G,n22,n33,d1)
        #ADD qualifier nodes edges
        if 'qualifier' in unique_SPO_dict[(nn1,nn2,nn3)] and n2 !='typeo of':
            #print('Qualifier oooo')
            #print((nn1,nn2,nn3), unique_SPO_dict[(nn1,nn2,nn3)])
            for qualct in range(0,len(unique_SPO_dict[(nn1,nn2,nn3)]['qualifier'])):
                qual=unique_SPO_dict[(nn1,nn2,nn3)]['qualifier'][qualct]
                qn2=qual[0]
                qn3=qual[1]
                if qn2.lower() in aux_list:
                    continue

                if qn2 not in pred_count:
                    pred_count[qn2]=1
                else:
                    pred_count[qn2]=pred_count[qn2]+1

                qn22=qn2.lower()+":Predicate:"+str(pred_count[qn2])
                qn33=qn3.lower()+":Entity"

                d1 = unique_SPO_dict[(nn1,nn2,nn3)]['score_qual'][qualct]

                n2split = set(qn2.lower().split())
                n3split = set(qn3.lower().split())
                if qn3.lower() in entity_names:
                    if qn33 not in entcornerdict:
                        entcornerdict[qn33] = []
                    entcornerdict[qn33].append(d1)
                if stempred == True:
                    n2splitstem = set([stemword(qts) for qts in n2split])
                    predalisstem = set([stemword(items) for items in predalias_tokens[qn2.lower()]])
                    if n2splitstem.intersection(Cornerstone) or Cornerstone.intersection(predalisstem):
                        if qn22 not in entcornerdict:
                            entcornerdict[qn22] = []
                        entcornerdict[qn22].append(d1)
                else:
                    if n2split.intersection(Cornerstone) or Cornerstone.intersection(predalias_tokens[qn2.lower()]):
                        if qn22 not in entcornerdict:
                            entcornerdict[qn22] = []
                        entcornerdict[qn22].append(d1)
                G.add_node(qn22,weight=0.0,did=['kg'], dtitle=['kg'], sid=['kg'], matched='', source='KG')
                if qn33 not in G.nodes():
                    G.add_node(qn33,weight=0.0,did=['kg'], dtitle=['kg'], sid=['kg'], matched=unique_SPO_dict[(nn1,nn2,nn3)]['matched_qual'][qualct], source='KG')
                #G.add_edge(n22,qn22, weight=0.0, wlist=[0.0], etype='Triple')
                #G.add_edge(qn22,qn33, weight=0.0, wlist=[0.0], etype='Triple')
                add_edge_triple_KG(G,n22,qn22,d1)
                add_edge_triple_KG(G,qn22,qn33,d1)
        c+=1

    nodeweightavg = dict((k, mean(v)) for k, v in entcornerdict.items())
    #print("node scores", nodeweightavg )
    return G,nodeweightavg


def build_graph_from_triple_edges_KG(unique_SPO_dict,questn_token, predalias_tokens, entity_names,stempred, pred_count={}):
    G = nx.DiGraph()
    c=0
    #pred_count={}
    type_count={}
    entcornerdict = {}
    predcornerdict = {}
    Cornerstone = set()
    if stempred == True:
       Cornerstone = set([stemword(qts) for qts in questn_token])

    else:
        Cornerstone = set(questn_token)
    print("Question tokens = ", Cornerstone)
    #print("Unique SPOs cornerstones ",len(unique_SPO_dict))
    for (nn1,nn2,nn3),v in unique_SPO_dict.items():
        n1=nn1.lower()
        n2=nn2.lower()
        n3=nn3.lower()
        #print(nn1,'->',nn2,'->',nn3)
        if n2 in aux_list:
            continue
        d1,d2 = v['score_n1'],v['score_n1']
        #print(d1,d2)

        n1split = set(n1.split())
        n2split = set(n2.split())
        n3split = set(n3.split())
        if stempred == True:
            n2splitstem = set([stemword(qts) for qts in n2split])
        '''
        if n1split.intersection(Cornerstone)  or n1 in entity_names:
            if n1 not in entcornerdict:
                entcornerdict[n1] = []
                #print(" Adding n1 ",n1, " that is a cornerstone ******************")
            entcornerdict[n1].append(d1)
            #print("Corner dict = ",cornerdict)
        if n3split.intersection(Cornerstone) or n3 in entity_names:
            if n3 not in entcornerdict:
                entcornerdict[n3] = []
                #print(" Adding n3 ",n3, " that is a cornerstone *******************")
            entcornerdict[n3].append(d2)
            #print("Corner dict = ",cornerdict)
        if n2split.intersection(Cornerstone):
            if n2 not in predcornerdict:
                predcornerdict[n2] = []
                #print(" Adding n2 ",n2, " that is a cornerstone *******************")
            predcornerdict[n2].append(d2)
        '''
        if n2=='typeo of': #Type edge
            if n1.lower()+":Entity" not in G.nodes():
                continue
            if n3 not in type_count:
                type_count[n3]=1
            else:
                #print('Adding Types')
                type_count[n3]=type_count[n3]+1
                n11=n1.lower()+":Entity"
                n33=n3.lower()+":Type:"+str(type_count[n3])
                if n11 not in G.nodes():
                    G.add_node(n11,weight=unique_SPO_dict[(nn1,nn2,nn3)]['score_n1'],did=[], dtitle=[], sid=[], matched=unique_SPO_dict[(nn1,nn2,nn3)]['matched_n1'], source='KG')
                G.add_node(n33,weight=0.0, did=[], dtitle=[], sid=[], matched='', source='KG')
                G.add_edge(n11,n33, weight=0.0, did=[], dtitle=[], sid=[],  wlist=[0.0], etype='Type', source='KG')
        elif n2=='instance of' or n2=='occupation':
            n11=n1.lower()+":Entity"
            if n11 not in G.nodes():
                #print(n11,' not in Graphs')
                #if the subject entity in the triple is not in the current grapb, then continue
                continue
            #print(n11, ' is in the Graph')
            if n2 not in pred_count:
                pred_count[n2]=1
            else:
                pred_count[n2]=pred_count[n2]+1
            n11=n1.lower()+":Entity"
            n22=n2.lower()+":Predicate:"+str(pred_count[n2])
            n33=n3.lower()+":Type"

            #get the weights for each node that is an anchor
            if n1 in entity_names:
                if n11 not in entcornerdict:
                    entcornerdict[n11] = []
                entcornerdict[n11].append(d1)
            if n3 in entity_names:
                if n33 not in entcornerdict:
                    entcornerdict[n33] = []
                entcornerdict[n33].append(d2)
            if stempred == True:
                predalisstem = set([stemword(items) for items in predalias_tokens[n2.lower()]])
                if n2splitstem.intersection(Cornerstone) or Cornerstone.intersection(predalisstem):
                    if n22 not in entcornerdict:
                        entcornerdict[n22] = []
                    entcornerdict[n22].append(d2)
            else:
                if n2split.intersection(Cornerstone) or Cornerstone.intersection(predalias_tokens[n2.lower()]):
                    if n22 not in entcornerdict:
                        entcornerdict[n22] = []
                    entcornerdict[n22].append(d2)

            if n11 not in G.nodes():
                G.add_node(n11,weight=0.0, did=['kg'], dtitle=['kg'], sid=['kg'], matched=unique_SPO_dict[(nn1,nn2,nn3)]['matched_n1'], source='KG')
            G.add_node(n22,weight=0.0, did=['kg'], dtitle=['kg'], sid=['kg'], matched='', source='KG')
            if n33 not in G.nodes():
                G.add_node(n33,weight=0.0,did=['kg'], dtitle=['kg'], sid=['kg'], matched=unique_SPO_dict[(nn1,nn2,nn3)]['matched_n3'], source='KG')
            #G.add_edge(n11,n22, weight=0.0, wlist=[0.0], etype='Triple')
            #G.add_edge(n22,n33, weight=0.0, wlist=[0.0], etype='Triple')
            add_edge_triple_KG(G,n11,n22,d1)
            add_edge_triple_KG(G,n22,n33,d1)

        else:
            if n2 not in pred_count:
                pred_count[n2]=1
            else:
                pred_count[n2]=pred_count[n2]+1
            n11=n1.lower()+":Entity"
            n22=n2.lower()+":Predicate:"+str(pred_count[n2])
            n33=n3.lower()+":Entity"
            #get the weights for each node that is an anchor
            if n1 in entity_names:
                if n11 not in entcornerdict:
                    entcornerdict[n11] = []
                entcornerdict[n11].append(d1)
            if n3 in entity_names:
                if n33 not in entcornerdict:
                    entcornerdict[n33] = []
                entcornerdict[n33].append(d2)
            if stempred == True:
                predalisstem = set([stemword(items) for items in predalias_tokens[n2.lower()]])
                if n2splitstem.intersection(Cornerstone) or Cornerstone.intersection(predalisstem):
                    if n22 not in entcornerdict:
                        entcornerdict[n22] = []
                    entcornerdict[n22].append(d2)
            else:
                if n2split.intersection(Cornerstone) or Cornerstone.intersection(predalias_tokens[n2.lower()]):
                    if n22 not in entcornerdict:
                        entcornerdict[n22] = []
                    entcornerdict[n22].append(d2)



            if n11 not in G.nodes():
                G.add_node(n11,weight=0.0,did=['kg'], dtitle=['kg'], sid=['kg'], matched=unique_SPO_dict[(nn1,nn2,nn3)]['matched_n1'], source='KG')
            G.add_node(n22,weight=0.0,did=['kg'], dtitle=['kg'], sid=['kg'], matched='', source='KG')
            if n33 not in G.nodes():
                G.add_node(n33,weight=0.0,did=['kg'], dtitle=['kg'], sid=['kg'], matched=unique_SPO_dict[(nn1,nn2,nn3)]['matched_n3'], source='KG')
            #G.add_edge(n11,n22, weight=0.0, wlist=[0.0], etype='Triple')
            #G.add_edge(n22,n33, weight=0.0, wlist=[0.0], etype='Triple')
            add_edge_triple_KG(G,n11,n22,d1)
            add_edge_triple_KG(G,n22,n33,d1)
        #ADD qualifier nodes edges
        if 'qualifier' in unique_SPO_dict[(nn1,nn2,nn3)] and n2 !='typeo of':
            #print('Qualifier oooo')
            #print((nn1,nn2,nn3), unique_SPO_dict[(nn1,nn2,nn3)])
            for qualct in range(0,len(unique_SPO_dict[(nn1,nn2,nn3)]['qualifier'])):
                qual=unique_SPO_dict[(nn1,nn2,nn3)]['qualifier'][qualct]
                qn2=qual[0]
                qn3=qual[1]
                if qn2.lower() in aux_list:
                    continue

                if qn2 not in pred_count:
                    pred_count[qn2]=1
                else:
                    pred_count[qn2]=pred_count[qn2]+1

                qn22=qn2.lower()+":Predicate:"+str(pred_count[qn2])
                qn33=qn3.lower()+":Entity"

                d1 = unique_SPO_dict[(nn1,nn2,nn3)]['score_qual'][qualct]

                n2split = set(qn2.lower().split())
                n3split = set(qn3.lower().split())
                if qn3.lower() in entity_names:
                    if qn33 not in entcornerdict:
                        entcornerdict[qn33] = []
                    entcornerdict[qn33].append(d1)
                if stempred == True:
                    n2splitstem = set([stemword(qts) for qts in n2split])
                    predalisstem = set([stemword(items) for items in predalias_tokens[qn2.lower()]])
                    if n2splitstem.intersection(Cornerstone) or Cornerstone.intersection(predalisstem):
                        if qn22 not in entcornerdict:
                            entcornerdict[qn22] = []
                        entcornerdict[qn22].append(d1)
                else:
                    if n2split.intersection(Cornerstone) or Cornerstone.intersection(predalias_tokens[qn2.lower()]):
                        if qn22 not in entcornerdict:
                            entcornerdict[qn22] = []
                        entcornerdict[qn22].append(d1)
                G.add_node(qn22,weight=0.0,did=['kg'], dtitle=['kg'], sid=['kg'], matched='', source='KG')
                if qn33 not in G.nodes():
                    G.add_node(qn33,weight=0.0,did=['kg'], dtitle=['kg'], sid=['kg'], matched=unique_SPO_dict[(nn1,nn2,nn3)]['matched_qual'][qualct], source='KG')
                #G.add_edge(n22,qn22, weight=0.0, wlist=[0.0], etype='Triple')
                #G.add_edge(qn22,qn33, weight=0.0, wlist=[0.0], etype='Triple')
                add_edge_triple_KG(G,n22,qn22,d1)
                add_edge_triple_KG(G,qn22,qn33,d1)
        c+=1

    nodeweightavg = dict((k, mean(v)) for k, v in entcornerdict.items())
    #print("node scores", nodeweightavg )

    #print(G.nodes())

    return G,nodeweightavg,entcornerdict,pred_count,type_count

############################################### The TextQA part is here ###################################################################################

def call_main_GRAPH_TEXT(G,topl,textspofile,textcooccur,textcontextscores,hearst_file,qtfile,topcontext,config,pred_count={} ):
    
    try:
        f22=open(textspofile,'r')
        f33=open(textcooccur,'r')
        f44=open(textcontextscores,'r')
        questn_tokens = [] # pickle.load(open(qtfile,'rb'))
    except FileNotFoundError as err:
        return G,[],[]
    contextscoDicts = {}
    Add_Type_flag = int(config['Add_Type'])
    
    #get the topl
    unique_SPO_dict = getSPOtopk(f22,f33,topl)
    #print('SPO Dict = ',unique_SPO_dict)

    #question term file
    #qtfile = writeqt+'/LcQUAD_'+qid
    #cornerstone_file = writecorner+'/LcQUAD_'+qid
    #allentpred = []
    print("question tokens == === === = ", questn_tokens)
    G,cornerdict, predcornerdict = build_graph_from_triple_edges_TEXT(G,unique_SPO_dict,questn_tokens,pred_count=pred_count)
    G = update_edge_weight(G)
    G = add_nodes_weights_TEXT(G,cornerdict, predcornerdict,meanval=False)
    if Add_Type_flag == 1:
        #hearst_file  = writehearst+'/LcQUAD_'+qid+'.json'
        final_hearst = json.load(open(hearst_file,'r'))['hearst']
        G = add_type_edges(G,final_hearst)

    return G,questn_tokens,list(unique_SPO_dict)

def nouse():
    if Add_Type_flag == 1:
        hearst_file  = writehearst+'/LcQUAD_'+qid+'.json'
        final_hearst = json.load(open(hearst_file,'r'))['hearst']
        G = add_type_edges(G,final_hearst)
    else:
        Type_Alignment_flag=0

    if degenerate == 1:
        #print("Setting weight to 1 ")
        G = degenerate(G)

    if Predicate_Alignment_flag == 1:
        #Add relation alignment edges from glove embeddings
        print ("\n\nAdding predicate alignment edges\n\n")
        predlookup  = pickle.load(open(writealign+'/LcQUAD_'+qid+'predicate.pickle','rb'))
        #print (" Pred lookup = ", type(predlookup), ' empty?? ',bool(predlookup))
        #print(predlookup)
        G = add_predicate_alignment_edges2(G,predlookup,pt)
    
    if Type_Alignment_flag==1:
        print ("\n\nAdding type alignment edges\n\n")
        typelookup  = pickle.load(open(writealign+'/LcQUAD_'+qid+'type.pickle','rb'))
        G = add_type_alignment_edges2(G,typelookup,tt)
    
    if Entity_Alignment_flag==1:
        print ("\n\nAdding entity alignment edges\n\n")
        entitylookup  = pickle.load(open(writealign+'/LcQUAD_'+qid+'entity.pickle','rb'))
        G = add_entity_alignment_edges2(G,entitylookup,et)
        
        if verbose:
            print("\n\nSize of the graph directed",len(G.nodes()),len(G.edges()))
    G=directed_to_undirected(G)
    #print ("Size of the graph ",len(G.nodes()),len(G.edges()))
    if len(G.nodes())>0:
        G=max(nx.connected_component_subgraphs(G), key=len)
        print ("\n\nSize of the graph ",len(G.nodes()),len(G.edges()))
    
    G = add_nodes_weights_TEXT(G,cornerdict, predcornerdict,meanval=False)
    
    corner = {}
    corner = getcornerstone_TEXT(G,questn_tokens)
    #print("corner = ", corner)
    print("cornerstone_file = ", cornerstone_file)
    pickle.dump(corner,open(cornerstone_file,'wb'))






############################################## The Alignment part is here #################################################################################
##compute and write to file

def get_vector(n,gdict):
        veclen=300
        n=n.lower()
        nw1=n.replace('-',' ').split()
        avec=np.zeros(veclen)
        c=0.0
        for el in nw1:
                if el in gdict and el.lower() not in stop_list:
                        avec=np.add(avec,np.array(gdict[el]))
                        c+=1.0
        if c>0:
                avec=np.divide(avec, c)
        return avec

def cosine_similarity(a,b):
        s1=0.0
        s2=0.0
        s3=0.0
        if len(a)!=len(b):
                return 0.0
        for i in range(0,len(a)):
                s1+=a[i]*b[i]
                s2+=a[i]*a[i]
                s3+=b[i]*b[i]
        if s2>0 and s3>0:
                val=(s1/(math.sqrt(s2)*math.sqrt(s3)))
                #if val<0.0:
                #       print "negative val"
                val_norm=(val+1.0)/2.0
                return val_norm
        else:
                return 0

def cosine_similarity_MAX_MATCH(a,b,gdict):
        a=a.lower()
        aw1=a.replace('-',' ').split()
        b=b.lower()
        bw1=b.replace('-',' ').split()


        max_match=-1
        for el1 in aw1:
                if el1 in gdict and el1 not in stop_list:
                        avec=gdict[el1]
                        for el2 in bw1:
                                if el2 in gdict and el2 not in stop_list:
                                        bvec=gdict[el2]
                                        val=cosine_similarity(avec,bvec)
                                        if val>max_match:
                                                max_match=val
        return max_match

def getcharngrams(text, n=3):
    return [text[i:i+n] for i in range(len(text)-(n-1))]

def addhash(text):
    return ['#'+word+'#' for word in text]

def flatten_list(listx):
    return [item for sublist in listx for item in sublist]

def jaccard_similarity(list1, list2):
    list1 = set(list1)
    list2 = set(list2)
    intersection = len(list1.intersection(list2))
    union = len(list1.union(list2))
    if union <= 0:
        return 0.0
    return float(intersection) / union

def update_edge_weight_TEXT(G):
    for (n11,n22) in G.edges():
        if (n22.split(':')[1]=='Predicate') or (n11.split(':')[1]=='Predicate'):
            weights = G[n11][n22]['wlist']
            avgweights = mean(weights)
            G[n11][n22]['weight'] = avgweights
            G[n11][n22]['wlist'] = [avgweights]
    return G

def build_graph_from_triple_edges_TEXT(G,unique_SPO_dict,Cornerstone,pred_count={}):
    #G = nx.DiGraph()
    c=0
    Cornerstone = set(Cornerstone)
    #pred_count={}
    spo_wt={}
    """
        #this section of his code gets the question words in q_ent and compute their embeddingss
        for n in q_ent:
                if n not in done:
                        done.add(n)
                        nw1=n.replace('-',' ').split()
                        avec=np.zeros(veclen)
                        c=0.0
                        for el in nw1:
                                if el in gdict and el.lower() not in stop_list:
                                        #if option=='GLOVE':
                                        #       avec=np.add(avec,gdict[el])
                                        #else:
                                        avec=np.add(avec,np.array(gdict[el]))
                                        c+=1.0
                                if c>0:
                                        avec=np.divide(avec, c)
                                        g_ques[n]=avec.tolist()
    """
    cornerdict = {}
    predcornerdict = {}
    #corners  = [x.lower() for x in Cornerstone]
    for (nn1,nn2,nn3) in unique_SPO_dict:
        #if bool(set((nn1,nn2,nn3)).intersection(Cornerstone))==False:
        #        continue

        doc_id=unique_SPO_dict[(nn1,nn2,nn3)]['doc_id']
        doc_title=unique_SPO_dict[(nn1,nn2,nn3)]['doc_title']
        sent_id=unique_SPO_dict[(nn1,nn2,nn3)]['sent_id']
        d1=unique_SPO_dict[(nn1,nn2,nn3)]['d1']
        d2=unique_SPO_dict[(nn1,nn2,nn3)]['d2']
        #print(d1)
        n1=nn1.lower()
        n2=nn2.lower()
        n3=nn3.lower()

        n1split = set(n1.split())
        n2split = set(n2.split())
        n3split = set(n3.split())
        if n1split.intersection(Cornerstone) :
            if n1 not in cornerdict:
                cornerdict[n1] = []
            #print(" Adding n1 ",n1, " that is a cornerstone ******************")
            cornerdict[n1].extend(d1)
            #print("Corner dict = ",cornerdict)
        if n3split.intersection(Cornerstone):
            if n3 not in cornerdict:
                cornerdict[n3] = []
            #print(" Adding n3 ",n3, " that is a cornerstone *******************")
            cornerdict[n3].extend(d2)
            #print("Corner dict = ",cornerdict)

        if n2split.intersection(Cornerstone):
            if n2 not in predcornerdict:
                predcornerdict[n2] = []
            #print(" Adding n2 ",n2, " that is a cornerstone *******************")
            predcornerdict[n2].extend(d2)
            #print("Corner dict = ",predcornerdict)

        #print("Corner dict = ",cornerdict)




        n11=n1.lower()+":Entity"
        n22=n2.lower()+":Predicate"
        n33=n3.lower()+":Entity"

        #Check if triple contains auxiliary verb or blank or single letter predicate, then remove it
        if n2=='' or n2 in aux_list or len(n2)==1:
            continue
        #Flags to check if the same left part or right part has appeared earlier, then find out the predicate index
        left_flag=0
        left_index=-1
        right_flag=0
        right_index=-1
        #print "Current Edges \n"
        for (x,y) in G.edges():
            data=G.get_edge_data(x,y)
            #print x, y, data,y.split(':')[1],y.split(':')[0],n22.split(':')[0],x,n11
            if y.split(':')[1]=='Predicate' and y.split(':')[0]==n22.split(':')[0] and x==n11 and len(set(data['did']).intersection(set(doc_id)))>0 and len(set(data['dtitle']).intersection(set(doc_title)))>0 and len(set(data['sid']).intersection(set(sent_id)))>0:
                left_flag=0
                left_index=y.split(':')[2]

            if x.split(':')[1]=='Predicate' and x.split(':')[0]==n22.split(':')[0] and y==n33 and len(set(data['did']).intersection(set(doc_id)))>0 and len(set(data['dtitle']).intersection(set(doc_title)))>0 and len(set(data['sid']).intersection(set(sent_id)))>0:
                right_flag=0
                right_index=x.split(':')[2]

        #print "flags -->",left_flag,left_index,right_flag,right_index
        if left_flag==1 and right_flag==0:
            #left part of SPO already there
            if n22 not in spo_wt:
                spo_wt[n22]={}
            if n33 not in spo_wt[n22]:
                spo_wt[n22][n33]=[]
            spo_wt[n22][n33].append(d2)

            n22=n22+':'+left_index

            G=add_node_triple_TEXT(G,n33,doc_id,doc_title, sent_id)
            G=add_edge_triple_TEXT(G,n22,n33,d2,doc_id,doc_title, sent_id)

        if right_flag==1 and left_flag==0:
            #Right part of SPO already there
            if n11 not in spo_wt:
                spo_wt[n11]={}
            if n22 not in spo_wt[n11]:
                spo_wt[n11][n22]=[]
            spo_wt[n11][n22].append(d1)
            n22=n22+':'+right_index
            G=add_node_triple_TEXT(G,n11,doc_id,doc_title, sent_id)
            G=add_edge_triple_TEXT(G,n11,n22,d1,doc_id,doc_title, sent_id)

        if left_flag==0 and right_flag==0:
            #no parts of SPO already there
            if n11 not in spo_wt:
                spo_wt[n11]={}
            if n22 not in spo_wt[n11]:
                spo_wt[n11][n22]=[]
            spo_wt[n11][n22].append(d1)

            if n22 not in spo_wt:
                spo_wt[n22]={}
            if n33 not in spo_wt[n22]:
                spo_wt[n22][n33]=[]
            spo_wt[n22][n33].append(d2)

            #Supply unique index to predicate n2
            n2=n2.lower()
            if n2 not in pred_count:
                pred_count[n2]=1
            else:
                pred_count[n2]=pred_count[n2]+1
            n22=n22+":"+str(pred_count[n2])

            G=add_node_triple_TEXT(G,n11,doc_id,doc_title, sent_id)
            #G=add_node_triple(G,n2.lower()+":Predicate",doc_id,doc_title, sent_id)
            G=add_node_triple_TEXT(G,n22,doc_id,doc_title,sent_id)
            G=add_node_triple_TEXT(G,n33,doc_id,doc_title, sent_id)
            G=add_edge_triple_TEXT(G,n11,n22,d1,doc_id,doc_title, sent_id)
            G=add_edge_triple_TEXT(G,n22,n33,d2,doc_id,doc_title, sent_id)
    return G,cornerdict, predcornerdict


def add_nodes_weights_TEXT(G,entdict,predict,meanval=False):
    for n1 in G.nodes():
        if G.node[n1]['source']!='TEXT':
            continue
        nn1=n1.split(':')
        if nn1[1]=='Entity':
            if nn1[0] in entdict:
                if meanval == True:
                    max_val = mean(entdict[nn1[0]])
                    #print("The weights tother = *********************************************************", entdict[nn1[0]], " The mean = ", max_val)

                elif meanval == False:
                    max_val = sum(entdict[nn1[0]])
                #print("The weights tother = *********************************************************", entdict[nn1[0]], " The mean = ", max_val)

                G.node[n1]['weight']=round(max_val,2)
        elif nn1[1] == 'Predicate':
            if nn1[0] in predict:
                if meanval == True:
                    max_val = mean(predict[nn1[0]])
                elif meanval == False:
                    max_val = sum(predict[nn1[0]])
                #print("The weights tother = *********************************************************", predict[nn1[0]], " The mean = ", max_val)
                G.node[n1]['weight']=round(max_val,2)
    return G

def add_node_triple_TEXT(G,n1,doc_id,doc_title, sent_id):
        if n1 not in G.nodes():
                did1=[]
                for dd in doc_id:
                        did1.append(dd)

                dtitle1=[]
                for dt in doc_title:
                        dtitle1.append(dt)

                sid1=[]
                for si in sent_id:
                        sid1.append(si)
                G.add_node(n1,weight=0.0,matched='', did=did1, dtitle=dtitle1, sid=sid1, source='TEXT')
                if verbose:
                        print ("New ", n1,G.node[n1]    )
        else:
                if verbose:
                        print ("Existing ", n1,G.node[n1],G.node[n1]['did'])
                for doc_id1 in doc_id:
                        G.node[n1]['did'].append(doc_id1)
                for doc_title1 in doc_title:
                        G.node[n1]['dtitle'].append(doc_title1)
                for sent_id1 in sent_id:
                        G.node[n1]['sid'].append(sent_id1)
                #G.add_node(n1,did=did1,dtitle=dtitle1,sid=sid1)
                if verbose:
                        print ("Existing Updated", n1,G.node[n1])
        return G


def add_edge_triple_TEXT(G,n1,n2,d,doc_id,doc_title, sent_id): #d, doc_id doc_title, sent_id all are lists
    wlist1=[]
    for d1 in d:
        d2=round(float(d1),2)#float(1.0/float(d1)),2) #No need to use d1+1. it is already taken care in SPOs
        wlist1.append(d2);#print("This is d = ",d, " d2 = ",d2, " wlist = ", wlist1)

    if (n1,n2) in G.edges():
        for (n11,n22) in G.edges():
            if (n2.split(':')[1]=='Predicate' and n22.split(':')[1]=='Predicate' and n2.split(':')[0]==n22.split(':')[0] and n1==n11) or (n1.split(':')[1]=='Predicate' and n11.split(':')[1]=='Predicate' and n1.split(':')[0]==n11.split(':')[0] and n2==n22):
                data=G.get_edge_data(n11,n22)
                d=data['weight']

                wlist=data['wlist']
                wlist.extend(wlist1)

                did1=data['did']
                did=[]
                for dd in doc_id:
                    did.append(dd)

                dtitle=[]
                for dt in doc_title:
                    dtitle.append(dt)

                did1.extend(did)
                dtitle1=data['dtitle']
                dtitle1.extend(dtitle)
                sid1=data['sid']
                sid=[]
                for si in sent_id:
                    sid.append(si)
                sid1.extend(sid)
                #if verbose:
                #print "yesss", n1,n2,d,d3


                G.add_edge(n1,n2,weight=d,wlist=wlist,etype='Triple',did=did1,dtitle=dtitle1,sid=sid1, source='TEXT')
                if verbose:
                    print ("Updated Triple Edge ",n1,n2,G.get_edge_data(n1,n2),d2,data['weight'])
    else:
        did1=[]
        for dd in doc_id:
            did1.append(dd)
        dtitle1=[]
        for dt in doc_title:
            dtitle1.append(dt)

        sid1=[]
        for si in sent_id:
            sid1.append(si)
        G.add_edge(n1,n2, weight=wlist1, wlist=wlist1, etype='Triple', did=did1, dtitle=dtitle1, sid=sid1, source='TEXT'); #print('using weight mean for edges .......................................................');
        if verbose:
            print ("New Triple Edge ",n1,n2,G.get_edge_data(n1,n2))
    return G

def add_type_edges(G,HP):
    countmatch = 0
    for pat in HP:
        p1=pat[0].lower()+':Entity'
        if p1 in G.nodes():
            countmatch += 1
            #print(p1, '   ', pat)
            p2=remove_leading_article(pat[1]).lower()
            #print ("Removed articles ",p2)
            #if p2+':Entity' in G.nodes():
            #       p2+=':Entity'
            #else:
            p2+=':Type' #creating type nodes irrespective of same name entity node exists or not
            if p2 not in G.nodes():
                G=add_node_triple_TEXT(G, p2, [], [], []) #Adding a "Type" node; not using typical nodes with only labeled "type"
                G.add_edge(p1, p2, weight=1.0, wlist=[1.0], etype='Type',did=[],dtitle=[],sid=[], source='TEXT')
            else:
                if (p1,p2) not in G.edges():
                    G.add_edge(p1, p2, weight=1.0, wlist=[1.0], etype='Type',did=[],dtitle=[],sid=[], source='TEXT') #Assuming there do not exist other edges between entity and "Type" nodes


    #print("Entity match = ", countmatch, ' whole hearst = ',len(HP))
    return G









def getSPOtopk(f22,f33,thefinaltopcontext):
    contextscoDicts = {}
    unique_SPO_dict = dict()
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
        if (doc_id,sent_id) not in thefinaltopcontext:
            #print((doc_id,sent_id),' not part of topk')
            continue
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
        if (doc_id,sent_id) not in thefinaltopcontext:
            #print((doc_id,sent_id),' not part of topk')
            continue
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
    ##to check if mean will do
    newSPOdict = {}

    for (n1,n2,n3) in unique_SPO_dict:
        newSPOdict[(n1,n2,n3)] = {}
        newSPOdict[(n1,n2,n3)]['d1'] = [mean(unique_SPO_dict[(n1,n2,n3)]['d1'])]
        newSPOdict[(n1,n2,n3)]['d2'] =  [mean(unique_SPO_dict[(n1,n2,n3)]['d2'])]
        newSPOdict[(n1,n2,n3)]['doc_id'] = unique_SPO_dict[(n1,n2,n3)]['doc_id']
        newSPOdict[(n1,n2,n3)]['doc_title'] = unique_SPO_dict[(n1,n2,n3)]['doc_title']
        newSPOdict[(n1,n2,n3)]['sent_id'] = unique_SPO_dict[(n1,n2,n3)]['sent_id']
    #return unique_SPO_dict
    return newSPOdict

def getSPO(f22,f33):
    unique_SPO_dict = dict()
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
    ##to check if mean will do
    newSPOdict = {}
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
    for (n1,n2,n3) in unique_SPO_dict:
        newSPOdict[(n1,n2,n3)] = {}
        newSPOdict[(n1,n2,n3)]['d1'] = [mean(unique_SPO_dict[(n1,n2,n3)]['d1'])]
        newSPOdict[(n1,n2,n3)]['d2'] =  [mean(unique_SPO_dict[(n1,n2,n3)]['d2'])]
        newSPOdict[(n1,n2,n3)]['doc_id'] = unique_SPO_dict[(n1,n2,n3)]['doc_id']
        newSPOdict[(n1,n2,n3)]['doc_title'] = unique_SPO_dict[(n1,n2,n3)]['doc_title']
        newSPOdict[(n1,n2,n3)]['sent_id'] = unique_SPO_dict[(n1,n2,n3)]['sent_id']
    #return unique_SPO_dict
    return newSPOdict

def getEntities(G,mode='KG'):
    entities = []
    for n1 in G.nodes():
        nn1=n1.split(':')
        if nn1[1]=='Entity' and  G.node[n1]['source']==mode:
            entities.append(n1)
    return entities 



##use computed values from file
def add_entity_alignment_edges1(G,mention_dict,g_ent,files):
    if os.path.exists(files):
        # file exists
        with open(files, 'rb') as handle:
            xdict = pickle.load(handle)
    else:
        xdict = defaultdict()    
    for n1 in G.nodes():
        nn1=n1.split(':')
        if nn1[1]=='Entity':
            for n2 in G.nodes():
                nn2=n2.split(':')
                if n2!=n1 and nn2[1]=='Entity' and (G.node[n1]['source']!='KG' or G.node[n2]['source']!='KG'):
                    if tuple(sorted((nn1[0],nn2[0]))) in xdict:
                        continue
                    part1=nn1[0].replace(',',' ')
                    part1=part1.replace('-',' ')
                    wn1=part1.split()
                    part2=nn2[0].replace(',',' ')
                    part2=part2.replace('-',' ')
                    wn2=part2.split()
                    x = addhash(wn1)
                    x = [getcharngrams(t) for  t in x ]
                    x = flatten_list(x)
                    y = addhash(wn2)
                    y = [getcharngrams(t) for  t in y ]
                    y = flatten_list(y)

                    jack = jaccard_similarity(x,y)
                    #print("n1 = ",n1, " n2 = ", n2, "Jack = ", jack,round(jack,2))
                    #xdict[tuple(sorted((nn1[0],nn2[0])))] = jack
                    if jack>=threshold_align:
                        #print("n1 = ",n1, " n2 = ", n2, "Jack = ",round(jack,2),G.node[n1]['source'],G.node[n2]['source'])
                        xdict[tuple(sorted((nn1[0],nn2[0])))] = jack
                        value=round(jack,2)
                        #Add bidirectional alignment edges between the predicates; no associated doc with edge
                        if G.node[n1]['source']=='TEXT' and G.node[n2]['source']=='TEXT':
                            G.add_edge(n1,n2,weight=value,wlist=[value],etype='Entity_Alignment',did=[],dtitle=[],sid=[], source='TEXT')
                            G.add_edge(n2,n1,weight=value,wlist=[value],etype='Entity_Alignment',did=[],dtitle=[],sid=[], source='TEXT')
                        else:
                            G.add_edge(n1,n2,weight=value,wlist=[value],etype='Entity_Alignment',did=[],dtitle=[],sid=[], source='KG_TEXT')
                            G.add_edge(n2,n1,weight=value,wlist=[value],etype='Entity_Alignment',did=[],dtitle=[],sid=[], source='KG_TEXT')
                        if verbose:
                            print ("Entity Alignment Edge ",n1,n2,G.get_edge_data(n1,n2))
                            print ("Entity Alignment Edge ",n2,n1,G.get_edge_data(n2,n1))
    #write the deafult dict to file

    return G, xdict
def add_entity_alignment_edges2(G,lookup,et):
    for n1 in G.nodes():
        nn1=n1.split(':')
        if nn1[1]=='Entity':
            for n2 in G.nodes():
                nn2=n2.split(':')
                if n2!=n1 and nn2[1]=='Entity' and (G.node[n1]['source']!='KG' or G.node[n2]['source']!='KG') and tuple(sorted((nn1[0],nn2[0]))) in lookup:
                    jack = lookup[tuple(sorted((nn1[0],nn2[0])))]
                    #print("n1 = ",n1, " n2 = ", n2, "Jack = ", jack, round(jack,2))
                    #if jack>0: print("******************************************************************************************************************************")
                    if jack>=et:
                        #print('entity align without round = ',jack)
                        value=round(jack,2)
                        #print("entity alignment ", value)
                        #print('n1 = ', nn1[0], 'n2 = ',nn2[0], value)
                        #Add bidirectional alignment edges between the predicates; no associated doc with edge
                        if G.node[n1]['source']=='TEXT' and G.node[n2]['source']=='TEXT':
                            dids = G.node[n1]['did'] + G.node[n2]['did']
                            sids = G.node[n1]['sid'] + G.node[n2]['sid']
                            dtitles = G.node[n1]['dtitle'] + G.node[n2]['dtitle']
                            G.add_edge(n1,n2,weight=value,wlist=[value],etype='Entity_Alignment',did=dids,dtitle=dtitles,sid=sids, source='TEXT')
                            G.add_edge(n2,n1,weight=value,wlist=[value],etype='Entity_Alignment',did=dids,dtitle=dtitles,sid=sids, source='TEXT')
                            #G.add_edge(n1,n2,weight=value,wlist=[value],etype='Entity_Alignment',did=[],dtitle=[],sid=[], source='TEXT')
                            #G.add_edge(n2,n1,weight=value,wlist=[value],etype='Entity_Alignment',did=[],dtitle=[],sid=[], source='TEXT')
                        else:
                            dids = G.node[n1]['did'] + G.node[n2]['did']
                            sids = G.node[n1]['sid'] + G.node[n2]['sid']
                            dtitles = G.node[n1]['dtitle'] + G.node[n2]['dtitle']
                            G.add_edge(n1,n2,weight=value,wlist=[value],etype='Entity_Alignment',did=dids,dtitle=dtitles,sid=sids, source='KG_TEXT')
                            G.add_edge(n2,n1,weight=value,wlist=[value],etype='Entity_Alignment',did=dids,dtitle=dtitles,sid=sids, source='KG_TEXT')
                        if verbose:
                            print ("Entity Alignment Edge ",n1,n2,G.get_edge_data(n1,n2))
                            print ("Entity Alignment Edge ",n2,n1,G.get_edge_data(n2,n1))
                            #write the deafult dict to file
    return G

def add_predicate_alignment_edges1(G,g_pred,gdict,files):
    if os.path.exists(files):
        # file exists
        with open(files, 'rb') as handle:
            xdict = pickle.load(handle)
    else:
        xdict = defaultdict()

    if MAX_MATCH==0:
        for n1 in G.nodes():
            nn1=n1.split(':')
            if nn1[1]=='Predicate' and n1 in g_pred:
                for n2 in G.nodes():
                    nn2=n2.split(':')
                    if n2!=n1 and nn2[1]=='Predicate' and n2 in g_pred and (G.node[n1]['source']!='KG' or G.node[n2]['source']!='KG'):
                        if tuple(sorted((nn1[0],nn2[0]))) in xdict:
                            continue
                        value=cosine_similarity(g_pred[n1],g_pred[n2])
            
                        #print("n1 = ",n1, " n2 = ", n2, "Jack = ", value)
                        #print("n1 = ",n1, " n2 = ", n2, "Jack = ",round(jack,2),G.node[n1]['source'],G.node[n2]['source'])
                        #xdict[tuple(sorted((nn1[0],nn2[0])))] = value
                        if value>=threshold_align:
                            #print("n1 = ",n1, " n2 = ", n2, "Jack = ",round(value,2),G.node[n1]['source'],G.node[n2]['source'])
                            xdict[tuple(sorted((nn1[0],nn2[0])))] = value
                            value=round(value,2)
                            #Add bidirectional alignment edges between the predicates; no associated doc with edge
                            if G.node[n1]['source']=='TEXT' and G.node[n2]['source']=='TEXT':
                                G.add_edge(n1,n2,weight=value,wlist=[value],etype='Predicate_Alignment',did=[],dtitle=[],sid=[], source='TEXT')
                                G.add_edge(n2,n1,weight=value,wlist=[value],etype='Predicate_Alignment',did=[],dtitle=[],sid=[], source='TEXT')
                            else:
                                G.add_edge(n1,n2,weight=value,wlist=[value],etype='Predicate_Alignment',did=[],dtitle=[],sid=[], source='KG_TEXT')
                                G.add_edge(n2,n1,weight=value,wlist=[value],etype='Predicate_Alignment',did=[],dtitle=[],sid=[], source='KG_TEXT')
                            if verbose:
                                print ("Predicate Alignment Edge ",n1,n2,G.get_edge_data(n1,n2))
                                print ("Predicate Alignment Edge ",n2,n1,G.get_edge_data(n2,n1))
    else:
        for n1 in G.nodes():
            nn1=n1.split(':')
            if nn1[1]=='Predicate':
                for n2 in G.nodes():
                    nn2=n2.split(':')
                    if n2!=n1 and nn2[1]=='Predicate':
                        value=cosine_similarity_MAX_MATCH(nn1[0],nn2[0],gdict)
                        #xdict[tuple(sorted((nn1[0],nn2[0])))] = value
                        if value>=threshold_align:
                            xdict[tuple(sorted((nn1[0],nn2[0])))] = value
                            value=round(value,2)
                            #Add bidirectional alignment edges between the predicates; no associated doc with edge
                            if G.node[n1]['source']=='TEXT' and G.node[n2]['source']=='TEXT':
                                G.add_edge(n1,n2,weight=value,wlist=[value],etype='Predicate_Alignment',did=[],dtitle=[],sid=[], source='TEXT')
                                G.add_edge(n2,n1,weight=value,wlist=[value],etype='Predicate_Alignment',did=[],dtitle=[],sid=[], source='TEXT')
                            else:
                                G.add_edge(n1,n2,weight=value,wlist=[value],etype='Predicate_Alignment',did=[],dtitle=[],sid=[], source='KG_TEXT')
                                G.add_edge(n2,n1,weight=value,wlist=[value],etype='Predicate_Alignment',did=[],dtitle=[],sid=[], source='KG_TEXT')
                            if verbose:
                                print ("Predicate Alignment Edge ",n1,n2,G.get_edge_data(n1,n2))
                                print ("Predicate Alignment Edge ",n2,n1,G.get_edge_data(n2,n1) )
    return G, xdict
def add_predicate_alignment_edges2(G,xdict,pt):
    #print(xdict)
    if MAX_MATCH==0:
        for n1 in G.nodes():
            nn1=n1.split(':')
            if nn1[1]=='Predicate' :
                for n2 in G.nodes():
                    nn2=n2.split(':')
                    if n2!=n1 and nn2[1]=='Predicate' and (G.node[n1]['source']!='KG' or G.node[n2]['source']!='KG') and tuple(sorted((nn1[0],nn2[0]))) in xdict:
                        #print(n1,n2)
                        value = xdict[tuple(sorted((nn1[0],nn2[0])))]
                        if value>=pt:
                            value=round(value,2)
                            #print(nn1[0], nn2[0],'predicate alignment = ',value)
                            #Add bidirectional alignment edges between the predicates; no associated doc with edge
                            if G.node[n1]['source']=='TEXT' and G.node[n2]['source']=='TEXT':
                                dids = G.node[n1]['did'] + G.node[n2]['did']
                                sids = G.node[n1]['sid'] + G.node[n2]['sid']
                                dtitles = G.node[n1]['dtitle'] + G.node[n2]['dtitle']
                                G.add_edge(n1,n2,weight=value,wlist=[value],etype='Predicate_Alignment',did=dids,dtitle=dtitles,sid=sids, source='TEXT')
                                G.add_edge(n2,n1,weight=value,wlist=[value],etype='Predicate_Alignment',did=dids,dtitle=dtitles,sid=sids, source='TEXT')
                                #G.add_edge(n1,n2,weight=value,wlist=[value],etype='Predicate_Alignment',did=[],dtitle=[],sid=[], source='TEXT')
                                #G.add_edge(n2,n1,weight=value,wlist=[value],etype='Predicate_Alignment',did=[],dtitle=[],sid=[], source='TEXT')
                            else:
                                dids = G.node[n1]['did'] + G.node[n2]['did']
                                sids = G.node[n1]['sid'] + G.node[n2]['sid']
                                dtitles = G.node[n1]['dtitle'] + G.node[n2]['dtitle']
                                G.add_edge(n1,n2,weight=value,wlist=[value],etype='Predicate_Alignment',did=dids,dtitle=dtitles,sid=sids, source='KG_TEXT')
                                G.add_edge(n2,n1,weight=value,wlist=[value],etype='Predicate_Alignment',did=dids,dtitle=dtitles,sid=sids, source='KG_TEXT')
                            if verbose:
                                print ("Predicate Alignment Edge ",n1,n2,G.get_edge_data(n1,n2))
                                print ("Predicate Alignment Edge ",n2,n1,G.get_edge_data(n2,n1))
    else:
        for n1 in G.nodes():
            nn1=n1.split(':')
            if nn1[1]=='Predicate' and n1 in g_pred:
                for n2 in G.nodes():
                    nn2=n2.split(':')
                    if n2!=n1 and nn2[1]=='Predicate' and tuple(sorted((nn1[0],nn2[0]))) in xdict:
                        #value=cosine_similarity_MAX_MATCH(nn1[0],nn2[0],gdict)
                        value = xdict[tuple(sorted((nn1[0],nn2[0])))]
                        if value>=pt:
                            value=round(value,2)
                            #Add bidirectional alignment edges between the predicates; no associated doc with edge
                            if G.node[n1]['source']=='TEXT' and G.node[n2]['source']=='TEXT':
                                G.add_edge(n1,n2,weight=value,wlist=[value],etype='Predicate_Alignment',did=[],dtitle=[],sid=[], source='TEXT')
                                G.add_edge(n2,n1,weight=value,wlist=[value],etype='Predicate_Alignment',did=[],dtitle=[],sid=[], source='TEXT')
                            else:
                                G.add_edge(n1,n2,weight=value,wlist=[value],etype='Predicate_Alignment',did=[],dtitle=[],sid=[], source='KG_TEXT')
                                G.add_edge(n2,n1,weight=value,wlist=[value],etype='Predicate_Alignment',did=[],dtitle=[],sid=[], source='KG_TEXT')
                            if verbose:
                                print ("Predicate Alignment Edge ",n1,n2,G.get_edge_data(n1,n2))
                                print ("Predicate Alignment Edge ",n2,n1,G.get_edge_data(n2,n1) )
    return G


def add_type_alignment_edges1(G,g_type,gdict,files):
    if os.path.exists(files):
        # file exists
        with open(files, 'rb') as handle:
            xdict = pickle.load(handle)
    else:
        xdict = defaultdict()

    dict_items = xdict.items()
    #print('xdict = ',dict_items[:5])
    if MAX_MATCH==0:
        for n1 in G.nodes():
            nn1=n1.split(':')
            if nn1[1]=='Type' and n1 in g_type:
                for n2 in G.nodes():
                    nn2=n2.split(':')
                    if n2!=n1 and nn2[1]=='Type' and (G.node[n1]['source']!='KG' or G.node[n2]['source']!='KG') and n2 in g_type:
                        if tuple(sorted((nn1[0],nn2[0]))) in xdict:
                            continue
                        value=cosine_similarity(g_type[n1],g_type[n2])
                        #print("n1 = ",n1, " n2 = ", n2, "Jack = ", value)
                        #xdict[tuple(sorted((nn1[0],nn2[0])))] = value
                        if value>=threshold_align:
                            xdict[tuple(sorted((nn1[0],nn2[0])))] = value
                            value=round(value,2)
                            #print("n1 = ",n1, " n2 = ", n2, "Jack = ",round(value,2),G.node[n1]['source'],G.node[n2]['source'])
                            #Add bidirectional alignment edges between the predicates; no associated doc with edge
                            if G.node[n1]['source']=='TEXT' and G.node[n2]['source']=='TEXT':
                                #print('Got some text types higher than threshold :     ',nn1[0],nn2[0])
                                G.add_edge(n1,n2,weight=value,wlist=[value],etype='Type_Alignment',did=[],dtitle=[],sid=[], source='TEXT')
                                G.add_edge(n2,n1,weight=value,wlist=[value],etype='Type_Alignment',did=[],dtitle=[],sid=[], source='TEXT')
                            else:
                                #print('Got some KG text types higher than threshold :     ',nn1[0],nn2[0])
                                G.add_edge(n1,n2,weight=value,wlist=[value],etype='Type_Alignment',did=[],dtitle=[],sid=[], source='KG_TEXT')
                                G.add_edge(n2,n1,weight=value,wlist=[value],etype='Type_Alignment',did=[],dtitle=[],sid=[], source='KG_TEXT')
                            if verbose:
                                print ("Type Alignment Edge ",n1,n2,G.get_edge_data(n1,n2))
                                print ("type Alignment Edge ",n2,n1,G.get_edge_data(n2,n1))
    else:
        for n1 in G.nodes():
            nn1=n1.split(':')
            if nn1[1]=='Type':
                for n2 in G.nodes():
                    nn2=n2.split(':')
                    if n2!=n1 and nn2[1]=='Type'and (G.node[n1]['source']!='KG' or G.node[n2]['source']!='KG'):
                        value=cosine_similarity_MAX_MATCH(nn1[0],nn2[0],gdict)
                        #xdict[tuple(sorted((nn1[0],nn2[0])))] = value
                        if value>=threshold_align:
                            xdict[tuple(sorted((nn1[0],nn2[0])))] = value
                            value=round(value,2)
                            #Add bidirectional alignment edges between the predicates; no associated doc with edge
                            G.add_edge(n1,n2,weight=value,wlist=[value],etype='Type_Alignment',did=[],dtitle=[],sid=[])
                            G.add_edge(n2,n1,weight=value,wlist=[value],etype='Type_Alignment',did=[],dtitle=[],sid=[])
                            if verbose:
                                print ("Type Alignment Edge ",n1,n2,G.get_edge_data(n1,n2))
                                print ("Type Alignment Edge ",n2,n1,G.get_edge_data(n2,n1))
    return G,xdict

def add_type_alignment_edges2(G,xdict,tt):
    if MAX_MATCH==0:
        for n1 in G.nodes():
            nn1=n1.split(':')
            if nn1[1]=='Type' :
                for n2 in G.nodes():
                    nn2=n2.split(':')
                    if n2!=n1 and nn2[1]=='Type' and (G.node[n1]['source']!='KG' or G.node[n2]['source']!='KG') and tuple(sorted((nn1[0],nn2[0]))) in xdict:
                        value = xdict[tuple(sorted((nn1[0],nn2[0])))]
                        if value>=tt:
                            value=round(value,2)
                            #print('type alignment = ',value)
                            #Add bidirectional alignment edges between the predicates; no associated doc with edge
                            if G.node[n1]['source']=='TEXT' and G.node[n2]['source']=='TEXT':
                                dids = G.node[n1]['did'] + G.node[n2]['did']
                                sids = G.node[n1]['sid'] + G.node[n2]['sid']
                                dtitles = G.node[n1]['dtitle'] + G.node[n2]['dtitle']
                                G.add_edge(n1,n2,weight=value,wlist=[value],etype='Type_Alignment',did=dids,dtitle=dtitles,sid=sids, source='TEXT')
                                G.add_edge(n2,n1,weight=value,wlist=[value],etype='Type_Alignment',did=dids,dtitle=dtitles,sid=sids, source='TEXT')
                                #G.add_edge(n1,n2,weight=value,wlist=[value],etype='Type_Alignment',did=[],dtitle=[],sid=[], source='TEXT')
                                #G.add_edge(n2,n1,weight=value,wlist=[value],etype='Type_Alignment',did=[],dtitle=[],sid=[], source='TEXT')
                            else:
                                dids = G.node[n1]['did'] + G.node[n2]['did']
                                sids = G.node[n1]['sid'] + G.node[n2]['sid']
                                dtitles = G.node[n1]['dtitle'] + G.node[n2]['dtitle']
                                G.add_edge(n1,n2,weight=value,wlist=[value],etype='Type_Alignment',did=dids,dtitle=dtitles,sid=sids, source='KG_TEXT')
                                G.add_edge(n2,n1,weight=value,wlist=[value],etype='Type_Alignment',did=dids,dtitle=dtitles,sid=sids, source='KG_TEXT')
                            if verbose:
                                print ("Type Alignment Edge ",n1,n2,G.get_edge_data(n1,n2))
                                print ("type Alignment Edge ",n2,n1,G.get_edge_data(n2,n1))
    else:
        for n1 in G.nodes():
            nn1=n1.split(':')
            if nn1[1]=='Type':
                for n2 in G.nodes():
                    nn2=n2.split(':')
                    if n2!=n1 and nn2[1]=='Type' and (G.node[n1]['source']!='KG' or G.node[n2]['source']!='KG') and tuple(sorted((nn1[0],nn2[0]))) in xdict:
                        value=cosine_similarity_MAX_MATCH(nn1[0],nn2[0],gdict)
                        xdict[tuple(sorted((n1,n2)))] = value
                        if value>=tt:
                            value=round(value,2)
                            #Add bidirectional alignment edges between the predicates; no associated doc with edge
                            G.add_edge(n1,n2,weight=value,wlist=[value],etype='Type_Alignment',did=[],dtitle=[],sid=[])
                            G.add_edge(n2,n1,weight=value,wlist=[value],etype='Type_Alignment',did=[],dtitle=[],sid=[])
                            if verbose:
                                print ("Type Alignment Edge ",n1,n2,G.get_edge_data(n1,n2))
                                print ("Type Alignment Edge ",n2,n1,G.get_edge_data(n2,n1))
    return G

def directed_to_undirected(G1):
    G=nx.Graph()
    for n in G1.nodes():
        G.add_node(n,weight=G1.node[n]['weight'],matched=G1.node[n]['matched'])
    done=set()
    elist=[]
    for (n1,n2) in G1.edges():
        if (n1,n2) not in done:
            done.add((n1,n2))
            done.add((n2,n1))
            data=G1.get_edge_data(n1,n2)
            d=data['weight']
            wlist1=data['wlist']
            etype1=data['etype']
            did1=data['did']
            dtitle1=data['dtitle']
            sid1=data['sid']

            #print "n1,n2 ->",data

            if (n2,n1) in G1.edges():
                data1=G1.get_edge_data(n2,n1)
                #print "n2,n1 ->",data1
                if data1['etype']=='Triple':
                    if data1['weight']>d: #Keeping maximum weight edge
                        d=data1['weight']
                    for w in data1['wlist']:
                        #print "wlist",w,wlist1
                        wlist1.append(w)
                    for di in data1['did']:
                        #print('Data weight = ',data1['did'])#print "did",di,did1
                        did1.append(di)
                    for dt in data1['dtitle']:
                        #print "dtitle",dt,dtitle1
                        dtitle1.append(dt)
                    for si in data1['sid']:
                        #print "sid",si,sid1
                        sid1.append(si)
            for i in range(0,len(wlist1)):
                #print "i ",i
                if wlist1[i]>1.0 and wlist1[i]<=1.0001:
                    wlist1[i]=1.0
            if d>1.0 and d<=1.0001:
                d=1.0
            elist.append((n1,n2,d,wlist1,etype1))
            G.add_edge(n1,n2,weight=d, wlist=wlist1, etype=etype1,did=did1,dtitle=dtitle1,sid=sid1)

    flag=0
    elist=sorted(elist,key=lambda x:x[2],reverse=True)
    #for ee in elist:
    #    print "edges ", ee[0],ee[1],ee[2],ee[3],ee[4]

    for (n1,n2) in G.edges():
        data=G.get_edge_data(n1,n2)
        d=data['weight']
        wlist1=data['wlist']

        if d>1:
            print ('Error neg weight from QKG d..',n1,n2,data['etype'],d,wlist1)
            flag+=1
        for ww in wlist1:
            if ww>1:
                print ('Error neg weight from QKG wlist..',n1,n2,data['etype'],ww,wlist1)
                flag+=1
    print ("no. of neg weights ",flag)
    return G

def read_glove(G,q_ent,gdict,option='WORD2VEC'):
        veclen=300
        g_pred={}
        g_ent={}
        g_type={}
        g_ques={}
        done=set(); veclen = 100 if option=='WIKI' else 300; print('veclen ===',veclen)

        '''
        gdict={}

        if option=='GLOVE':
                glove_file='/home/pramanik/QUEST/files/glove.6B/glove.6B.300d.txt'
                fg=open(glove_file,'r')
                for line in fg:
                        line=(line.strip()).split()
                        vec=[]
                        for i in range(1,len(line)):
                                vec.append(float(line[i]))
                        gdict[line[0]]=np.array(vec)

        else:
                gdict=word_vectors
        '''


        for n in G.nodes():
                if n not in done:
                        done.add(n)
                        nn=n.split(':')
                        nw1=nn[0].replace('-',' ').split()
                        avec=np.zeros(veclen)
                        c=0.0
                        for el in nw1:
                                if el in gdict and el.lower() not in stop_list:
                                        #if option=='GLOVE':
                                        #       avec=np.add(avec,gdict[el])
                                        #else:
                                        avec=np.add(avec,np.array(gdict[el]))
                                        c+=1.0
                        if c>0:
                                avec=np.divide(avec, c)

                                if nn[1]=='Predicate' and nn[0].replace('-',' ') != 'cooccur':
                                        g_pred[n]=avec.tolist()
                                else:
                                        if nn[1]=='Entity':
                                                g_ent[n]=avec.tolist()
                                        else:
                                                g_type[n]=avec.tolist()
                        #if verbose:
                                #print len(G.nodes()),len(done)

        for n in q_ent:
                if n not in done:
                        done.add(n)
                        nw1=n.replace('-',' ').split()
                        avec=np.zeros(veclen)
                        c=0.0
                        for el in nw1:
                                if el in gdict and el.lower() not in stop_list:
                                        #if option=='GLOVE':
                                        #       avec=np.add(avec,gdict[el])
                                        #else:
                                        avec=np.add(avec,np.array(gdict[el]))
                                        c+=1.0
                        if c>0:
                                avec=np.divide(avec, c)
                                g_ques[n]=avec.tolist()


        return  g_pred,g_ent,g_type,g_ques

def remove_leading_article(t): #remove leading articles from type nodes detected by hearst algo
        t=t.split()
        s=''
        articles=['a','an','the','some','one','few']
        i=0
        for i in range(0,len(t)):
                if t[i] not in articles:
                        break
        #while t[i] in articles:
        #       i+=1
        if len(t)>0:
                s=t[i]
                for j in range(i+1,len(t)):
                        s+=' '+t[j]
        else:
                print ('PROBLEM leading article ',t)
        return s

def LookUpTable(qid,G,writealign,alignEnt,alignPred,alignType,et,pt,tt):
    
    if alignPred==True:
        print ("\nAdding predicate alignment edges\n")
        with open(writealign+'/LcQUAD_'+qid+'predicate.pickle', 'rb') as handle:
            predicatefile = pickle.load(handle)
            G=add_predicate_alignment_edges2(G,predicatefile,pt)
    
    if alignType==True:
        print ("\nAdding type alignment edges\n")
        with open(writealign+'/LcQUAD_'+qid+'type.pickle', 'rb') as handle:
            typefile = pickle.load(handle)
            G=add_type_alignment_edges2(G,typefile,tt)
    
    if alignEnt==True:
        print ("\nAdding entity alignment edges\n")
        with open(writealign+'/LcQUAD_'+qid+'entity.pickle', 'rb') as handle:
            entityfile = pickle.load(handle)
            G=add_entity_alignment_edges2(G,entityfile,et)
    return G


def createLookUpTable(qid,G,gdict,writealign,config):
    g_pred,g_ent,g_type,g_ques=read_glove(G,[],gdict,option=config['embedding'])
    Type_Alignment_flag = int(config['Type_Alignment'])
    Predicate_Alignment_flag = int(config['Predicate_Alignment'])
    Entity_Alignment_flag = int(config['Entity_Alignment'])
    xp = defaultdict()
    xp = defaultdict()
    xt = defaultdict()
    xdict = defaultdict()
    
    if Predicate_Alignment_flag==1:
        print ("\nAdding predicate alignment edges\n")
        FILE = writealign+'/LcQUAD_'+qid+'predicate.pickle'
        G,xp=add_predicate_alignment_edges1(G,g_pred,gdict,FILE)
        with open(FILE, 'wb') as handle:
            pickle.dump(xp, handle)
    
    if Type_Alignment_flag==1:
        print ("\nAdding type alignment edges\n")
        FILE = writealign+'/LcQUAD_'+qid+'type.pickle'
        G,xt=add_type_alignment_edges1(G,g_type,gdict,FILE)
        with open(FILE, 'wb') as handle:
            pickle.dump(xt, handle)
    
    if Entity_Alignment_flag==1:
        print ("\nAdding entity alignment edges\n")
        FILE = writealign+'/LcQUAD_'+qid+'entity.pickle'
        G,xe=add_entity_alignment_edges1(G,"",{},FILE)
        with open(FILE, 'wb') as handle:
            pickle.dump(xe, handle)

    xp = defaultdict()
    xp = defaultdict()
    xt = defaultdict()
    return G


#merge text and KG cornerstone
def mergeCornerstone(x,y):
    return {**x, **y}


def call_main_GRAPH(qid,kgspofile,kgtypefile,kgcontextscores,kgentity_dir, kgquestn_tokens,path_file,kgpred_aliases_dir, textspofile,textcooccur,textcontextscores, qtfile, hearst_file, config,topk,gdict,finalxg='', finalcorner='',et=0,pt=0,tt=0,ablate=0,we=''):
    #call the KG
    #call_main_GRAPH(kgspo_dir,kgtype_dir,kgentityname, kgquestn_tokens, kgaliases, textspo_dir, textcoocur_dir, textcontextscore_dir,textqt_dir,texthearst_dir,  config,   topcontext=topcontext )
    stempred = True if int(config['stem_pred']) == 1 else False
    degenerates = True if int(config['degenerate']) == 1 else False
    computeAlign = True if int(config['computeAlign']) == 1 else False
    alignDir = config['alignDir']
    verbose = int(config['verbose'])
    G = nx.DiGraph()
    pred_count = {}
    
    #get the scored triples and pick the best
    try:
        kgsn = [splittext(line.strip()) for line in open(kgcontextscores,'r')]
    except FileNotFoundError as err:
        kgsn = []

    try:
        kgsn.extend([splittext(line.strip()) for line in open(textcontextscores,'r')])
    except FileNotFoundError as err:
        pass
    kgsn = set(kgsn)
    topl = rank_and_select_triples(list(kgsn),topcontext=topk)
    #print(topl)

    #call the KG graph construction
    G, corner_KG, kgtrip = call_main_GRAPH_KG(kgspofile,topl,kgtypefile,kgentity_dir, kgquestn_tokens,path_file, kgpred_aliases_dir, config, topcontext=topk,stempred=stempred,degeneratex=degenerates,pred_count = pred_count)
    #if config['connectseed'] == '1':
    #G,nodeweightavg = addtoGraph(G,nodeweightavg,questn_tokens,predalias_tokens,path_file,entity_names,stempred,entcornerdict,pred_count,type_count)
    print(len(G.nodes()))
    #print(len(G.edges()))
    #print(corner_KG)
 

    #call the TEXT
    G, questn_tokens, txttrip = call_main_GRAPH_TEXT(G,topl,textspofile,textcooccur,textcontextscores,hearst_file,qtfile,topk,config,pred_count=pred_count )
    corner_TEXT= getcornerstone_TEXT(G,questn_tokens)
    print(len(G.nodes()))
    #print(len(G.edges()))
    #print(corner_TEXT)

    entyKG = getEntities(G,'KG')
    with open(we+'_KG.json', 'w') as f:
        json.dump(entyKG, f)

    entyTEXT = getEntities(G,'TEXT')
    with open(we+'_TEXT.json', 'w') as f:
        json.dump(entyTEXT, f)


    if ablate=='1':
        return topl, kgtrip, txttrip
    if not os.path.exists(alignDir):
        os.mkdir(alignDir)
    if computeAlign:
        G = createLookUpTable(qid,G,gdict,alignDir,config)
        print(len(G.edges()))
        return 
    else:
        alignEnt = True if int(config['Entity_Alignment']) == 1 else False
        alignPred = True if int(config['Predicate_Alignment']) == 1 else False
        alignType = True if int(config['Type_Alignment']) == 1 else False
        G = LookUpTable(qid,G,alignDir,alignEnt,alignPred,alignType,et,pt,tt)

    if degenerates == True:
        G = degenerate(G)
    #pass the graph to the glove to get their rep
    G=directed_to_undirected(G)
    if len(G.nodes())>0:
        G=max(nx.connected_component_subgraphs(G), key=len)

    corneritems = mergeCornerstone(corner_KG, corner_TEXT)
    #remove corner items that are not in the graph 
    corner = dict((k, v) for k, v in corneritems.items() if k in G.nodes())

    print(corner_KG)

    nx.write_gpickle(G,finalxg)
    pickle.dump(corner,open(finalcorner,'wb'))
    
    return True








if __name__ == "__main__":
    main(sys.argv)
