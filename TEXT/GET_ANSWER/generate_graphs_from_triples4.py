import re
import networkx as nx
import sys
import matplotlib.pyplot as plt
import math
import numpy as np
#import hearstPatterns
#from hearst_patterns_python.hearstPatterns.hearstPatterns import HearstPatterns
import nltk
import pickle
import requests
import json
from collections import defaultdict
from statistics import mean 
import random
#np.random.seed(123)
#random.seed(123)
#from hearstPatterns.hearstPatterns import HearstPatterns
#from hearstPatterns import HearstPatterns
#from nltk.corpus import stopwords
#sw = stopwords.words("english")
verbose=0
threshold=0.75
threshold2=0.5

threshold_align=0.5
#threshold2_align=0.6
MAX_MATCH=0

aux_list=set(['be','am','being','been','is','are','was','were','has','have','had','having','do','does','did','done','will','would','shall','should','can','could','dare','may','might', 'must','need','ought'])

stop_list=set(['a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'aren\'t', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can\'t', 'cannot', 'could', 'couldn\'t', 'did', 'didn\'t', 'do', 'does', 'doesn\'t', 'doing', 'don\'t', 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn\'t', 'has', 'hasn\'t', 'have', 'haven\'t', 'having', 'he', 'he\'d', 'he\'ll', 'he\'s', 'her', 'here', 'here\'s', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'how\'s', 'i', 'i\'d', 'i\'ll', 'i\'m', 'i\'ve', 'if', 'in', 'into', 'is', 'isn\'t', 'it', 'it\'s', 'its', 'itself', 'let\'s', 'me', 'more', 'most', 'mustn\'t', 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours	ourselves', 'out', 'over', 'own', 'same', 'shan\'t', 'she', 'she\'d', 'she\'ll', 'she\'s', 'should', 'shouldn\'t', 'so', 'some', 'such', 'than', 'that', 'that\'s', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'there\'s', 'these', 'they', 'they\'d', 'they\'ll', 'they\'re', 'they\'ve', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', 'wasn\'t', 'we', 'we\'d', 'we\'ll', 'we\'re', 'we\'ve', 'were', 'weren\'t', 'what', 'what\'s', 'when', 'when\'s', 'where', 'where\'s', 'which', 'while', 'who', 'who\'s', 'whom', 'why', 'why\'s', 'with', 'won\'t', 'would', 'wouldn\'t', 'you', 'you\'d', 'you\'ll', 'you\'re', 'you\'ve', 'your', 'yours', 'yourself', 'yourselves'])

def round(v,k): #overwriting round to get edges of different weight
	return v


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

def visualize_graph(G):
        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, pos, node_size = 500)
        nx.draw_networkx_labels(G, pos)
        nx.draw_networkx_edges(G, pos, edge_color='r', arrows=True)
        edge_labels=dict([((u,v,),d['weight'])  for u,v,d in G.edges(data=True)])
        nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels)
        plt.savefig('/GW/qa2/work/uniqorn/Crawl_Google_docs/LCQuAD_2/graphexample/train_431.png')
        plt.show()
	

def add_node_triple(G,n1,doc_id,doc_title, sent_id):
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
		G.add_node(n1,weight=0.0,matched='', did=did1, dtitle=dtitle1, sid=sid1)
		if verbose:
			print ("New ", n1,G.node[n1]	)
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


def update_edge_weight(G):
    for (n11,n22) in G.edges():
        if (n22.split(':')[1]=='Predicate') or (n11.split(':')[1]=='Predicate'):
            weights = G[n11][n22]['wlist']
            avgweights = mean(weights)
            G[n11][n22]['weight'] = avgweights
            #G[n11][n22]['wlist'] = [avgweights]
    return G

            


def add_edge_triple(G,n1,n2,d,doc_id,doc_title, sent_id): #d, doc_id doc_title, sent_id all are lists
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
                
                
                G.add_edge(n1,n2,weight=wlist,wlist=wlist,etype='Triple',did=did1,dtitle=dtitle1,sid=sid1)
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
        G.add_edge(n1,n2, weight=wlist1, wlist=wlist1, etype='Triple', did=did1, dtitle=dtitle1, sid=sid1); #print('using weight mean for edges .......................................................');
        if verbose:
            print ("New Triple Edge ",n1,n2,G.get_edge_data(n1,n2))
    return G		
	
	
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
		#	print "negative val"
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
					
def replacespace(text):
    x = re.sub(' +', ' ', text)
    return x

def build_graph_from_triple_edges2(unique_SPO_dict,Cornerstone):
    G = nx.DiGraph()
    c=0
    Cornerstone = set(Cornerstone)
    pred_count={}
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
    #print(unique_SPO_dict[0])
    #print(unique_SPO_dict.keys())
    for (nn1,nn2,nn3) in unique_SPO_dict:
        #print(nn1,nn2,nn3)
        #if bool(set((nn1,nn2,nn3)).intersection(Cornerstone))==False:
        #        continue
        
        doc_id=unique_SPO_dict[(nn1,nn2,nn3)]['doc_id']
        doc_title=unique_SPO_dict[(nn1,nn2,nn3)]['doc_title']
        sent_id=unique_SPO_dict[(nn1,nn2,nn3)]['sent_id']
        d1=unique_SPO_dict[(nn1,nn2,nn3)]['d1']
        d2=unique_SPO_dict[(nn1,nn2,nn3)]['d2']
        #print(d1)
        n1=replacespace(nn1).strip().lower()
        n2=replacespace(nn2).strip().lower()
        n3=replacespace(nn3).strip().lower()

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
            
            G=add_node_triple(G,n33,doc_id,doc_title, sent_id)
            G=add_edge_triple(G,n22,n33,d2,doc_id,doc_title, sent_id)
        
        if right_flag==1 and left_flag==0:
            #Right part of SPO already there
            if n11 not in spo_wt:
                spo_wt[n11]={}
            if n22 not in spo_wt[n11]:
                spo_wt[n11][n22]=[]
            spo_wt[n11][n22].append(d1)
            n22=n22+':'+right_index
            G=add_node_triple(G,n11,doc_id,doc_title, sent_id)
            G=add_edge_triple(G,n11,n22,d1,doc_id,doc_title, sent_id)
        
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
            
            G=add_node_triple(G,n11,doc_id,doc_title, sent_id)
            #G=add_node_triple(G,n2.lower()+":Predicate",doc_id,doc_title, sent_id)
            G=add_node_triple(G,n22,doc_id,doc_title,sent_id)
            G=add_node_triple(G,n33,doc_id,doc_title, sent_id)
            G=add_edge_triple(G,n11,n22,d1,doc_id,doc_title, sent_id)
            G=add_edge_triple(G,n22,n33,d2,doc_id,doc_title, sent_id)
    return G,cornerdict, predcornerdict

def add_nodes_weights(G,entdict,predict,meanval=False):
    for n1 in G.nodes():
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
                G.node[n1]['wlist']=[round(max_val,2)]
        elif nn1[1] == 'Predicate':
            if nn1[0] in predict:
                if meanval == True:
                    max_val = mean(predict[nn1[0]])
                elif meanval == False:
                    max_val = sum(predict[nn1[0]])
                #print("The weights tother = *********************************************************", predict[nn1[0]], " The mean = ", max_val)
                G.node[n1]['weight']=round(max_val,2)
                G.node[n1]['wlist']=[round(max_val,2)]
    return G

def getSPOtopkx2(f44,topcontext=5):
    contextscoDicts = {}
    unique_SPO_dict = dict()
    for line in f44:
        triple=line.strip().split('|')
        doc_id=triple[0]
        sent_id=triple[1]
        score=float(triple[2])
        if (doc_id,sent_id) not in contextscoDicts:
            contextscoDicts[(doc_id,sent_id)] = score
        contextlist = [(k,v) for k,v in contextscoDicts.items()]
        contextlist1=sorted(contextlist,key=lambda x:x[1],reverse=True)
        rankcontextlist,lenx = rank_list_answers(contextlist1)
        groupedcontext = group_rank(rankcontextlist, lenx)
        thetopcontexts = gettopk(groupedcontext, rank=topcontext)
        thefinaltopcontext = [xcont[0] for xcont in thetopcontexts]
    return thefinaltopcontext


def getSPOtopk(f22,f33,f44,topcontext,addcocor=False):
    contextscoDicts = {}
    allspos = list()
    unique_SPO_dict = dict()
    for line in f44:
        triple=line.strip().split('|')
        doc_id=triple[0]
        sent_id=triple[1]
        score=float(triple[2])
        if (doc_id,sent_id) not in contextscoDicts:
            contextscoDicts[(doc_id,sent_id)] = score
    contextlist = [(k,v) for k,v in contextscoDicts.items()]
    contextlist1=sorted(contextlist,key=lambda x:x[1],reverse=True)
    rankcontextlist,lenx = rank_list_answers(contextlist1)
    groupedcontext = group_rank(rankcontextlist, lenx)
    thetopcontexts = gettopk(groupedcontext, rank=topcontext)
    thefinaltopcontext = [xcont[0] for xcont in thetopcontexts]
        #print(thefinaltopcontext
    for line in f22:
        triple=line.strip().split(' | ')
        if len(triple)<8:
            print('troubled line = ',line)
        doc_id=triple[0]
        doc_title=triple[1]
        sent_id=triple[2]
        n1=replacespace(triple[3]).strip()
        d1=float(triple[4])
        n2=replacespace(triple[5]).strip()
        d2=float(triple[6])
        n3=replacespace(triple[7]).strip()
        if (doc_id,sent_id) not in thefinaltopcontext:
            #print((doc_id,sent_id),' not part of topk')
            continue
        allspos.append(line)
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
    if addcocor == False:
        #if there is no need for co-occur prediates, then just return the dict of facts here
        newSPOdict = {}
        for (n1,n2,n3) in unique_SPO_dict:
            newSPOdict[(n1,n2,n3)] = {}
            newSPOdict[(n1,n2,n3)]['d1'] = [mean(unique_SPO_dict[(n1,n2,n3)]['d1'])]
            newSPOdict[(n1,n2,n3)]['d2'] =  [mean(unique_SPO_dict[(n1,n2,n3)]['d2'])]
            newSPOdict[(n1,n2,n3)]['doc_id'] = unique_SPO_dict[(n1,n2,n3)]['doc_id']
            newSPOdict[(n1,n2,n3)]['doc_title'] = unique_SPO_dict[(n1,n2,n3)]['doc_title']
            newSPOdict[(n1,n2,n3)]['sent_id'] = unique_SPO_dict[(n1,n2,n3)]['sent_id']
        return newSPOdict,contextscoDicts,len(allspos)
    #if the cocor is set to True, then co-occur will be evaluated also
    for line in f33:
        triple=line.strip().split(' | ')
        if len(triple) < 8:
            print(line, '^M' in line)
        doc_id=triple[0]
        doc_title=triple[1]
        sent_id=triple[2]
        n1=replacespace(triple[3]).strip()
        d1=float(triple[4])
        n2=replacespace(triple[5]).strip()
        d2=float(triple[6])
        n3=replacespace(triple[7]).strip()
        if (doc_id,sent_id) not in thefinaltopcontext:
            #print((doc_id,sent_id),' not part of topk')
            continue
        allspos.append(line)
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
    return newSPOdict,contextscoDicts,len(allspos)

def getSPO(f22,f33,addcocor = False):
    contextscoDicts = {}
    unique_SPO_dict = dict()
    for line in f22:
        triple=line.strip().split(' | ')
        doc_id=triple[0]
        doc_title=triple[1]
        sent_id=triple[2]
        n1=replacespace(triple[3]).strip()
        d1=float(triple[4])
        n2=replacespace(triple[5]).strip()
        d2=float(triple[6])
        n3=replacespace(triple[7]).strip()
        if (doc_id,sent_id) not in contextscoDicts:
            contextscoDicts[(doc_id,sent_id)] = d2
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
    if addcocor == False:
        #if there is no need for co-occur prediates, then just return the dict of facts here
        newSPOdict = {}
        for (n1,n2,n3) in unique_SPO_dict:
            newSPOdict[(n1,n2,n3)] = {}
            newSPOdict[(n1,n2,n3)]['d1'] = [mean(unique_SPO_dict[(n1,n2,n3)]['d1'])]
            newSPOdict[(n1,n2,n3)]['d2'] =  [mean(unique_SPO_dict[(n1,n2,n3)]['d2'])]
            newSPOdict[(n1,n2,n3)]['doc_id'] = unique_SPO_dict[(n1,n2,n3)]['doc_id']
            newSPOdict[(n1,n2,n3)]['doc_title'] = unique_SPO_dict[(n1,n2,n3)]['doc_title']
            newSPOdict[(n1,n2,n3)]['sent_id'] = unique_SPO_dict[(n1,n2,n3)]['sent_id']
        return newSPOdict,contextscoDicts
    #if the cocor is set to True, then co-occur will be evaluated also   
    ##to check if mean will do
    newSPOdict = {}
    for line in f33:
        triple=line.strip().split(' | ')
        doc_id=triple[0]
        doc_title=triple[1]
        sent_id=triple[2]
        n1=replacespace(triple[3]).strip()
        d1=float(triple[4])
        n2=replacespace(triple[5]).strip()
        d2=float(triple[6])
        n3=replacespace(triple[7]).strip()
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
    return newSPOdict,contextscoDicts

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

def getcornerstone(G,questn_tokens):
    corner = {}
    count = 0
    for  n in G.nodes():
        nn2 = n.split(':')
        if not (nn2[1]=='Entity' or nn2[1]=='Predicate' or nn2[1]=='Type'):
            continue
        #print(' the nodes are -------> ', n)
        nsplit = nn2[0]
        wordsplit = set(n.split())
        wordsplit = set(nsplit.split())
        if bool(wordsplit.intersection(set(questn_tokens))):
            #print(n, ' is a cornerstone')
            count = count + 1
            inter = list(wordsplit.intersection(set(questn_tokens)))
            corner[n] = ' | '.join(inter)#inter[-1]
    return corner








def read_glove(G,q_ent,gdict,option='WORD2VEC'):
	veclen=300
	g_pred={}
	g_ent={}
	g_type={}
	g_ques={}
	done=set(); veclen = 100 if option=='WIKI' else 300
	
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
					#	avec=np.add(avec,gdict[el])
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
					#	avec=np.add(avec,gdict[el])
					#else:
					avec=np.add(avec,np.array(gdict[el]))	
					c+=1.0
			if c>0:
				avec=np.divide(avec, c)
				g_ques[n]=avec.tolist()
				
							
	return 	g_pred,g_ent,g_type,g_ques	

	
	
def add_type_alignment_edges(G,g_type,gdict):
    xdict = defaultdict()
    if MAX_MATCH==0:
        for n1 in G.nodes():
            nn1=n1.split(':')
            if nn1[1]=='Type' and n1 in g_type:
                for n2 in G.nodes():
                    nn2=n2.split(':')
                    if n2!=n1 and nn2[1]=='Type' and n2 in g_type:
                        value=cosine_similarity(g_type[n1],g_type[n2])
                        #print("n1 = ",n1, " n2 = ", n2, "Jack = ", value)
                        #xdict[tuple(sorted((nn1[0],nn2[0])))] = value
                        if value>=threshold_align:
                            xdict[tuple(sorted((nn1[0],nn2[0])))] = value
                            value=round(value,2)
                            #Add bidirectional alignment edges between the predicates; no associated doc with edge
                            G.add_edge(n1,n2,weight=value,wlist=[value],etype='Type_Alignment',did=[],dtitle=[],sid=[])
                            G.add_edge(n2,n1,weight=value,wlist=[value],etype='Type_Alignment',did=[],dtitle=[],sid=[])
                            if verbose:
                                print ("Type Alignment Edge ",n1,n2,G.get_edge_data(n1,n2))
                                print ("type Alignment Edge ",n2,n1,G.get_edge_data(n2,n1))
    else:
        for n1 in G.nodes():
            nn1=n1.split(':')
            if nn1[1]=='Type':
                for n2 in G.nodes():
                    nn2=n2.split(':')
                    if n2!=n1 and nn2[1]=='Type':
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
                    if n2!=n1 and nn2[1]=='Type' and tuple(sorted((nn1[0],nn2[0]))) in xdict:
                        value = xdict[tuple(sorted((nn1[0],nn2[0])))] 
                        if value>=tt:
                            value=round(value,2)
                            #print('type alignment = ',value)
                            #Add bidirectional alignment edges between the predicates; no associated doc with edge
                            G.add_edge(n1,n2,weight=value,wlist=[value],etype='Type_Alignment',did=[],dtitle=[],sid=[])
                            G.add_edge(n2,n1,weight=value,wlist=[value],etype='Type_Alignment',did=[],dtitle=[],sid=[])
                            if verbose:
                                print ("Type Alignment Edge ",n1,n2,G.get_edge_data(n1,n2))
                                print ("type Alignment Edge ",n2,n1,G.get_edge_data(n2,n1))
    else:
        for n1 in G.nodes():
            nn1=n1.split(':')
            if nn1[1]=='Type':
                for n2 in G.nodes():
                    nn2=n2.split(':')
                    if n2!=n1 and nn2[1]=='Type' and tuple(sorted((nn1[0],nn2[0]))) in xdict:
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
			
			
def add_predicate_alignment_edges(G,g_pred,gdict):
    xdict = defaultdict()
    if MAX_MATCH==0:
        for n1 in G.nodes():
            nn1=n1.split(':')
            if nn1[1]=='Predicate' and n1 in g_pred:
                for n2 in G.nodes():
                    nn2=n2.split(':')
                    if n2!=n1 and nn2[1]=='Predicate' and n2 in g_pred:
                        value=cosine_similarity(g_pred[n1],g_pred[n2])
                        #print("n1 = ",n1, " n2 = ", n2, "Jack = ", value)
                        #xdict[tuple(sorted((nn1[0],nn2[0])))] = value
                        if value>=threshold_align:
                            xdict[tuple(sorted((nn1[0],nn2[0])))] = value
                            value=round(value,2)
                            #Add bidirectional alignment edges between the predicates; no associated doc with edge
                            G.add_edge(n1,n2,weight=value,wlist=[value],etype='Predicate_Alignment',did=[],dtitle=[],sid=[])
                            G.add_edge(n2,n1,weight=value,wlist=[value],etype='Predicate_Alignment',did=[],dtitle=[],sid=[])
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
                            G.add_edge(n1,n2,weight=value,wlist=[value],etype='Predicate_Alignment',did=[],dtitle=[],sid=[])
                            G.add_edge(n2,n1,weight=value,wlist=[value],etype='Predicate_Alignment',did=[],dtitle=[],sid=[])
                            if verbose:
                                print ("Predicate Alignment Edge ",n1,n2,G.get_edge_data(n1,n2))
                                print ("Predicate Alignment Edge ",n2,n1,G.get_edge_data(n2,n1)	)						
    return G, xdict

def add_predicate_alignment_edges2(G,xdict,pt):
    #print(xdict)
    if MAX_MATCH==0:
        for n1 in G.nodes():
            nn1=n1.split(':')
            if nn1[1]=='Predicate' :
                for n2 in G.nodes():
                    nn2=n2.split(':')
                    if n2!=n1 and nn2[1]=='Predicate' and tuple(sorted((nn1[0],nn2[0]))) in xdict:
                        #print(n1,n2)
                        value = xdict[tuple(sorted((nn1[0],nn2[0])))] 
                        if value>=pt:
                            value=round(value,2)
                            #print(nn1[0], nn2[0],'predicate alignment = ',value)
                            #Add bidirectional alignment edges between the predicates; no associated doc with edge
                            G.add_edge(n1,n2,weight=value,wlist=[value],etype='Predicate_Alignment',did=[],dtitle=[],sid=[])
                            G.add_edge(n2,n1,weight=value,wlist=[value],etype='Predicate_Alignment',did=[],dtitle=[],sid=[])
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
                            G.add_edge(n1,n2,weight=value,wlist=[value],etype='Predicate_Alignment',did=[],dtitle=[],sid=[])
                            G.add_edge(n2,n1,weight=value,wlist=[value],etype='Predicate_Alignment',did=[],dtitle=[],sid=[])
                            if verbose:
                                print ("Predicate Alignment Edge ",n1,n2,G.get_edge_data(n1,n2))
                                print ("Predicate Alignment Edge ",n2,n1,G.get_edge_data(n2,n1) )
    return G

def replace_symbols(s):
        s=s.replace('(',' ')
        s=s.replace(')',' ')
        s=s.replace('[',' ')
        s=s.replace(']',' ')
        s=s.replace('{',' ')
        s=s.replace('}',' ')
        s=s.replace('|',' ')
        s=s.replace('"',' ')
        s=s.replace('\'',' ')
        s=s.replace('\n',' ')   
        #s=s.replace(',','')
        #s=s.replace('others','other ones')
        #s=s.replace('Others','Other ones')
        s=s.strip(',')
        s=s.strip()
        s=s+'.'
        return s
									
def get_hearst_patterns_from_context(f22):
	'''
	s=""
	for l in f22:
		l=(l.strip()).split('\t')
		if l[0].startswith('sent-') and len(l)>1:
			s=s+replace_symbols(l[2])+' '
			
	h = HearstPatterns(extended = True)
	HP=h.find_hyponyms(s.decode('utf-8'))
	
        '''
	h = HearstPatterns(extended=True)
	HP=set()
	lc=0
	for l in f22:
		l=(l.strip()).split('\t')
		#print "line ",lc,l
		if l[0].startswith('sent-') and len(l)>1:
			s=replace_symbols(l[2])
			#s=l[1]#.strip()
                        #print "line ",lc,s.decode('utf-8')
			HP1=set(h.find_hyponyms(s.decode('utf-8')))
			HP=HP.union(HP1)
		lc+=1	
	
        #HP=[]
	#if verbose:
	print ("HP are ",HP)
	return HP
	
def get_wiki_categories_from_context(f22,Wiki_Threshold):
        wiki_cat={}
        for l in f22:
                l=(l.strip()).split('\t')
                #print "line ",lc,l
                if l[0].startswith('sent-') and len(l)>1:
                        s=replace_symbols(l[2])
                        s=s.replace('%','') #Tagme cannot process
                        s=s.rstrip('.')
			#s=l[1]#.strip()
                        #print "line ",lc,s.decode('utf-8')
                        req_string='https://tagme.d4science.org/tagme/tag?lang=en&include_abstract=true&include_categories=true&gcube-token=9dc5f6c0-3040-411b-9687-75ca53249072-843339462&text='+s#.encode('utf-8')
                        try:
                                r = requests.get(req_string)
                                wiki=r.json()
                                annotations=wiki['annotations']
                                #print "Annotations ",wiki,annotations
                                for doc in annotations:
                                        if doc['rho']>=Wiki_Threshold:
                                                wiki_cat[doc['spot']]=doc['dbpedia_categories']
                                                #print "Wiki added ",doc['rho'],doc['spot'], doc['dbpedia_categories']
                        except:
                                print ("wiki prob-> ") #,s.encode('utf-8'))		    
        #HP=[]
        #if verbose:
        #print "Wiki Categories ",wiki_cat
        return wiki_cat		
	
def remove_leading_article(t): #remove leading articles from type nodes detected by hearst algo
	t=t.split()
	s=''
	articles=['a','an','the','some','one','few']
	i=0
	for i in range(0,len(t)):
		if t[i] not in articles:
			break
	#while t[i] in articles:
	#	i+=1
	if len(t)>0:
		s=t[i]	
		for j in range(i+1,len(t)):
			s+=' '+t[j]	
	else:
		print ('PROBLEM leading article ',t)			
	return s					

def remove_type_edges(G,HP):
    alltypes = set()
    for pat in HP:
        #print(pat)
        p1=(pat[0]).lower()+':Entity'
        ##if p1 in G.nodes():
        ##print(p1)
        p2=(remove_leading_article(pat[1])).lower()
        p2+=':Type'
        #f G.has_edge(p1,p2):
        #remove the edge
        #G.remove_edge(p1,p2)
        alltypes.add(p2)
    for item in alltypes:
        if item in G.nodes():
            #print("Type is present ooooo")
            G.remove_node(item)
    return G

def degenerate(G):
    #print("is it direted = ", nx.is_directed(G))
    for u,v,d in G.edges(data=True):
        d['wlist'] = [0.0]
        d['weight'] = 0
    '''
    wlist = []
    nx.set_edge_attributes(G,values = 0.0, name = 'weight')
    nx.set_edge_attributes(G,values = wlist, name = 'wlist')
    wlist.append(0.0)
    '''
    return G

def add_type_edges(G,HP,topsnipets):
    countmatch = 0
    for pat in HP:
        snipet = (pat[2],pat[3])
        #print(snipet)
        #print(topsnipets)
        if snipet not in topsnipets:
            #if the hearst is not part of the top snipets
            continue
        score = topsnipets[snipet]
        p1=replacespace(pat[0]).strip().lower()+':Entity'
        if p1 in G.nodes():
            countmatch += 1
            #print(p1, '   ', pat)
            p2=remove_leading_article(replacespace(pat[1]).strip()).lower()
            #print ("Removed articles ",p2)
            #if p2+':Entity' in G.nodes():
            #       p2+=':Entity'
            #else:
            p2+=':Type' #creating type nodes irrespective of same name entity node exists or not
            if p2 not in G.nodes():#type node does not exist
                G=add_node_triple(G, p2, [], [], []) #Adding a "Type" node; not using typical nodes with only labeled "type"
                G.add_edge(p1, p2, weight=score, wlist=[score], etype='Type',did=[pat[2]],dtitle=[],sid=[pat[3]])
            else:
                if (p1,p2) not in G.edges():
                    G.add_edge(p1, p2, weight=score, wlist=[score], etype='Type',did=[pat[2]],dtitle=[],sid=[pat[3]]) #Assuming there do not exist other edges between entity and "Type" nodes
                elif (p1,p2) not in G.edges():
                    data=G.get_edge_data(p1,p2)
                    wlit=data['wlist'] #get the cuurent edge weight and append to it
                    wlit.append(score)
                    did = data['did']
                    did.append(pat[2])
                    sid = data['sid']
                    sid.append(pat[3])
                    G.add_edge(p1, p2, weight=score, wlist=wlit, etype='Type',did=did,dtitle=[],sid=sid) #Assuming there do not exist other edges between entity and "Type" nodes
                    

    #print("Entity match = ", countmatch, ' whole hearst = ',len(HP))
    return G




def add_NE_type_edges(G,NE_types):
	for pat in NE_types:
		p1=(pat[0].encode('utf-8')).lower()+':Entity'
		if p1 in G.nodes():
			p2=(pat[1]).lower()
			print ("NE_Type ",p1,p2)
			#if p2+':Entity' in G.nodes():
			#	p2+=':Entity'
			#else:
			p2+=':Type' #creating type nodes irrespective of same name entity node exists or not
			if p2 not in G.nodes():
				G=add_node_triple(G, p2, [], [], []) #Adding a "Type" node; not using typical nodes with only labeled "type"
				G.add_edge(p1, p2, weight=1.0, wlist=[1.0], etype='Type',did=[],dtitle=[],sid=[])		
			else:
				if (p1,p2) not in G.edges():
					G.add_edge(p1, p2, weight=1.0, wlist=[1.0], etype='Type',did=[],dtitle=[],sid=[]) #Assuming there do not exist other edges between entity and "Type" nodes  
			
	return G

def add_wiki_type_edges(G,wiki_cat):
	for spot in wiki_cat:
		p1=(spot).lower()+':Entity'
		if p1 in G.nodes():
			for cat in wiki_cat[spot]:
				p2=(remove_leading_article(cat)).lower()
				#print "Removed articles ",p2
				#if p2+':Entity' in G.nodes():
				#	p2+=':Entity'
				#else:
				p2+=':Type' #creating type nodes irrespective of same name entity node exists or not
				if p2 not in G.nodes():
					G=add_node_triple(G, p2, [], [], []) #Adding a "Type" node; not using typical nodes with only labeled "type"
					G.add_edge(p1, p2, weight=1.0, wlist=[1.0], etype='Type',did=[],dtitle=[],sid=[])		
				else:
					if (p1,p2) not in G.edges():
						G.add_edge(p1, p2, weight=1.0, wlist=[1.0], etype='Type',did=[],dtitle=[],sid=[]) #Assuming there do not exist other edges between entity and "Type" nodes  
			
	return G	

'''
def add_entity_node_weights(G,q_ent,mention_dict):
	for n1 in G.nodes():
		nn1=n1.split(':')
		if nn1[1]=='Entity' and nn1[0] in mention_dict:
			max_val=0.0
			max_word=""
			for w in q_ent:
				if w in mention_dict:
					inter=len(mention_dict[nn1[0]].intersection(mention_dict[w]))
					uni=len(mention_dict[nn1[0]].union(mention_dict[w]))
					jack=float(inter)/float(uni)
					if jack>max_val:
						max_val=jack
						max_word=w
			#if max_val>=threshold:
			G.node[n1]['weight']=round(max_val,2)
			G.node[n1]['matched']=max_word
	return G					
			
			
def add_predicate_type_node_weights(G,g_pred,g_type,g_dict,question):
	for n1 in G.nodes():
		nn1=n1.split(':')
		if nn1[1]=='Entity':
			continue 
			
		if nn1[1]=='Predicate' and n1 in g_pred:
			avec=g_pred[n1]
		else:
			if n1 in g_type:
				avec=g_type[n1]
			
		max_val=0.0
		max_word=""
		for w in question:
			if w in g_dict:
				bvec=g_dict[w].tolist()
				value=cosine_similarity(avec,bvec)				
				if value>max_val:
					max_val=value
					max_word=w
		#if max_val>=threshold:
		G.node[n1]['weight']=round(max_val,2)
		G.node[n1]['matched']=max_word
		
	return G						
'''

	
def add_node_weights(G,g_ent,g_pred,g_type,gdict,g_ques,q_ent,mention_dict, type_qent,Cornerstone_Matching):
	if verbose:
		print ("\n\nQuestion entities->",q_ent)
	for qe in g_ent:
		if verbose:
			print (qe)
	veclen=300
	for n1 in G.nodes():
                nn1=n1.split(':')
                if nn1[1]=='Entity':
                        if nn1[0] in q_ent and (type_qent[nn1[0]]=='S' or type_qent[nn1[0]]=='NE' or Cornerstone_Matching==0) :
                                max_val=1.0
                                max_word=nn1[0]
                                G.node[n1]['weight']=round(max_val,2)
                                G.node[n1]['matched']=max_word
                                if verbose:
                                        print ("\nExact match entity, weight",n1, max_val, max_word)
                        else:	
                                max_val=0.0 
                                max_word=""
                                for w in q_ent:
                                        if type_qent[w]=='S' or type_qent[w]=='NE' or Cornerstone_Matching==0:	
                                                if False:#nn1[0] in mention_dict and w in mention_dict:
                                                        inter=len(mention_dict[nn1[0]].intersection(mention_dict[w]))
                                                        uni=len(mention_dict[nn1[0]].union(mention_dict[w]))
                                                        if uni>0:
                                                                jack=float(inter)/float(uni)
                                                        else:
                                                                jack=0.0
                                                else:
                                                        part1=nn1[0].replace(',',' ')
                                                        part1=part1.replace('-',' ')
                                                        wn1=set(part1.split())
                                                        part2=w.replace(',',' ')
                                                        part2=part2.replace('-',' ')
                                                        wn2=set(part2.split())                              
                                                        inter=len(wn1.intersection(wn2))
                                                        uni=len(wn1.union(wn2))
                                                        if uni>0:
                                                                jack=float(inter)/float(uni)
                                                        else:
                                                                jack=0.0
                                                if jack>max_val:
                                                        max_val=jack
                                                        max_word=w
                                #if max_val>=threshold2:
                                G.node[n1]['weight']=round(max_val,2)
                                G.node[n1]['matched']=max_word
                                if verbose:
                                        print ("\nDICT Entity, match, weight",n1, max_val, max_word)		
                else:
                        if nn1[0] in q_ent and (type_qent[nn1[0]]=='P' or Cornerstone_Matching==0):
                                max_val=1.0
                                max_word=nn1[0]
                                G.node[n1]['weight']=round(max_val,2)
                                G.node[n1]['matched']=max_word
                                if verbose:
                                        print ("\nExact match predicate, weight",n1, max_val, max_word)
                        else:	
                                if MAX_MATCH==0:
                                        avec=np.zeros(veclen)
                                        update=0
                                        if n1 in g_pred:# and nn1[1]=='Predicate':
                                                avec=g_pred[n1]
                                                update=1
                                        else:
                                                if n1 in g_type:
                                                        avec=g_type[n1]
                                                        update=1
                                        if update==0:
                                                print ("\n\nNO VECTOR FOR ->",n1.encode('utf-8')	)			
                                        max_val=0.0
                                        max_word=""
                                        for w in q_ent:
                                                if w in g_ques and (type_qent[w]=='P' or Cornerstone_Matching==0):
                                                        bvec=g_ques[w]
                                                        value=cosine_similarity(avec,bvec)
                                                        if value>max_val:
                                                                max_val=value
                                                                max_word=w
					#if max_val>=threshold:
                                        G.node[n1]['weight']=round(max_val,2)
                                        G.node[n1]['matched']=max_word
                                        if verbose:
                                                print ("\nPredicate, match, weight",n1, max_val, max_word)
                                else:
                                        max_val=0.0
                                        max_word=""
                                        for w in q_ent:
                                                if type_qent[w]=='P' or Cornerstone_Matching==0:
                                                        value=cosine_similarity_MAX_MATCH(nn1[0],w,gdict)
                                                        if value>max_val:
                                                                max_val=value
                                                                max_word=w
                                        #if max_val>=threshold:
                                        G.node[n1]['weight']=round(max_val,2)
                                        G.node[n1]['matched']=max_word	
                                        if verbose:
                                                print ("\nPredicate, match, weight",n1, max_val, max_word)			
			
	return G		

def add_entity_alignment_edges(G,mention_dict,g_ent):
    xdict = defaultdict()
    for n1 in G.nodes():
        nn1=n1.split(':')
        if nn1[1]=='Entity':
            for n2 in G.nodes():
                nn2=n2.split(':')
                if n2!=n1 and nn2[1]=='Entity':
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
                        xdict[tuple(sorted((nn1[0],nn2[0])))] = jack
                        value=round(jack,2)
                        #Add bidirectional alignment edges between the predicates; no associated doc with edge
                        G.add_edge(n1,n2,weight=value,wlist=[value],etype='Entity_Alignment',did=[],dtitle=[],sid=[])
                        G.add_edge(n2,n1,weight=value,wlist=[value],etype='Entity_Alignment',did=[],dtitle=[],sid=[])
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
                if n2!=n1 and nn2[1]=='Entity' and tuple(sorted((nn1[0],nn2[0]))) in lookup:
                    jack = lookup[tuple(sorted((nn1[0],nn2[0])))] 
                    #print("n1 = ",n1, " n2 = ", n2, "Jack = ", jack, round(jack,2))
                    #if jack>0: print("******************************************************************************************************************************")
                    if jack>=et:
                        #print('entity align without round = ',jack)
                        value=round(jack,2)
                        #print("entity alignment ", value)
                        #print('n1 = ', nn1[0], 'n2 = ',nn2[0], value)
                        #Add bidirectional alignment edges between the predicates; no associated doc with edge
                        G.add_edge(n1,n2,weight=value,wlist=[value],etype='Entity_Alignment',did=[],dtitle=[],sid=[])
                        G.add_edge(n2,n1,weight=value,wlist=[value],etype='Entity_Alignment',did=[],dtitle=[],sid=[])
                        if verbose:
                            print ("Entity Alignment Edge ",n1,n2,G.get_edge_data(n1,n2))
                            print ("Entity Alignment Edge ",n2,n1,G.get_edge_data(n2,n1))
                            #write the deafult dict to file
    return G






def directed_to_undirected(G1):
        G=nx.Graph()
        for n in G1.nodes():
                G.add_node(n,weight=G1.node[n]['weight'],matched=G1.node[n]['matched'],did=G1.node[n]['did'],dtitle=G1.node[n]['dtitle'],sid=G1.node[n]['sid'])
        done=set()
        elist=[]
        for (n1,n2) in G1.edges():
                if (n1,n2) not in done:
                        done.add((n1,n2))
                        done.add((n2,n1))
                        data=G1.get_edge_data(n1,n2)
                        d=data['weight']
                        wlist1=data['wlist']
                        did1=data['did']
                        dtitle1=data['dtitle']
                        sid1=data['sid']
                        etype1=data['etype']
                        #print "n1,n2 ->",data
                        if (n2,n1) in G1.edges():
                                data1=G1.get_edge_data(n2,n1)
				#print "n2,n1 ->",data1
                                if data1['etype']=='Triple':
                                        if data1['weight']>d: #Keeping maximum weight edge
                                                d=data1['weight']
                                        for w in data1['wlist']:
                                                #print('Data weight = ',data1['wlist'])#print "wlist",w,wlist1
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


def distribute_node_weights(G):
	#Distribute node weights into edges
        for (n1,n2) in G.edges():
            w1=float(G.node[n1]['weight'])
            d1=float(G.degree(n1))
            w2=float(G.node[n2]['weight'])
            d2=float(G.degree(n2))
            data=G.get_edge_data(n1,n2)
            #print 'Current edge weight ',n1,n2,data['weight'],w1,d1,w2,d2
            if data['weight']<0.0:
            	print ("Current weight negative")
            #data['weight']+=w1/d1+w2/d2
            data['weight']+=w1/2.0+w2/2.0           
            if data['weight']>1.0:
                data['weight']=1.0
            #data['wlist'].append(w1/d1)
            #data['wlist'].append(w2/d2)
            data['wlist'].append(w1/2.0)
            data['wlist'].append(w2/2.0)
            data=G.get_edge_data(n1,n2)
            #print 'Updated edge weight ',n1,n2,data['weight']
            if data['weight']<0.0:
                print ("Updated weight negative")
        return G   
		
	
	
if __name__ == "__main__":
    #print(round(2.7872229222,3))
    main(sys.argv)			
			
