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
from statistics import mean 
#from hearstPatterns.hearstPatterns import HearstPatterns
#from hearstPatterns import HearstPatterns
from nltk.corpus import stopwords
sw = stopwords.words("english")
from nltk.stem import PorterStemmer
porter = PorterStemmer()
verbose=0
threshold=0.75
threshold2=0.5
MAX_MATCH=1

aux_list=set(['be','am','being','been','is','are','was','were','has','have','had','having','do','does','did','done','will','would','shall','should','can','could','dare','may','might', 'must','need','ought'])

stop_list=set(['a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'aren\'t', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can\'t', 'cannot', 'could', 'couldn\'t', 'did', 'didn\'t', 'do', 'does', 'doesn\'t', 'doing', 'don\'t', 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn\'t', 'has', 'hasn\'t', 'have', 'haven\'t', 'having', 'he', 'he\'d', 'he\'ll', 'he\'s', 'her', 'here', 'here\'s', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'how\'s', 'i', 'i\'d', 'i\'ll', 'i\'m', 'i\'ve', 'if', 'in', 'into', 'is', 'isn\'t', 'it', 'it\'s', 'its', 'itself', 'let\'s', 'me', 'more', 'most', 'mustn\'t', 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours	ourselves', 'out', 'over', 'own', 'same', 'shan\'t', 'she', 'she\'d', 'she\'ll', 'she\'s', 'should', 'shouldn\'t', 'so', 'some', 'such', 'than', 'that', 'that\'s', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'there\'s', 'these', 'they', 'they\'d', 'they\'ll', 'they\'re', 'they\'ve', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', 'wasn\'t', 'we', 'we\'d', 'we\'ll', 'we\'re', 'we\'ve', 'were', 'weren\'t', 'what', 'what\'s', 'when', 'when\'s', 'where', 'where\'s', 'which', 'while', 'who', 'who\'s', 'whom', 'why', 'why\'s', 'with', 'won\'t', 'would', 'wouldn\'t', 'you', 'you\'d', 'you\'ll', 'you\'re', 'you\'ve', 'your', 'yours', 'yourself', 'yourselves'])

def round(v,k): #overwriting round to get edges of different weight
	return v
	
def visualize_graph(G):
	pos = nx.spring_layout(G)
	nx.draw_networkx_nodes(G, pos, node_size = 500)
	nx.draw_networkx_labels(G, pos)
	nx.draw_networkx_edges(G, pos, edge_color='r', arrows=True)
	
	edge_labels=dict([((u,v,),d['weight'])
                 for u,v,d in G.edges(data=True)])
	nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels)
	
	plt.show()
	
	
	
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


def add_edge_triple(G,n1,n2,d):
    wlist1=[d]
    if (n1,n2) in G.edges():
        for (n11,n22) in G.edges():
            if (n2.split(':')[1]=='Predicate' and n22.split(':')[1]=='Predicate' and n2.split(':')[0]==n22.split(':')[0] and n1==n11) or (n1.split(':')[1]=='Predicate' and n11.split(':')[1]=='Predicate' and n1.split(':')[0]==n11.split(':')[0] and n2==n22):
                data=G.get_edge_data(n11,n22)
                #d=data['weight']
                wlist=data['wlist']
                wlist.extend(wlist1)
                G.add_edge(n1,n2, weight=wlist, wlist=wlist, etype='Triple')
    else:
        G.add_edge(n1,n2, weight=wlist1, wlist=wlist1, etype='Triple')
    return G


def update_edge_weight(G):
    for (n11,n22) in G.edges():
        if (n22.split(':')[1]=='Predicate') or (n11.split(':')[1]=='Predicate'):
            weights = G[n11][n22]['wlist']
            avgweights = mean(weights)
            G[n11][n22]['weight'] = avgweights
            G[n11][n22]['wlist'] = [avgweights]
    return G

def build_graph_from_triple_edges2(G,unique_SPO_dict,questn_token, predalias_tokens, entity_names,stempred,entcornerdict,pred_count,type_count):
    c = 0
    
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
                    G.add_node(n11,weight=unique_SPO_dict[(nn1,nn2,nn3)]['score_n1'],matched=unique_SPO_dict[(nn1,nn2,nn3)]['matched_n1'])
                G.add_node(n33,weight=0.0,matched='')
                G.add_edge(n11,n33, weight=0.0, wlist=[0.0], etype='Type')
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
                G.add_node(n11,weight=0.0,matched=unique_SPO_dict[(nn1,nn2,nn3)]['matched_n1'])
            G.add_node(n22,weight=0.0,matched='')
            if n33 not in G.nodes():
                G.add_node(n33,weight=0.0,matched=unique_SPO_dict[(nn1,nn2,nn3)]['matched_n3'])
            #G.add_edge(n11,n22, weight=0.0, wlist=[0.0], etype='Triple')
            #G.add_edge(n22,n33, weight=0.0, wlist=[0.0], etype='Triple')
            add_edge_triple(G,n11,n22,d1)
            add_edge_triple(G,n22,n33,d1)

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
                if n2split.intersection(Cornerstone):# or Cornerstone.intersection(predalias_tokens[n2.lower()]):
                    if n22 not in entcornerdict:
                        entcornerdict[n22] = []
                    entcornerdict[n22].append(d2)
                elif n2.lower() in predalias_tokens:
                    if Cornerstone.intersection(predalias_tokens[n2.lower()]):
                        if n22 not in entcornerdict:
                            entcornerdict[n22] = []
                        entcornerdict[n22].append(d2)



            if n11 not in G.nodes():
                G.add_node(n11,weight=0.0,matched=unique_SPO_dict[(nn1,nn2,nn3)]['matched_n1'])
            G.add_node(n22,weight=0.0,matched='')
            if n33 not in G.nodes():
                G.add_node(n33,weight=0.0,matched=unique_SPO_dict[(nn1,nn2,nn3)]['matched_n3'])
            #G.add_edge(n11,n22, weight=0.0, wlist=[0.0], etype='Triple')
            #G.add_edge(n22,n33, weight=0.0, wlist=[0.0], etype='Triple')
            add_edge_triple(G,n11,n22,d1)
            add_edge_triple(G,n22,n33,d1)
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

                G.add_node(qn22,weight=0.0,matched='')
                if qn33 not in G.nodes():
                    G.add_node(qn33,weight=0.0,matched=unique_SPO_dict[(nn1,nn2,nn3)]['matched_qual'][qualct])
                #G.add_edge(n22,qn22, weight=0.0, wlist=[0.0], etype='Triple')
                #G.add_edge(qn22,qn33, weight=0.0, wlist=[0.0], etype='Triple')
                add_edge_triple(G,n22,qn22,d1)
                add_edge_triple(G,qn22,qn33,d1)
        c+=1

    nodeweightavg = dict((k, mean(v)) for k, v in entcornerdict.items())
    #print("node scores", nodeweightavg )

    #print(G.nodes())

    return G,nodeweightavg

def build_graph_from_triple_edges(unique_SPO_dict,questn_token, predalias_tokens, entity_names,stempred):
    G = nx.DiGraph()
    c=0
    pred_count={}
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
                    G.add_node(n11,weight=unique_SPO_dict[(nn1,nn2,nn3)]['score_n1'],matched=unique_SPO_dict[(nn1,nn2,nn3)]['matched_n1'])
                G.add_node(n33,weight=0.0,matched='')
                G.add_edge(n11,n33, weight=0.0, wlist=[0.0], etype='Type')
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
                G.add_node(n11,weight=0.0,matched=unique_SPO_dict[(nn1,nn2,nn3)]['matched_n1'])
            G.add_node(n22,weight=0.0,matched='')
            if n33 not in G.nodes():
                G.add_node(n33,weight=0.0,matched=unique_SPO_dict[(nn1,nn2,nn3)]['matched_n3'])
            #G.add_edge(n11,n22, weight=0.0, wlist=[0.0], etype='Triple')
            #G.add_edge(n22,n33, weight=0.0, wlist=[0.0], etype='Triple')
            add_edge_triple(G,n11,n22,d1)
            add_edge_triple(G,n22,n33,d1)

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
                G.add_node(n11,weight=0.0,matched=unique_SPO_dict[(nn1,nn2,nn3)]['matched_n1'])    
            G.add_node(n22,weight=0.0,matched='')
            if n33 not in G.nodes():
                G.add_node(n33,weight=0.0,matched=unique_SPO_dict[(nn1,nn2,nn3)]['matched_n3'])
            #G.add_edge(n11,n22, weight=0.0, wlist=[0.0], etype='Triple')
            #G.add_edge(n22,n33, weight=0.0, wlist=[0.0], etype='Triple')
            add_edge_triple(G,n11,n22,d1)
            add_edge_triple(G,n22,n33,d1)
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

                G.add_node(qn22,weight=0.0,matched='')
                if qn33 not in G.nodes():
                    G.add_node(qn33,weight=0.0,matched=unique_SPO_dict[(nn1,nn2,nn3)]['matched_qual'][qualct])
                #G.add_edge(n22,qn22, weight=0.0, wlist=[0.0], etype='Triple')
                #G.add_edge(qn22,qn33, weight=0.0, wlist=[0.0], etype='Triple')
                add_edge_triple(G,n22,qn22,d1)
                add_edge_triple(G,qn22,qn33,d1)
        c+=1

    nodeweightavg = dict((k, mean(v)) for k, v in entcornerdict.items())
    #print("node scores", nodeweightavg )

    #print(G.nodes())

    return G,nodeweightavg,entcornerdict,pred_count,type_count


def read_glove(G,q_ent,gdict):
	veclen=300
	g_pred={}
	g_ent={}
	g_type={}
	g_ques={}
	done=set()
	
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
				
				if nn[1]=='Predicate':
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
	if MAX_MATCH==0:
		for n1 in G.nodes():
			nn1=n1.split(':')
			if nn1[1]=='Type' and n1 in g_type:
				for n2 in G.nodes():
					nn2=n2.split(':')
					if n2!=n1 and nn2[1]=='Type' and n2 in g_type:
						value=cosine_similarity(g_type[n1],g_type[n2])
						if value>=threshold:
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
						if value>=threshold:
							value=round(value,2)
							#Add bidirectional alignment edges between the predicates; no associated doc with edge
							G.add_edge(n1,n2,weight=value,wlist=[value],etype='Type_Alignment',did=[],dtitle=[],sid=[])
							G.add_edge(n2,n1,weight=value,wlist=[value],etype='Type_Alignment',did=[],dtitle=[],sid=[])
							if verbose:
								print ("Type Alignment Edge ",n1,n2,G.get_edge_data(n1,n2))
								print ("Type Alignment Edge ",n2,n1,G.get_edge_data(n2,n1))							
	return G
			
			
def add_predicate_alignment_edges(G,g_pred,gdict):
	if MAX_MATCH==0:
		for n1 in G.nodes():
			nn1=n1.split(':')
			if nn1[1]=='Predicate' and n1 in g_pred:
				for n2 in G.nodes():
					nn2=n2.split(':')
					if n2!=n1 and nn2[1]=='Predicate' and n2 in g_pred:
						value=cosine_similarity(g_pred[n1],g_pred[n2])
						if value>=threshold:
							value=round(value,2)
							#Add bidirectional alignment edges between the predicates; no associated doc with edge
							G.add_edge(n1,n2,weight=value,wlist=[value],etype='Predicate_Alignment')
							G.add_edge(n2,n1,weight=value,wlist=[value],etype='Predicate_Alignment')
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
						if value>=threshold:
							value=round(value,2)
							#Add bidirectional alignment edges between the predicates; no associated doc with edge
							G.add_edge(n1,n2,weight=value,wlist=[value],etype='Predicate_Alignment')
							G.add_edge(n2,n1,weight=value,wlist=[value],etype='Predicate_Alignment')
							if verbose:
								print ("Predicate Alignment Edge ",n1,n2,G.get_edge_data(n1,n2))
								print ("Predicate Alignment Edge ",n2,n1,G.get_edge_data(n2,n1))							
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
		print ('PROBLEM leading article ',t			)
	return s					

def add_type_edges(G,HP):
	for pat in HP:
		p1=(pat[0].decode('utf-8')).lower()+':Entity'
		if p1 in G.nodes():
			p2=(remove_leading_article(pat[1]).decode('utf-8')).lower()
			print ("Removed articles ",p2)
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


    
def add_entity_alignment_edges(G,mention_dict,g_ent):
	for n1 in G.nodes():
		nn1=n1.split(':')
		if nn1[1]=='Entity':
			for n2 in G.nodes():
				nn2=n2.split(':')
				if n2!=n1 and nn2[1]=='Entity':		
					if nn1[0] in mention_dict and nn2[0] in mention_dict:
						inter=len(mention_dict[nn1[0]].intersection(mention_dict[nn2[0]]))
						uni=len(mention_dict[nn1[0]].union(mention_dict[nn2[0]]))
						jack=float(inter)/float(uni)
						if jack>=threshold2:
							value=round(jack,2)
							#Add bidirectional alignment edges between the predicates; no associated doc with edge
							G.add_edge(n1,n2,weight=value,wlist=[value],etype='Entity_Alignment')
							G.add_edge(n2,n1,weight=value,wlist=[value],etype='Entity_Alignment')
							if verbose:
								print ("Entity Alignment Edge ",n1,n2,G.get_edge_data(n1,n2))
								print ("Entity Alignment Edge ",n2,n1,G.get_edge_data(n2,n1))	
		
					else:
						'''
						if n1 in g_ent:
							avec=g_ent[n1]
							for n2 in G.nodes():
								nn2=n2.split(':')
								if n2!=n1 and nn2[1]=='Entity' and n2 in g_ent:
									bvec=g_ent[n2]
									value=cosine_similarity(avec,bvec)	
									if value>threshold2:
										value=round(value,2)
										#Add bidirectional alignment edges between the predicates; no associated doc with edge
										G.add_edge(n1,n2,weight=value,wlist=[value],etype='Entity_Alignment',did=[],dtitle=[],sid=[])
										G.add_edge(n2,n1,weight=value,wlist=[value],etype='Entity_Alignment',did=[],dtitle=[],sid=[])
										if verbose:
											print "Entity Alignment Edge ",n1,n2,G.get_edge_data(n1,n2)
											print "Entity Alignment Edge ",n2,n1,G.get_edge_data(n2,n1)					
						'''
						part1=nn1[0].replace(',',' ')
						part1=part1.replace('-',' ')
						wn1=set(part1.split())
						part2=nn2[0].replace(',',' ')
						part2=part2.replace('-',' ')
						wn2=set(part2.split())
						
						inter=len(wn1.intersection(wn2))
						uni=len(wn1.union(wn2))
						jack=float(inter)/float(uni)
						if jack>=threshold2:
							value=round(jack,2)
							#Add bidirectional alignment edges between the predicates; no associated doc with edge
							G.add_edge(n1,n2,weight=value,wlist=[value],etype='Entity_Alignment')
							G.add_edge(n2,n1,weight=value,wlist=[value],etype='Entity_Alignment')
							if verbose:
								print ("Entity Alignment Edge ",n1,n2,G.get_edge_data(n1,n2))
								print ("Entity Alignment Edge ",n2,n1,G.get_edge_data(n2,n1))	
				
				
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
            for i in range(0,len(wlist1)):
                #print "i ",i
                if wlist1[i]>1.0 and wlist1[i]<=1.0001:
                    wlist1[i]=1.0 	
            if d>1.0 and d<=1.0001:
                d=1.0
            elist.append((n1,n2,d,wlist1,etype1))
            G.add_edge(n1,n2,weight=d, wlist=wlist1, etype=etype1)	
	
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


def add_nodes_weights2(G,cornerdict):
    for n1 in G.nodes():
        if n1 in cornerdict:
            val = cornerdict[n1]
            G.node[n1]['weight']=round(val,2)
    return G



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
                    #print("The weights tother = *********************************************************", nn1[0], entdict[nn1[0]], " The mean = ", max_val)
        elif nn1[1] == 'Predicate':
            if nn1[0] in predict:
                if meanval == True:
                    max_val = mean(predict[nn1[0]])
                elif meanval == False:
                    max_val = sum(predict[nn1[0]])
                #print("The weights tother = *********************************************************", predict[nn1[0]], " The mean = ", max_val)
                G.node[n1]['weight']=round(max_val,2)
    return G

def getmainFact(xp, score=0.0):
    mainfact = xp[0:3]
    mainfact.insert(1, str(score))
    mainfact.insert(3, str(score))
    return [' ### '.join(mainfact)]
def getQualifiers(xp, score=0.0):
    #return ['Qualifier | '+' | '.join([xp[i],xp[i+1]]) for i in range(3,len(xp)-1) if i%2 != 0 ]
    return ['Qualifier ### '+str(score)+' ###'+' ### '.join([xp[i],xp[i+1]]) for i in range(3,len(xp)-1) if i%2 != 0 ]

def getcornerstone(G,questn_tokens, entitynames,predalias_tokens,stempred):
    corner = {}
    count = 0
    ques_corner = set(questn_tokens)
    for  n in G.nodes():
        nn2 = n.split(':')
        if not (nn2[1]=='Entity' or nn2[1]=='Predicate'):
            continue
        #print(' the nodes are -------> ', n)
        if nn2[1]=='Entity':
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

#merge predicate aliases so that I can use them for collecting predicate cornerstone
def mergealiases(textlist):
    text = ' '.join(textlist)
    #print(text)
    finaltok = [w for w in text.lower().split() if not w in sw]
    return set(finaltok)

def stemword(item):
    return porter.stem(item)


def degenerate(G):
    #print("is it direted = ", nx.is_directed(G))
    for u,v,d in G.edges(data=True):
        d['wlist'] = [0.0]
        d['weight'] = 0 
    return G

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
    G, nodeweightavg = build_graph_from_triple_edges2(G,unique_SPO_dict,questn_tokens,predalias_tokens,entity_names,stempred,entcornerdict,pred_count,type_count)

    #return the graph
    return G, nodeweightavg





def formatTriple(triples):
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
    print(allfacts)

    for line in allfacts:
        #sent[s_id][0]+' | '+sent[s_id][1]+' | '+sent[s_id][2]+' | '+s.encode('utf-8')+' | '+str(d1)+' | '+p.encode('utf-8')+' | '+str(d2)+' | '+o.encode('utf-8')
        #if verbose:
        #       print line
        #l=line.strip().split(' | ')
        triple=line.strip().split(' ### ')
        print('tple == ', triple)
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



def call_main_GRAPH(spofile,typefile,cornerstone_file, graph_file,path_file, entity_dir, questn_tokens,pred_aliases_dir, config,topcontext=0,stempred=False,degeneratex=False):
    f11=open(spofile,'r')
    #f22=open(typefile,'r')
    scoretriplesx = list(set([line.strip() for line in f11 if '### instance of ###' not in line and '### occupation ###' not in line]))
    scoretriples = select_triples(scoretriplesx,topcontext)
    
    with open(entity_dir, 'r') as filehandle:
        entity_names = json.load(filehandle)
    #print("enity names = ", entity_names)

    with open(pred_aliases_dir, 'r') as filehandle:
        aliases = json.load(filehandle)

    predalias_tokens = {al.lower():mergealiases(listtext) for al,listtext in aliases.items()}

    #print("predicate aliases = ",predalias_tokens)



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

    if int(config['addType']) == 1:
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
    G, nodeweightavg, entcornerdict,pred_count,type_count = build_graph_from_triple_edges(unique_SPO_dict,questn_tokens,predalias_tokens,entity_names,stempred)
    if config['connectseed'] == '1':
        G,nodeweightavg = addtoGraph(G,nodeweightavg,questn_tokens,predalias_tokens,path_file,entity_names,stempred,entcornerdict,pred_count,type_count)
    G = update_edge_weight(G)
    if degeneratex == True:
        #print("Setting weight to 1 ")
        G = degenerate(G)
    #print([G[n11][n22]['wlist'] for (n11,n22) in G.edges()])
    #G, nodeweightavg = build_graph_from_triple_edges(unique_SPO_dict)
    print("The graph has ", len(G.nodes()), ' nodes and ', len(G.edges()), ' edges ')
    #G=G2.to_undirected() #make QKG Undirected
    G=directed_to_undirected(G)
    print("The graph has ", len(G.nodes()), ' nodes and ', len(G.edges()), ' edges ') 
    if len(G.nodes())>0:
        G=max(nx.connected_component_subgraphs(G), key=len)
    #print("The graph has ", len(G.nodes()), ' nodes and ', len(G.edges()), ' edges ')
    #print(list(G.edges))
    #G = add_nodes_weights(G,entdict, predcornerdict,meanval=False)
    #get the corner stone
    nodeweights = {}
    nodeweights = dict((k, v) for k, v in nodeweightavg.items() if k in G.nodes())
    #if the nodes are still present in the graph after removal and they have a score above the threshold, they are cornerstone
    #corner = dict((k, v) for k, v in nodeweightavg.items() if v >= cornerthreshold and k in G.nodes())
    G = add_nodes_weights2(G,nodeweights)

    corner = getcornerstone(G,questn_tokens,entity_names, predalias_tokens,stempred)
    nx.write_gpickle(G,graph_file)
    pickle.dump(corner,open(cornerstone_file,'wb'))
    return True



def call_main_GRAPH2(f1, f4, f5, f6, f7, gdict, prune, verbose, gt1, config,h1,h2):
	global threshold
	global threshold2
	global MAX_MATCH
	
	'''
	if verbose:
	print argv
	if len(argv)==8:
		f1=argv[1] #input triple file
		f2=argv[2] #input context file
		f3=argv[3] #question file
		f4=argv[4]
		f5=argv[5] #mention dictionary pickle files
		f6=argv[6] #output graph path
		f7=argv[7] #OUTput cornerstone path
		
	else:
		if verbose:
	print "Wrong Number of Arguments to generate graph\n"
		sys.exit(2)	
	'''
	
	
	
	threshold=h2 #float(config['Cosine_threshold'])
	threshold2=h1 #float(config['jackard_threshold'])
	MAX_MATCH=int(config['Embedding_Similarity'])
	
	print ("At call main graph ",threshold, threshold2, MAX_MATCH)
	
	Wikitype_flag=int(config['Wikitype'])
	Wiki_Threshold=float(config['Wiki_Threshold'])

	Type_Alignment_flag=int(config['Type_Alignment'])
	Distribute_Node_wt_flag=int(config['Distribute_Node_wt'])

	Cornerstone_Matching= int(config['Cornerstone_Matching'])
	
	f11=open(f1,'r')
	#f22=open(f2,'r')
	#f33=open(f3,'r')
	f44=open(f4,'r')
	mention_dict=pickle.load(open(f5,'r'))
	
	
	#Read Question text
	#for line in f33:
	#	ques=line.strip()
	
	#ques=ques.split()
	#question=[]
	#if verbose:
		#print "\n\nQuestion ",ques
	
	'''
	#Remove stopwords
	for w in ques:	
		if w not in sw:
			question.append(w)
	if verbose:
		print "\n\nQuestion W/O Stopword ",question
	'''
	#Read entities of question
	q_ent=set()
	type_qent={}
	for line in f44:
		line=line.strip()
		line=line.split()
		s=line[0]
		for i in range(1,len(line)-1):
			s+=' '+line[i]
		q_ent.add(s.lower())
		type_qent[s.lower()]=line[len(line)-1]
		
	print ("Query terms ->",len(q_ent),q_ent,type_qent	)
	if len(q_ent)>prune:
		print ("Pruning..")
		p=frozenset(q_ent)
		for s in p:
			if type_qent[s]=='P':
				q_ent.remove(s)
						
	print ("Without Predicate Query terms ->",len(q_ent),q_ent,type_qent)
	
	p=frozenset(q_ent)
	for s in p:
		if s in aux_list and type_qent[s]=='P':
			q_ent.remove(s)
			
	print ("Without Auxiliary Query terms ->",len(q_ent),q_ent,p)
		
	#triple_list=[]
	
	corner_ent={}
	unique_SPO_dict={}
	first_flag=0
	
	for line in f11:
		#sent[s_id][0]+' | '+sent[s_id][1]+' | '+sent[s_id][2]+' | '+s.encode('utf-8')+' | '+str(d1)+' | '+p.encode('utf-8')+' | '+str(d2)+' | '+o.encode('utf-8')
		#if verbose:
		#	print line
		
		#l=line.strip().split(' | ')
		triple=line.strip().split(' | ')
		#triple_list.append((l[0],l[1],l[2],l[3],l[4],l[5],l[6],l[7]))
		
		if triple[0]!='Qualifier':
			first_flag=1
			score_n1=0.0
			score_n3=0.0
			matched_n1=''
			matched_n3=''
			
			n1_id=triple[0].decode('utf-8')
			n1=triple[1].decode('utf-8').replace(':',' ')
			n2=triple[2].decode('utf-8').replace(':',' ')
			n3_id=triple[3].decode('utf-8')	
			n3=triple[4].decode('utf-8').strip('"').replace(':',' ')
			
			if (n1,n2,n3) not in unique_SPO_dict:
				unique_SPO_dict[(n1,n2,n3)]={}
			
			if n1_id.startswith('corner#'):
				n1_id=n1_id.replace('corner#','')
				
				n11=n1_id.split('#')
				n1_id=n11[0]
				score_n1=float(n11[1]) 
				term=n11[2] #The match from Ambiverse
				term_words=set(term.split())
				
				for terms in q_ent:
					qterm=set(terms.split())
					if len(term_words.intersection(qterm))>0:
						matched_n1=terms
						break
				
				if matched_n1 not in corner_ent:
					corner_ent[matched_n1]=set()
				corner_ent[matched_n1].add(n1.lower()+':Entity')
			
			if n3_id.startswith('corner#'):
				n3_id=n3_id.replace('corner#','')
				
				n33=n3_id.split('#')
				n3_id=n33[0]
				score_n3=float(n33[1]) 
				term=n33[2] #The match from Ambiverse
				term_words=set(term.split())
				
				for terms in q_ent:
					qterm=set(terms.split())
					if len(term_words.intersection(qterm))>0:
						matched_n3=terms
						break
				
				if matched_n3 not in corner_ent:
					corner_ent[matched_n3]=set()
				corner_ent[matched_n3].add(n3.lower()+':Entity')
				
			unique_SPO_dict[(n1,n2,n3)]['score_n1']=score_n1
			unique_SPO_dict[(n1,n2,n3)]['score_n3']=score_n3
			unique_SPO_dict[(n1,n2,n3)]['matched_n1']=matched_n1
			unique_SPO_dict[(n1,n2,n3)]['matched_n3']=matched_n3		
		
		else:
			if first_flag==0:
				continue
			score_qual=0.0
			matched_qual=''
			
			qual1=triple[2].decode('utf-8').replace(':',' ')
			qual2_id=triple[3].decode('utf-8')
			qual2=triple[4].decode('utf-8').strip('"').replace(':',' ')
			
			if qual2_id.startswith('corner#'):
				qual2_id=qual2_id.replace('corner#','')
				
				qual22=qual2_id.split('#')
				qual2_id=qual22[0]
				score_qual=float(qual22[1]) 
				term=qual22[2] #The match from Ambiverse
				term_words=set(term.split())
				
				for terms in q_ent:
					qterm=set(terms.split())
					if len(term_words.intersection(qterm))>0:
						matched_qual=terms
						break
				
				if matched_qual not in corner_ent:
					corner_ent[matched_qual]=set()
				corner_ent[matched_qual].add(qual2.lower()+':Entity')
			
			if 'qualifier' not in unique_SPO_dict[(n1,n2,n3)]:
				unique_SPO_dict[(n1,n2,n3)]['qualifier']=[]
			unique_SPO_dict[(n1,n2,n3)]['qualifier'].append((qual1,qual2))		
			
			if 'score_qual' not in unique_SPO_dict[(n1,n2,n3)]:
				unique_SPO_dict[(n1,n2,n3)]['score_qual']=[]
			unique_SPO_dict[(n1,n2,n3)]['score_qual'].append(score_qual)
			
			if 'matched_qual' not in unique_SPO_dict[(n1,n2,n3)]:
				unique_SPO_dict[(n1,n2,n3)]['matched_qual']=[]
			unique_SPO_dict[(n1,n2,n3)]['matched_qual'].append(matched_qual)
			
	#if verbose:
		#print triple_list
	if verbose:
		print ("\n\nAdding SPO triple edges\n\n")
		
		
	#Question vectors and node vectors are built inside	
	G=build_graph_from_triple_edges(unique_SPO_dict,q_ent,type_qent,gdict,mention_dict,Cornerstone_Matching)
	del unique_SPO_dict
	#if verbose:
	print ("\n\nSize of the graph after SPO ",len(G.nodes()),len(G.edges()))
	#visualize_graph(G)
	
	context_match_flag=0
	for gt in gt1:
		for n in G.nodes():
			nn=n.split(':')
			if len(nn)>0:
				if nn[0]==(gt.lower()):#.decode('utf-8'):
					context_match_flag=1
		
			else:
				if n==(gt.lower()):#.decode('utf-8'):
					context_match_flag=1	
					
	print ("Context Match flag ",context_match_flag	)
	
	'''
	#Add type edges from hearst patterns
	if verbose:
		print "\n\nAdding type nodes and type edges\n\n"
	HP=get_hearst_patterns_from_context(f22)
	G=add_type_edges(G,HP)
	#if verbose:
	print "\n\nSize of the graph after HP",len(G.nodes()),len(G.edges())
		
	G=add_NE_type_edges(G,NE_types)
	#if verbose:
	print "\n\nSize of the graph after NE type",len(G.nodes()),len(G.edges())	
	#visualize_graph(G)
	
	if Wikitype_flag==1:
		f22.close()
		f22=open(f2,'r')
	
		#Add wiki categories as types
		wiki_cat=get_wiki_categories_from_context(f22,Wiki_Threshold)
		G=add_wiki_type_edges(G,wiki_cat)
		#if verbose:
		print "\n\nSize of the graph after wiki edges",len(G.nodes()),len(G.edges())
	'''
	#Read Glove embeddings
	g_pred,g_ent,g_type,g_ques=read_glove(G,q_ent,gdict)
	
	
	#Add relation alignment edges from glove embeddings
	#print "\n\nAdding predicate alignment edges\n\n"
	#G=add_predicate_alignment_edges(G,g_pred,gdict)
	#print "\n\nSize of the graph ",len(G.nodes()),len(G.edges())
	#visualize_graph(G)
	
	#Add type alignment edges from glove embeddings
	#if Type_Alignment_flag==1:
	#	print "\n\nAdding type alignment edges\n\n"
	#	G=add_type_alignment_edges(G,g_type,gdict)
	#	if verbose:
	#		print "\n\nSize of the graph ",len(G.nodes()),len(G.edges())
	#visualize_graph(G)
	
	
	#Add entity alignment edges from mention dictonary
	#if verbose:
	#print "\n\nAdding entity alignment edges\n\n"
	#G2=add_entity_alignment_edges(G,mention_dict,g_ent)
	#if verbose:
	#print "\n\nSize of the graph directed",len(G.nodes()),len(G.edges())
	#visualize_graph(G)
	
	#G=G2.to_undirected() #make QKG Undirected
	G=directed_to_undirected(G)
	
	if len(G.nodes())>0:
		G=max(nx.connected_component_subgraphs(G), key=len)
	
	#Add node weights		
	if verbose:
		print ("\n\nAdding node weights\n\n")
	#G=add_predicate_type_node_weights(G,g_pred,g_type,g_dict,question)
	#G=add_entity_node_weights(G,q_ent,mention_dict)		
	
	G=add_node_weights(G,g_ent,g_pred,g_type,gdict,g_ques,q_ent,mention_dict,type_qent,Cornerstone_Matching)
	print ("\n\nSize of the graph ",len(G.nodes()),len(G.edges())	)
	#visualize_graph(G)
	
	
	if Distribute_Node_wt_flag==2:
		G=distribute_node_weights(G)
	
	nx.write_gpickle(G,f6)
	
	
	match_flag=0
	for gt in gt1:
		for n in G.nodes():
			nn=n.split(':')
			if len(nn)>0:
				if nn[0]==(gt.lower()):#.decode('utf-8'):
					match_flag=1
			else:
				if n==(gt.lower()):#.decode('utf-8'):
					match_flag=1	
	
	corner1={}
	corner2={}
	for n in G.nodes():
		#if n.split(':')[1]=='Entity' and G.node[n]['matched']!='':
		if G.node[n]['matched']!='':
			if G.node[n]['matched'] not in corner2:
				corner2[G.node[n]['matched']]=[]
			corner2[G.node[n]['matched']].append((n,G.node[n]['weight']))
	
	for term in corner2:
		corner2[term]=sorted(corner2[term],key=lambda x:x[1], reverse=True)
		corner1[term]=set()
		mmax=5
		if len(corner2[term])<mmax:
			mmax=len(corner2[term])
		for mm in range(0,mmax):
			tt=corner2[term][mm][0].split(':')
			if tt[1]=='Entity' and corner2[term][mm][1]>=threshold2:
				corner1[term].add(corner2[term][mm][0])
			else:
				if corner2[term][mm][1]>=threshold:
					corner1[term].add(corner2[term][mm][0])
	
	print ("Entity+Predicate Cornerstones ",corner1						)
	cornerstone={}
	
	'''
	for n in G.nodes():
		flag=0
		if flag==0 and n.split(':')[1]!='Entity' and G.node[n]['weight']>=threshold: #((n.split(':')[1]=='Entity' and G.node[n]['weight']>=threshold2) or (n.split(':')[1]!='Entity' and G.node[n]['weight']>=threshold)): 
			#cornerstone[n]=G.node[n]['matched']
			
			if G.node[n]['matched'] not in corner1:
				corner1[G.node[n]['matched']]=set()
			corner1[G.node[n]['matched']].add(n)
	
	print "Entity+Predicate Cornerstones ",corner1
	'''
	q_ent=set() #keeping terms with non-zero cornerstones
	corner_ct=0
	for ent in corner1:
		if ent!='':
			q_ent.add(ent)
			corner_ct+=len(corner1[ent])
	print ("Current non zero corner ",q_ent, len(corner1)	)
	
	
	#for e in corner1:
	#	print "corner 1 ",e,corner1[e]
		
	#for e in corner_ent:
	#	print "corner ent ",e,corner_ent[e]
	
	#if verbose:
	
	'''
	corner1={}
	for ent in q_ent:
		corner1[ent]=set()
		for v in cornerstone:
			if cornerstone[v]==ent:
				corner1[ent].add(v)
	'''
	'''
	if corner_ct > 50:
		print "Pruning.."
		p=frozenset(q_ent)
		for s in p:
			if type_qent[s]=='P':
				q_ent.remove(s)
						
	print "2nd time Without Predicate Query terms ->",len(q_ent),q_ent
	''' 
	if len(q_ent)>prune:
		print ("prunning again based on NE...")#,q_ent,corner1
		p=frozenset(q_ent)
		for s in p:
			if s in type_qent and type_qent[s]=='S':
				q_ent.remove(s)
		print ("With only NE Query terms ->",len(q_ent),q_ent	)
			
		if len(q_ent)<prune:
			print ("adding back few Subject terms to fulfil upto prune ...")
			
			clist=[]
			for ent in corner1:
				if ent not in q_ent and type_qent[ent]=='S':
					clist.append((ent,len(corner1[ent])))
			clist=sorted(clist,key=lambda x:x[1],reverse=True)
			p=frozenset(q_ent)
			
			print ("p, q_ent, clist, prune-len(p)",p, q_ent, clist, prune-len(p))
			
			for i in range(0,prune-len(p)):
				q_ent.add(clist[i][0])
		else:
			if len(q_ent)>prune:
				print ("prunning again based on number of cornerstones per query term...")
				clist=[]
				for ent in q_ent:
					clist.append((ent,len(corner1[ent])))	
				clist=sorted(clist,key=lambda x:x[1],reverse=True)
				q_ent=set()
				for i in range(0,prune):
					q_ent.add(clist[i][0]) 		
			
		corner2={}
		for v in corner1:
			if v in q_ent:
				corner2[v]=corner1[v]
		corner1=corner2						
						
	
	cornerstone={}
	for v in corner1:
		for e in corner1[v]:
			cornerstone[e]=v					
	
	print ("Final query terms -->",len(q_ent),q_ent		)
	print ("\nCornerStones Are --->",len(corner1), len(cornerstone))
	for v in corner1:
		print ("\n",v, corner1[v],len(corner1[v])	)
	pickle.dump(cornerstone,open(f7,'w'))
	
	return q_ent,len(cornerstone),match_flag,  context_match_flag 
		
 
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

def select_triples(lsto,topcontext=0):
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
















if __name__ == "__main__":
    main(sys.argv)			
			
