import networkx as nx
from heapq import heappush,heappop,heapify
import queue
import pickle
import sys
import numpy as np
import math

import gensim
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
#model = Word2Vec(common_texts, size=300, window=5, min_count=1, workers=4)
#model.save("word2vec.model")
#model = gensim.models.Word2Vec.load_word2vec_format('./../files/GoogleNews-vectors-negative300.bin', binary=True)  
#model = gensim.models.KeyedVectors.load_word2vec_format('./../files/GoogleNews-vectors-negative300.bin.gz', binary=True)  
#word_vectors = model.wv
verbose=0
threshold=0.75
MAX_MATCH=1
Distribute_Node_wt_flag=0
chain_join_flag=0

stop_list=set(['a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'aren\'t', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can\'t', 'cannot', 'could', 'couldn\'t', 'did', 'didn\'t', 'do', 'does', 'doesn\'t', 'doing', 'don\'t', 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn\'t', 'has', 'hasn\'t', 'have', 'haven\'t', 'having', 'he', 'he\'d', 'he\'ll', 'he\'s', 'her', 'here', 'here\'s', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'how\'s', 'i', 'i\'d', 'i\'ll', 'i\'m', 'i\'ve', 'if', 'in', 'into', 'is', 'isn\'t', 'it', 'it\'s', 'its', 'itself', 'let\'s', 'me', 'more', 'most', 'mustn\'t', 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours	ourselves', 'out', 'over', 'own', 'same', 'shan\'t', 'she', 'she\'d', 'she\'ll', 'she\'s', 'should', 'shouldn\'t', 'so', 'some', 'such', 'than', 'that', 'that\'s', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'there\'s', 'these', 'they', 'they\'d', 'they\'ll', 'they\'re', 'they\'ve', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', 'wasn\'t', 'we', 'we\'d', 'we\'ll', 'we\'re', 'we\'ve', 'were', 'weren\'t', 'what', 'what\'s', 'when', 'when\'s', 'where', 'where\'s', 'which', 'while', 'who', 'who\'s', 'whom', 'why', 'why\'s', 'with', 'won\'t', 'would', 'wouldn\'t', 'you', 'you\'d', 'you\'ll', 'you\'re', 'you\'ve', 'your', 'yours', 'yourself', 'yourselves'])
'''
g1=nx.fast_gnp_random_graph(5, 0.5, seed=None, directed=False)
g2=nx.fast_gnp_random_graph(6, 0.3, seed=None, directed=False)
g3=nx.fast_gnp_random_graph(10, 0.8, seed=None, directed=False)

if verbose:
	print g1.nodes(),g1.edges()


if verbose:
	print g2.nodes(),g2.edges()
if verbose:
	print g3.nodes(),g3.edges()


h=[]
heappush(h, (5, g1))
heappush(h, (15, g2))
heappush(h, (2, g3))

x=heappop(h)
if verbose:
	print x[0]
if verbose:
	print x[1].nodes(),x[1].edges()
'''

def get_cost(t,G):
	c=0.0
	for (n1,n2) in t.edges():
		if (n1,n2) in G.edges():
			data=G.get_edge_data(n1,n2)
		else:
			data=G.get_edge_data(n2,n1)
		for d in data['wlist']:	#Use if sum of cost is needed	
			#d=data['weight']
			c+=1-d #cost=1-weight
		#if d==0:
		#	print "Edge d 0",data['weight'],data['wlist'],data['etype'],n1,n2
		#c+=1-d
	if c<0:
		print ("\n\n ==== =========== ERROR NEG COST ===== \n\n",c	)
	
	if Distribute_Node_wt_flag==1:
		for n in t.nodes():
			node_weight=G.node[n]['weight']
			#if node_weight>1:
			#	node_weight=1
			c+=1-node_weight	
		if c<0:
			print ("\n\n ==== =========== ERROR NEG COST Node===== \n\n",c		)
				
	return c
	
def grow_graph(g1,v,u):
	g=nx.Graph()
	for n in g1.nodes():
		g.add_node(n)
	for (n1,n2) in g1.edges():
		g.add_edge(n1,n2)
	#flag=0		
	if u not in g.nodes():
		g.add_node(u)
		g.add_edge(v,u) #to keep it a tree #access weight from G if needed
		#flag=1
		
	return g#,flag			

def merge_graph(g1,g2):
	g=nx.Graph()
	for n in g1.nodes():
		g.add_node(n)
	for (n1,n2) in g1.edges():
		g.add_edge(n1,n2)
	#flag=0	
	for n in g2.nodes():
		if n not in g.nodes():
			g.add_node(n)
			#flag=1
	for (n1,n2) in g2.edges():
		if (n1,n2) not in g.edges() and (n2,n1) not in g.edges():
			g.add_edge(n1,n2)
			#flag=1
	return g	 		


def update_queue(Q,cg,u,p):
	for i in range(0,len(Q)):
		if Q[i][1]==u and Q[i][2]==p:
			#if verbose:
				#print "i th ",i,Q[i]
			Q[i]=(cg,u,p)
			heapify(Q)
			break
	return	Q	
			
def check_history(pop_hist,v,p):
	if v in pop_hist:
		if p in pop_hist[v]:
			if verbose:
				print ("\nsame item popped")
			return 0
	return 1
	
def save_GST(GST_set,v,g1,corner,G):
    g=nx.Graph()
    if chain_join_flag==1:
        for n in g1.nodes():
            g.add_node(n)
            nn=n.split(':')
            if nn[1]=='Predicate' and n in corner: #Add neighbour entities from SPO which are not corner stones
                for nb in G.neighbors(n):
                    data=G.get_edge_data(n,nb)
                    if nb.split(':')[1]=='Entity' and nb not in g1.neighbors(n) and nb not in corner and data['etype']=='Triple':
                        g.add_node(nb)
                        g.add_edge(n,nb)
    for (n1,n2) in g1.edges():
        g.add_edge(n1,n2)
    
    pot_ans_flag=0
    for n in g.nodes():
        nn=n.split(':')
        if nn[1]=='Entity' and n not in corner: #At least one non-cornerstone entity node
            pot_ans_flag=1
    flag=1
    for (v2,g2) in GST_set:
        if set(g.edges())==set(g2.edges()):
            #if g==g2:
            flag=0
            break
    
    if flag==1 and pot_ans_flag==1:
        GST_set.append((v,g))
    return GST_set,flag, pot_ans_flag	
			
def get_GST(Q,T,P,G,no_GST, corner,verbose):
	pop_hist={}
	ite=0
	pop_cov=-1
	merge_cov=-1
	min_cost=999999
	GST_count=0
	GST_set=[]
	final_GST_cost=-99999
	final_GST_flag=0
	leave_loop=0
	
	while len(Q)>0:
		x=heappop(Q)
		
		cc=x[0]
		v=x[1]
		p=x[2]
		
		
		if verbose:
		    print ("======================================================================================>>>>>>>>>")
		if verbose:
		    print ("iteration ->", ite)
		ite+=1
		if verbose:
		    print ("\n\nPopped --->", x[0],x[1],x[2],len(p.intersection(P)), len(P),len(Q),pop_cov,merge_cov,min_cost)
		if len(p)>pop_cov:
			pop_cov=len(p)
			if verbose:
				print ("pop length increased to ",len(p))
			
		check=check_history(pop_hist,v,p)
		if check==0:
			continue #To avoid executing same pop
		if v not in pop_hist:
			pop_hist[v]=set()
		pop_hist[v].add(p)		
			
		if p==P:
			current_cost=get_cost(T[v][p],G)
			if final_GST_cost>-99999 and current_cost>final_GST_cost: #No more GST with same cost as final GST
				leave_loop=1
				break
				
			GST_set,New_GST_flag,pot_ans_flag=save_GST(GST_set,v,T[v][p],corner,G)
			if New_GST_flag==1 and pot_ans_flag==1:
				GST_count+=1
			if GST_count==no_GST and final_GST_flag==0:
				final_GST_cost=get_cost(T[v][p],G)
				final_GST_flag=1
		
		if leave_loop==1:
			break					
		#GROW	
		for u in G.neighbors(v):
			if verbose:
				print ("\nneighbor ",u,T[v][p].nodes(),T[v][p].edges(),len(Q))
			#gflag=0
			g=grow_graph(T[v][p],v,u) #Crete temporary merge tree/graph
			if verbose:
				print ("\nGrown ",v,p,g.nodes(),g.edges(),len(Q))
			cg=get_cost(g,G) 
			flag=0
			if u in T and p in T[u]: 	
				cu=get_cost(T[u][p],G)
				if cg<cu: #If T[u][p] already exists, check if cost(merged_graph)<T[u][p]; in that case update T[u][p]
					if verbose:
						print ("\nBefore Growth updated ",u,p,cg,cu, T[u][p].nodes(),T[u][p].edges(), get_cost(T[u][p],G),len(Q))
					T[u][p]=g
					flag=1
					if verbose:
						print ("\nAfter Growth updated ",u,p,cg,cu, T[u][p].nodes(),T[u][p].edges(), get_cost(T[u][p],G),len(Q))
					Q=update_queue(Q,cg,u,p)
					
			else: #If T[u][p] does not exist, create T[u][p]=merged graph
				#if gflag==1:
				if u not in T:
					T[u]={}	
				T[u][p]=g
				flag=1
				#if flag==1:
				heappush(Q,(cg,u,p))
				if verbose:
					print ("\nGrowth created new root, query",u,p,cg, T[u][p].nodes(),T[u][p].edges(), get_cost(T[u][p],G),len(Q))
				
				
		
		#MERGE
		p1=p
		
		all_p2=set()
		for p2 in T[v]:
			all_p2.add(p2)
		
		for p2 in all_p2: #because T[v] changes during iteration
			if len(p1.intersection(p2))==0:
				#mflag=0
				g=merge_graph(T[v][p1],T[v][p2])
				cg=get_cost(g,G) 	
				p=frozenset(p1.union(p2))
				if p in T[v]:#all_p2:
					cp=get_cost(T[v][p],G)
					if cg<cp:
						if verbose:
							print ("\nBefore merge updated",v,p,cg,cp,T[v][p].nodes(),T[v][p].edges(), get_cost(T[v][p],G),len(Q))
						T[v][p]=g
						#heappush(Q,(cg,v,pp))
						Q=update_queue(Q,cg,v,p)
						if verbose:
							print ("\nmerge updated",v,p1,p2,p,cg,cp,T[v][p].nodes(),T[v][p].edges(), get_cost(T[v][p],G),len(Q))
						
				else:
					#if mflag==1:
					T[v][p]=g
					heappush(Q,(cg,v,p))	
					if len(p)>merge_cov:
						merge_cov=len(p)
						if verbose:
							print ("merge length increased to ",len(p))
						if len(p)==len(P):
							if verbose:
								print ("\nmerge created new query",v,p,cg,T[v][p].nodes(),T[v][p].edges(), get_cost(T[v][p],G),len(Q))
							if cg<min_cost:
								min_cost=cg
					if verbose:
						print ("\nmerge created new query",v,p,cg,T[v][p].nodes(),T[v][p].edges(), get_cost(T[v][p],G),len(Q))
						print ("v p1",v,p1,T[v][p1].nodes(),T[v][p1].edges(), get_cost(T[v][p1],G),len(Q))
						print ("v p2",v,p2,T[v][p2].nodes(),T[v][p2].edges(), get_cost(T[v][p2],G),len(Q))
	return 	GST_set		
				
							

def initialize_queue(G,corner):
	Q=[]
	T={}
	for v in corner:
		T[v]={}
		g = nx.Graph()
		g.add_node(v) #access weight from G if needed
		p=frozenset(corner[v]) #Query term
		T[v][p]=g
		c=get_cost(g,G)
		heappush(Q,(c,v,p))
		
	for v in T:
		if verbose:
			print ("Vertex ",v)
		for p in T[v]:
			if verbose:
				print ("Query and tree --->",p,T[v][p].nodes(),T[v][p].edges())
	if verbose:
		print ("Queue --->",Q 		)
	return T,Q				

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
	
	
def get_type_simi(n0,G,ans_type,gdict):
	veclen=300
	term_types=set()
	for (n1,n2) in G.edges():
		if n0==n2 and n1.split(':')[1]=='Type':
			term_types.add(n1.split(':')[0])
		else:
			if n0==n1 and n2.split(':')[1]=='Type': 
				term_types.add(n2.split(':')[0])
				
	#if verbose:
		#print "Answer type, term type ",ans_type,term_types			
	#gdict=word_vectors
	
	if len(term_types)==0 or len(ans_type)==0: #if answer does not have type or node n0 does not have type, keep n0
		return 1.0
	
	
	if MAX_MATCH==1:
		maxval=0.0
		
		for n1 in term_types:
			for n2 in ans_type:
				val=cosine_similarity_MAX_MATCH(n1,n2,gdict)
				if val>maxval:
					maxval=val
		
	else:	
		t_dict={}
		a_dict={}
	
	
	
		for n in term_types:
			nw1=n.split()
			avec=np.zeros(veclen)
			c=0.0
			for el in nw1:
				if el in gdict and el.lower() not in stop_list:
					avec=np.add(avec,np.array(gdict[el]))	
					c+=1.0
			if c>0:
				avec=np.divide(avec, c)
		
			t_dict[n]=avec.tolist()
	
		for n in ans_type:
			nw1=n.split()
			avec=np.zeros(veclen)
			c=0.0
			for el in nw1:
				if el in gdict and el.lower() not in stop_list:
					avec=np.add(avec,np.array(gdict[el]))	
					c+=1.0
			if c>0:
				avec=np.divide(avec, c)
		
			a_dict[n]=avec.tolist()
	
		maxval=0.0
		
		for n1 in t_dict:
			for n2 in a_dict:
				val=cosine_similarity(t_dict[n1],a_dict[n2])
				if val>maxval:
					maxval=val
	return maxval					

def get_cornerstone_distance(T, G, corner,n0):
    w=0.0
    for n in T.nodes():
        if n in corner:
            w+=len(nx.shortest_path(T,n0,n))-1
    return w    
  
def get_cornerstone_distance_wt(T, G, corner,n0):
    w=0.0
    for n in T.nodes():
        if n in corner:
            path=nx.shortest_path(T,n0,n)
            c=0.0
            for i in range(0,len(path)-1):
                n1=path[i]
                n2=path[i+1]
                if (n1,n2) in G.edges():
                    data=G.get_edge_data(n1,n2)
                else:
                    data=G.get_edge_data(n2,n1)
                for d in data['wlist']:	#Use if sum of cost is needed
                    #d=data['weight']
                    c+=1-d #cost=1-weight
            w+=c
    return w      
        
def get_cornerstone_weight(T, G, corner):
	w=0.0
	for n in T.nodes():
		if n in corner:
			w+=G.node[n]['weight']
	return w		 

def issubseq(n1,n2):
	nw1=(n1.split(':'))[0].split()
	nw2=(n2.split(':'))[0].split()
	if len(nw1)==0:
		return 0
	i=0
	flag=0
	for j in range(0,len(nw2)):
		if nw1[i].lower()==nw2[j].lower():
			i+=1	
			if i==len(nw1):
				flag=1
				break
				
	if flag==1:
		return 1
	else:
		return 0				 




def get_corner_wt(ans_set,tree_pot,G,corner,gt1):
	#Add weights to answers
	done=set()
	answer_list=[]
	candidate_match_flag=0
	for ans in ans_set:
		w=0.0
		support=set()
		for (v1,T1) in tree_pot:
			for n1 in tree_pot[(v1,T1)]:
				if n1 in ans: 
					w+=get_cornerstone_weight(T1, G, corner)
					sent=get_edge_sentid_GST(T1,G)
					support=support.union(sent)
		
		anslist=list(ans)
		s=anslist[0]
		for i in range(1,len(anslist)):
			s=s+'|'+anslist[i]				
		answer_list.append((s,support,w))
		
		for gt in gt1:
			for i in range(0,len(anslist)):
				nnc1=anslist[i].split(':')
				#if nnc1[0]==gt.lower():
				#	candidate_match_flag=1
				if len(nnc1)>0:
					if nnc1[0]==(gt.lower()):#.decode('utf-8'):
						candidate_match_flag=1
				else:
					if anslist[i]==(gt.lower()):#.decode('utf-8'):
						candidate_match_flag=1	
				
	answer_list=sorted(answer_list,key=lambda x:x[2],reverse=True)
	top_match_flag=0
	for i in range(0,5):
		if i>=len(answer_list):
			break
		nnc1=answer_list[0][0].split(':')
		
		ans1=answer_list[i][0].split('|')
		for s in ans1:
			s=s.split(':')[0]
			s=s.strip()
			s=s.strip(',')
			#s=s.encode('utf-8')	
		
			for gt in gt1:
				if s==(gt.lower()):#.decode('utf-8'):
					top_match_flag=1
					
	if top_match_flag==0:			
		for i in range(4,len(answer_list)):
			if answer_list[i][2]==answer_list[4][2]: #same score as the 5th one	
				ans1=answer_list[i][0].split('|')
				for s in ans1:
					s=s.split(':')[0]
					s=s.strip()
					s=s.strip(',')
					#s=s.encode('utf-8')	
		
					for gt in gt1:
						if s==(gt.lower()): #.decode('utf-8'):
							top_match_flag=1	
			else:
				break					
	
	#if verbose:
	print ("\n\nFinal answer ranking --->",len(answer_list))#,answer_list
	return answer_list,candidate_match_flag,top_match_flag				

def get_tree_cost(ans_set,tree_pot,G,corner,gt1):
	#Add weights to answers
	done=set()
	answer_list=[]
	candidate_match_flag=0
	for ans in ans_set:
		w=0.0
		support=set()
		for (v1,T1) in tree_pot:
			for n1 in tree_pot[(v1,T1)]:
				if n1 in ans: 
					cost_tree=float(get_cost(T1,G))
					if cost_tree>0.0:
						w+=1.0/cost_tree#get_cornerstone_weight(T1, G, corner)
					sent=get_edge_sentid_GST(T1,G)
					support=support.union(sent)
		
		anslist=list(ans)
		s=anslist[0]
		for i in range(1,len(anslist)):
			s=s+'|'+anslist[i]				
		answer_list.append((s,support,w))
		
		for gt in gt1:
			for i in range(0,len(anslist)):
				nnc1=anslist[i].split(':')
				#if nnc1[0]==gt.lower():
				#	candidate_match_flag=1
				if len(nnc1)>0:
					if nnc1[0]==(gt.lower()):#.decode('utf-8'):
						candidate_match_flag=1
				else:
					if anslist[i]==(gt.lower()):#.decode('utf-8'):
						candidate_match_flag=1	
				
	answer_list=sorted(answer_list,key=lambda x:x[2],reverse=True)
	top_match_flag=0
	for i in range(0,5):
		if i>=len(answer_list):
			break
		nnc1=answer_list[0][0].split(':')
		
		ans1=answer_list[i][0].split('|')
		for s in ans1:
			s=s.split(':')[0]
			s=s.strip()
			s=s.strip(',')
			#s=s.encode('utf-8')	
		
			for gt in gt1:
				if s==(gt.lower()):#.decode('utf-8'):
					top_match_flag=1
					
	if top_match_flag==0:			
		for i in range(4,len(answer_list)):
			if answer_list[i][2]==answer_list[4][2]: #same score as the 5th one	
				ans1=answer_list[i][0].split('|')
				for s in ans1:
					s=s.split(':')[0]
					s=s.strip()
					s=s.strip(',')
					#s=s.encode('utf-8')	
		
					for gt in gt1:
						if s==(gt.lower()): #.decode('utf-8'):
							top_match_flag=1	
			else:
				break					
	
	#if verbose:
	print ("\n\nFinal answer ranking --->",len(answer_list))#,answer_list
	return answer_list,candidate_match_flag,top_match_flag
	
def get_tree_count(ans_set,tree_pot,G,corner,gt1):
	#Add weights to answers
	done=set()
	answer_list=[]
	candidate_match_flag=0
	for ans in ans_set:
		w=0.0
		support=set()
		for (v1,T1) in tree_pot:
			for n1 in tree_pot[(v1,T1)]:
				if n1 in ans: 
					w+=1.0#get_cornerstone_weight(T1, G, corner)
					sent=get_edge_sentid_GST(T1,G)
					support=support.union(sent)
		
		anslist=list(ans)
		s=anslist[0]
		for i in range(1,len(anslist)):
			s=s+'|'+anslist[i]				
		answer_list.append((s,support,w))
		
		for gt in gt1:
			for i in range(0,len(anslist)):
				nnc1=anslist[i].split(':')
				#if nnc1[0]==gt.lower():
				#	candidate_match_flag=1
				if len(nnc1)>0:
					if nnc1[0]==(gt.lower()): #.decode('utf-8'):
						candidate_match_flag=1
				else:
					if anslist[i]==(gt.lower()): #.decode('utf-8'):
						candidate_match_flag=1	
				
	answer_list=sorted(answer_list,key=lambda x:x[2],reverse=True)
	top_match_flag=0
	for i in range(0,5):
		if i>=len(answer_list):
			break
		nnc1=answer_list[0][0].split(':')
		
		ans1=answer_list[i][0].split('|')
		for s in ans1:
			s=s.split(':')[0]
			s=s.strip()
			s=s.strip(',')
			#s=s.encode('utf-8')	
		
			for gt in gt1:
				if s==(gt.lower()): #.decode('utf-8'):
					top_match_flag=1
					
	if top_match_flag==0:			
		for i in range(4,len(answer_list)):
			if answer_list[i][2]==answer_list[4][2]: #same score as the 5th one	
				ans1=answer_list[i][0].split('|')
				for s in ans1:
					s=s.split(':')[0]
					s=s.strip()
					s=s.strip(',')
					#s=s.encode('utf-8')	
		
					for gt in gt1:
						if s==(gt.lower()): #.decode('utf-8'):
							top_match_flag=1	
			else:
				break					
	
	#if verbose:
	print ("\n\nFinal answer ranking --->",len(answer_list))#,answer_list
	return answer_list,candidate_match_flag,top_match_flag	


def get_corner_dist(ans_set,tree_pot,G,corner,gt1):
	#Add weights to answers
	done=set()
	answer_list=[]
	candidate_match_flag=0
	for ans in ans_set:
		w=0.0
		support=set()
		for (v1,T1) in tree_pot:
			for n1 in tree_pot[(v1,T1)]:
				if n1 in ans: 
					dist_corner=float(get_cornerstone_distance(T1, G, corner, n1))
					if dist_corner>0.0:
						w+=1.0/dist_corner#get_cornerstone_weight(T1, G, corner)
					sent=get_edge_sentid_GST(T1,G)
					support=support.union(sent)
		
		anslist=list(ans)
		s=anslist[0]
		for i in range(1,len(anslist)):
			s=s+'|'+anslist[i]				
		answer_list.append((s,support,w))
		
		for gt in gt1:
			for i in range(0,len(anslist)):
				nnc1=anslist[i].split(':')
				#if nnc1[0]==gt.lower():
				#	candidate_match_flag=1
				if len(nnc1)>0:
					if nnc1[0]==(gt.lower()): #.decode('utf-8'):
						candidate_match_flag=1
				else:
					if anslist[i]==(gt.lower()): #.decode('utf-8'):
						candidate_match_flag=1	
				
	answer_list=sorted(answer_list,key=lambda x:x[2],reverse=True)
	top_match_flag=0
	for i in range(0,5):
		if i>=len(answer_list):
			break
		nnc1=answer_list[0][0].split(':')
		
		ans1=answer_list[i][0].split('|')
		for s in ans1:
			s=s.split(':')[0]
			s=s.strip()
			s=s.strip(',')
			#s=s.encode('utf-8')	
		
			for gt in gt1:
				if s==(gt.lower()): #.decode('utf-8'):
					top_match_flag=1
					
	if top_match_flag==0:			
		for i in range(4,len(answer_list)):
			if answer_list[i][2]==answer_list[4][2]: #same score as the 5th one	
				ans1=answer_list[i][0].split('|')
				for s in ans1:
					s=s.split(':')[0]
					s=s.strip()
					s=s.strip(',')
					#s=s.encode('utf-8')	
		
					for gt in gt1:
						if s==(gt.lower()): #.decode('utf-8'):
							top_match_flag=1	
			else:
				break					
	
	#if verbose:
	print ("\n\nFinal answer ranking --->",len(answer_list))#,answer_list
	return answer_list,candidate_match_flag,top_match_flag


def get_corner_dist_wt(ans_set,tree_pot,G,corner,gt1):
	#Add weights to answers
	done=set()
	answer_list=[]
	candidate_match_flag=0
	for ans in ans_set:
		w=0.0
		support=set()
		for (v1,T1) in tree_pot:
			for n1 in tree_pot[(v1,T1)]:
				if n1 in ans: 
					dist_corner=float(get_cornerstone_distance_wt(T1, G, corner, n1))
					if dist_corner>0.0:
						w+=1.0/dist_corner#get_cornerstone_weight(T1, G, corner)
					sent=get_edge_sentid_GST(T1,G)
					support=support.union(sent)
		
		anslist=list(ans)
		s=anslist[0]
		for i in range(1,len(anslist)):
			s=s+'|'+anslist[i]				
		answer_list.append((s,support,w))
		
		for gt in gt1:
			for i in range(0,len(anslist)):
				nnc1=anslist[i].split(':')
				#if nnc1[0]==gt.lower():
				#	candidate_match_flag=1
				if len(nnc1)>0:
					if nnc1[0]==(gt.lower()): #.decode('utf-8'):
						candidate_match_flag=1
				else:
					if anslist[i]==(gt.lower()): #.decode('utf-8'):
						candidate_match_flag=1	
				
	answer_list=sorted(answer_list,key=lambda x:x[2],reverse=True)
	top_match_flag=0
	for i in range(0,5):
		if i>=len(answer_list):
			break
		nnc1=answer_list[0][0].split(':')
		
		ans1=answer_list[i][0].split('|')
		for s in ans1:
			s=s.split(':')[0]
			s=s.strip()
			s=s.strip(',')
			#s=s.encode('utf-8')	
		
			for gt in gt1:
				if s==(gt.lower()): #.decode('utf-8'):
					top_match_flag=1
					
	if top_match_flag==0:			
		for i in range(4,len(answer_list)):
			if answer_list[i][2]==answer_list[4][2]: #same score as the 5th one	
				ans1=answer_list[i][0].split('|')
				for s in ans1:
					s=s.split(':')[0]
					s=s.strip()
					s=s.strip(',')
					#s=s.encode('utf-8')	
		
					for gt in gt1:
						if s==(gt.lower()): #.decode('utf-8'):
							top_match_flag=1	
			else:
				break							
	
	#if verbose:
	print ("\n\nFinal answer ranking --->",len(answer_list))#,answer_list
	return answer_list,candidate_match_flag,top_match_flag

def get_edge_sentid_GST(T,G):
    sent=set()
    for (n1,n2) in T.edges():
        if (n1,n2) in G.edges():
            data=G.get_edge_data(n1,n2)
        else:
            data=G.get_edge_data(n2,n1)
        try:
            for i in range(0,len(data['dtitle'])):
                sent.add((data['dtitle'][i],data['sid'][i]))
        except:
            print(n1,n2)
            print(data)
            i = 0/0
    return sent



def get_rank_docs(rank_doc):
	
	ans_1_10=0
	ans_11_20=0
	ans_21_30=0
	ans_31_40=0
	ans_41_50=0
	ans_51_100=0
	ans_101_200=0
	ans_201_300=0
	ans_301_400=0
	ans_401_500=0
	
	ans1=0
	ans2=0
	ans3=0
	ans4=0
	ans5=0
	
	for tup in rank_doc:
		rank=tup[0]
		aa=tup[1]
		if rank>=1 and rank<=10:
			if aa==1:
				ans_1_10=1
		else:
			if rank>=11 and rank<=20:
				if aa==1:
					ans_11_20=1
			else:
				if rank>=21 and rank<=30:
					if aa==1:
						ans_21_30=1			
				else:
					if rank>=31 and rank<=40:
						if aa==1:
							ans_31_40=1
					else:
						if rank>=41 and rank<=50:
							if aa==1:
								ans_41_50=1
						else:
							if rank>=51 and rank<=100:
								if aa==1:
									ans_51_100=1
							else:
								if rank>=101 and rank<=200:
									if aa==1:
										ans_101_200=1
								else:
									if rank>=201 and rank<=300:
										if aa==1:
											ans_201_300=1
									else:
										if rank>=301 and rank<=400:
											if aa==1:
												ans_301_400=1
										else:
											if rank>=401 and rank<=500:
												if aa==1:
													ans_401_500=1										
			
	res=[]
	res.append(ans_1_10)
	res.append(ans_11_20)
	res.append(ans_21_30)
	res.append(ans_31_40)
	res.append(ans_41_50)
	res.append(ans_51_100)
	res.append(ans_101_200)
	res.append(ans_201_300)
	res.append(ans_301_400)
	res.append(ans_401_500)
	
									
	return res

	
def call_main_GST(f1,f2,f4,f5,no_GST,gdict,verbose,gt1,config,h2):
    global threshold
    global MAX_MATCH
    global Distribute_Node_wt_flag
    global chain_join_flag
    '''
        if verbose:
            print argv
        if len(argv)==7:
            f1=argv[1] #input QKG
            f2=argv[2] #input Cornerstones
            f3=argv[3] #answer match
            f4=argv[4] #answer type
            no_GST=int(argv[5]) #number of GSTs
            verbose=int(argv[6])
	else:
            if verbose:
                print "Wrong Number of Arguments to generate graph\n"
            sys.exit(2)	
    '''
    
    threshold=h2 #float(config['Cosine_threshold'])
    MAX_MATCH=int(config['Embedding_Similarity'])
    
    #Type filtering can be none (0), Relaxed i.e. only apply if candidate list greater than 10 (1), or Strict i.e. apply always (2)
    Type_Filtering_flag=int(config['Type_Filtering'])
    Distribute_Node_wt_flag=int(config['Distribute_Node_wt'])
    chain_join_flag=int(config['chain_join_flag'])
    
    try:
        G1=nx.read_gpickle(f1)
        if verbose:
            print ("\n\nSize of the read graph ",len(G1.nodes()),len(G1.edges()))
        corner1=pickle.load(open(f2,'rb'))
    except:
        print ("No graph or Cornerstones\n")
        answer_list=[]
        pickle.dump(answer_list,open(f5,'wb'))
        return
    
    '''
        keep=5
        corner={}
        P=set()
        for v in corner1:
            corner[v]=corner1[v]
            P.add(corner1[v])
            if len(P)==keep:
                break
    '''
    
    corner={}
    for v in corner1:
    	if ' | ' in corner1[v]:
    		corner[v]=corner1[v].split(' | ')
    	else:
    		corner[v]=[corner1[v]]
    
    #corner=corner1
    #if verbose:
    print ("\n\nSize of the unconverted directed graph, Number of cornerstones ",len(G1.nodes()),len(G1.edges()),len(corner))
    #G=G1.to_undirected() #make QKG Undirected
    #G=directed_to_undirected(G1)
    G=G1
    
    #G2=directed_to_undirected(G1)
    #print "\n\nSize of the converted undirected graph, Number of cornerstones ",len(G2.nodes()),len(G2.edges()),len(corner),corner
    #G=max(nx.connected_component_subgraphs(G2), key=len)
    
    #if verbose:
    print ("\n\nSize of the converted undirected graph largest connected component, Number of cornerstones ",len(G.nodes()),len(G.edges()),len(corner),corner)
    
    T, Q=initialize_queue(G,corner)
    P=set() #Entire query
    for v in corner:
        for v1 in corner[v]:
            P.add(v1)
    #for v in corner:
    #    P.add(corner[v])
    if verbose:
        print (P)
    
    count={}
    for v in corner:
        for v1 in corner[v]:
            if v1 not in count:
                count[v1]=set()
            count[v1].add(v)
    #for v in corner:
    #    if corner[v] not in count:
    #        count[corner[v]]=set()
    #    count[corner[v]].add(v)
    
    if verbose:
        print ("\n\nCorner stone count per query term -->", count		)
    
    GST_set=get_GST(Q,T,P,G,no_GST,corner,verbose)
    with open(f5+'_GSTs','wb') as f:
        pickle.dump(GST_set, f)
    '''
        answer=open(f3,'r')
        for line in answer:
            ans=line.strip()
    '''
    
    #answer_type=open(f4,'r')
    ans_type=set()
    #for line in answer_type:
    #ans_type.add(line.strip())
    
    match_flag=0
    unique_nodes=set()
    tree_pot={} #trees with potential answers
    tc=0
    ans_tree=[]
    
    if Type_Filtering_flag==1:
        candidate_set=set()
        #Check number of candidate answers
        for (v,T) in GST_set:
            for n in T.nodes():
                nn=n.split(':')
                if nn[1]=='Entity' and n not in corner: #Non entities and Cornernerstones are removed
                    candidate_set.add(n)
        if len(candidate_set)>=10:
            type_threshold=threshold
        else:
            type_threshold=-1	
    else:
        if Type_Filtering_flag==0:
            type_threshold=-1
        else:
            type_threshold=threshold
    
    '''
    rank_doc=[]
    for index in range(0,len(GST_set)):
        T=GST_set[index][1]
        aa_flag=0
        for n in T.nodes():
            nn=n.split(':')
            for gt in gt1:
                if len(nn)>0:
                    if nn[0]==(gt.lower()):
                        aa_flag=1
                else:
                    if n==(gt.lower()):
                        aa_flag=1
        rank1=index+1
        rank_doc.append((rank1,aa_flag))
    
    reslist=get_rank_docs(rank_doc)	
    print "GST RANKS DOCS ",reslist
    '''
	
	
    for (v,T) in GST_set:
        #if verbose:
        #print "\n\nObtained GST-->",tc,v.encode('utf-8'),T.nodes(),T.edges(),get_cost(T,G)
        #for (n,m) in T.edges():
        #data1=G.get_edge_data(n,m)
        #print "Edges ",n,m,data1['weight'],data1['etype'],data1['wlist']
        #print "\nPotential answers filtered by type -->"
        
        tree_pot[(v,T)]=set()
        for n in T.nodes():
            nn=n.split(':')
            for gt in gt1:
                if len(nn)>0:
                    if nn[0]==(gt.lower()): #.decode('utf-8'):
                        match_flag=1
                else:
                    if n==(gt.lower()): #.decode('utf-8'):
                        match_flag=1
            if nn[1]=='Entity' and n not in corner: #Non entities and Cornernerstones are removed 
                type_chk=get_type_simi(n,G,ans_type,gdict)
                if verbose:
                    print ("\nType match ", n,type_chk)
                if type_chk>=type_threshold: #(len(ans_type)>0 and type_chk>=0.5) or (len(ans_type)==0 and type_chk>=-1):       #Answer type non matching are removed
                    tree_pot[(v,T)].add(n)
                    #if verbose:
                    #print n.encode('utf-8')
        '''
        for n in T.nodes():
            unique_nodes.add(n.encode('utf-8'))
            nn=n.split(':')
            if nn[0]==ans:
                ans_tree.append(tc)
        ''' 
        tc+=1
    if verbose:
        print ("\n\ntrees before merging --->",tree_pot)
    
    #Find unique sets of answers
    ans_set=set()
    done=set()
    for (v1,T1) in tree_pot:
        for n1 in tree_pot[(v1,T1)]: 
            if n1 not in done:
                curr=set()
                done.add(n1)
                curr.add(n1)
                #q = Queue.Queue()
                #q.put(n1)
                #while not q.empty():
                #	nn=q.get()
                #	curr.add(nn)
                #	done.add(nn)
                #	if nn in match:
                #		for n2 in match[nn]:
                #			if n2 not in curr and n2 not in q.queue:	
                #				q.put(n2)
                ans_set.add(frozenset(curr))
    answer_list,candidate_match_flag1,top_match_flag1=get_corner_wt(ans_set,tree_pot,G,corner,gt1)
    str1=f5+'_node_wt'
    pickle.dump(answer_list,open(str1,'wb'))
    answer_list,candidate_match_flag2,top_match_flag2=get_tree_cost(ans_set,tree_pot,G,corner,gt1)
    str1=f5+'_tree_cost'
    pickle.dump(answer_list,open(str1,'wb'))
    answer_list,candidate_match_flag3,top_match_flag3=get_tree_count(ans_set,tree_pot,G,corner,gt1)
    str1=f5+'_tree_count'
    pickle.dump(answer_list,open(str1,'wb'))
    
    answer_list,candidate_match_flag4,top_match_flag4=get_corner_dist(ans_set,tree_pot,G,corner,gt1)
    str1=f5+'_corner_dist'
    pickle.dump(answer_list,open(str1,'wb'))
    
    answer_list,candidate_match_flag5,top_match_flag5=get_corner_dist_wt(ans_set,tree_pot,G,corner,gt1)
    str1=f5+'_corner_dist_wt'
    pickle.dump(answer_list,open(str1,'wb'))
    
    return len(G.nodes()),len(G.edges()), match_flag, candidate_match_flag1, top_match_flag1, candidate_match_flag2, top_match_flag2, candidate_match_flag3, top_match_flag3, candidate_match_flag4, top_match_flag4, candidate_match_flag5, top_match_flag5
	
if __name__ == "__main__":
    main(sys.argv)		
	
