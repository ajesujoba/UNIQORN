write_dir: ./../Files/Results/
read_KGfiles: ./../../KG/Files/
read_TEXTfiles: ./../../TEXT/Files/
alignDir: ./../Files/HeterAlignments2

#SPO option, 0 means only nearest S and O; 1 means S, O until no intruding predicates, 2 means upto a context length and 3 means all possible SPOs
SPO_option: 1
context_length: 10

#Number of GSTs to be considered 
n_GST: 10

#Word Embedding for predicates and types 
embedding: WIKI

#Prune the predicate terms from the query if the query has more than 'prune' number of terms
prune: 5

#To print intermediate statements in verbose manner make it 1, other wise 0  
verbose: 0

benchmark2: LCQUAD_TEXT

#Corpora can be top10, strata1, strata2, strata3, strata4, strata4, strata5
corpora: WIKIDATA

stem_pred: 0

degenerate: 0
connectseed: 1
createtable: 0
Add_Type: 1
computeAlign: 0


#Use 0 if you do not want to use types from dbpedia using tagme; use 1 otherwise. If it is 1, tagme types with rho >=Wiki_Threshold will be used 
Wikitype: 0
Wiki_Threshold: 0.25

#Draw alignment edges between type nodes, 0 means no, 1 means yes
Type_Alignment: 1
Predicate_Alignment: 1
Entity_Alignment: 1

#Distribute node weights to edge weights; 0 means no distribution, 1 means using joint optimization, 2 means adding half of endnode weights to the edge-weight
Distribute_Node_wt: 0

#Type filtering can be none (0), Relaxed i.e. only apply if candidate list greater than 10 (1), or Strict i.e. apply always (2)
Type_Filtering: 0

#Embedding similarity can be avg of vectors (0) or maximum match (1), Cosine_threshold gives the corresponding threshold
Embedding_Similarity: 1
Cosine_threshold: 0.9

#Threshold for entity similarity
jackard_threshold: 0.5

#Use 0 if roles need not be matched while checking for cornerstones, otherwise 1
Cornerstone_Matching: 0

#Use 0 if no chain join type inclusion of entities, use 1 otherwise
chain_join_flag: 1
