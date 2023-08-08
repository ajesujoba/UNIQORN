import argparse

def getarguments():
    # Create the parser
    parser = argparse.ArgumentParser(description='Command Line argument for uniqorn KG setup')
    # Add an argument
    parser.add_argument('--write_dir', type=str, required=True,help='The directory to store the resulting outputs from UNIQORN')
    parser.add_argument('--kgcorpus',type=str, required=True,help='The directory containing the scored facts, types, aliases and seed entity names.')
    parser.add_argument('--n_GST', type=int, default=10, help='The number of GST')
    parser.add_argument('--embedding', type=str, default='WIKI2VEC', help='Word Embedding for predicates and types')
    parser.add_argument('--prune', type=int, default=5, help='Prune the predicate terms from the query if the query has more than \'prune\' number of terms')
    parser.add_argument('--verbose', type=int, default=0, help='To print intermediate statements in verbose manner make it 1, other wise 0')
    parser.add_argument('--benchmark', type=str, default='LCQUAD2', help='To print intermediate statements in verbose manner make it 1, other wise 0')
    parser.add_argument('--corpora', type=str, default='WIKIDATA', help='')
    parser.add_argument('--stem_pred', type=int, default=0, help='Stemming the predicates')
    parser.add_argument('--degenerate', type=int, default=0, help='Degenerate edge weights')
    parser.add_argument('--addType', type=int, default=1, help='Add type nodes')
    parser.add_argument('--connectseed', type=int, default=1, help='Add type nodes')
    parser.add_argument('--Wikitype', type=int, default=0, help='Use 0 if you do not want to use types from dbpedia using tagme; use 1 otherwise.')
    parser.add_argument('--Wiki_Threshold', type=float, default=0.25, help='tagme types with rho >=Wiki_Threshold will be used')
    parser.add_argument('--Type_Alignment', type=int, default=0, help='Draw alignment edges between type nodes, 0 means no, 1 means yes')
    parser.add_argument('--Distribute_Node_wt', type=int, default=0, help='Distribute node weights to edge weights; 0 means no distribution, 1 means using joint optimization, 2 means adding half of endnode weights to the edge-weight')
    parser.add_argument('--Type_Filtering', type=int, default=0, help='Type filtering can be none (0), Relaxed i.e. only apply if candidate list greater than 10 (1), or Strict i.e. apply always (2)')
    parser.add_argument('--Embedding_Similarity', type=int, default=1, help='Embedding similarity can be avg of vectors (0) or maximum match (1)')
    parser.add_argument('--jackard_threshold', type=float, default=0.5, help='Threshold for entity similarity')
    parser.add_argument('--Cornerstone_Matching', type=int, default=0, help='Use 0 if roles need not be matched while checking for cornerstones, otherwise 1')
    #parser.add_argument('--Embedding_Similarity', type=int, default=1, help='Embedding similarity can be avg of vectors (0) or maximum match (1)')
    parser.add_argument('--chain_join_flag', type=int, default=1, help='Use 0 if no chain join type inclusion of entities, use 1 otherwise')
    

    parser.add_argument('--inputfile', type=str, required=True,help='The json file containing the questions.')
    parser.add_argument('--core', type=int, default=8,help='The number of threads to use')
    parser.add_argument('--topk', type=int, default=5,help='The number of facts to use')
    parser.add_argument('--logfile', type=str, required=True,help='Name of file to store the progress')
    # Parse the argument
    argv = parser.parse_args()
    return argv
