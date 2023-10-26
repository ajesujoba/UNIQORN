#For every positive class example for a particular question
# Get a 5 negative examples whose predicate are different
import sys
import time
import pandas as pd
from multiprocessing import Pool

def sample_negatives(row):
    #print(row[0])
    global data1,n
    allcom = []
    questionid = row[1]['questionId']
    label = row[1]['label']
    pred = row[1]['predicate']
    #
    therows = data1[(data1['questionId']==questionid) & (data1['predicate']!=pred) & (data1['label']==0)][['questionId','NEDEntityId','TripleEntities','subject','predicate','question','context','label']]
    xk = min(n,len(therows.index))
    the50 = therows.sample(xk)
    #allcom.append(the50)
    #print('The Length = ',len(allcom))
    return the50

def main():
    global data1,n
    inputfile = sys.argv[1] # the file containing all the positve and negative examples extracted (kgexample.csv )
    n = int(sys.argv[2]) # number of negative examples
    outputfile = sys.argv[3] # file that should contain the extracted examples

    data1 = pd.read_csv(inputfile)
    count = data1.groupby(['label']).count()
    print(data1)
    allpositive = data1[data1["label"] == 1]
    index = allpositive.index
    number_of_rows = len(index)
    print('all positive example = ', number_of_rows)
    
    allcombined = [allpositive]
    
    p = Pool(10)
    
    start = time.time()
    #allcombined.extend(p.map(sample_negatives, allpositive[0:400000].iterrows()))
    allcombined.extend(p.map(sample_negatives, allpositive[1:20].iterrows())) # if you have a lot of data, this line could crash, you can find a way to run this in batches on the subset of posiitve examples
    #print(allcombined)
    print('The Length after all = ',len(allcombined))
    result = pd.concat(allcombined)
    #print(result)
    #result.to_csv(r'posne_samples_mp_nopred504.csv', index = False)
    result.to_csv(outputfile, index = False)
    print("Done")
    print(time.time()-start)
    #posne_samples_mp_nopred548.csv   > predicate 
    # #posne_samples_mp_nopred548.csv   > titan 1
if __name__ == '__main__':
    main()
