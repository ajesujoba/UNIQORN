#import json library
# extarct all possible triples that has their answers and those that don't (positive and negative examples). This extract as many as exist in the training data
import json
import sys
import pickle
# importing NLTK libarary stopwords 
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import re
from itertools import combinations 
from itertools import product 
from itertools import chain
import pickle
import time
import numpy as np
import pandas as pd
#punc = ["!","(",")","-","[","]","{","}",";",":","'","\"","\\",","<",">",".","\/","?","@","#","$","%","^","&","*"_~]
punc = '''!()"-[]{};:'"\, <>./?@#$%^&*_~`'''
# word pattern
predicate_pattern   = re.compile('^P[0-9]*$')
entity_pattern      = re.compile('^Q[0-9]*$')

def main():
    trainfile = sys.argv[1] # the file containing the question and answes
    tripledir = sys.argv[2] # the directory to the triples extracted from CLOCQ
    outputfile = sys.argv[3] # the file to store the psotive and negative examples
    startindext = int(sys.argv[4])
    endindex =  int(sys.argv[5])

    questionsId = []
    questions = []
    answers = []
    qno = []
    for line in open(trainfile,'r'): #'/GW/qa2/work/uniqorn/Crawl_Google_docs/LCQuAD_2/LcQUAD2.0_data/train.json','r'):
        data = json.loads(line)
        if data['answers'] == None or data['answers']==True or data['answers']==False:
            continue
        qno.append(data['id'])
        # /GW/qa2/work/uniqorn/Crawl_Google_docs/LCQuAD_2/KGmode/hdt/WikiTriplesUniqorn2
        questionsId.append(tripledir+"/SPO_"+data['id']+".pickle")
        questions.append(data['question'])
        if type(data['answers'][0]) == list:
            if type(data['answers'][0]) == str:
                #print("String")
                answers.append(' '.join(data['answers']))
            elif type(data['answers'][0]) == list or type(data['answers'][0]) == tuple:
                flat_list = [item for sublist in data['answers'] for item in sublist]
                answers.append(' '.join(flat_list))
                #print(' '.join(flat_list))
                #time.sleep(1)
        else:
            answers.append(' '.join(data['answers']))
            #print(' '.join(data['answers']))
            #time.sleep(1)
    countnop = 0
    countnoc = 0
    foundans = 0
    print("Got IDs for ", len(questions), " and ", len(answers) , " train questions ")
    
    positive_ex = []
    for i in range(startindext,endindex):#len(questions)):
        print(i)
        questn = questions[i]
        #print(questions[i])
        try:
            kg = read_kg(questionsId[i])
        except IOError:
            countnop = countnop + 1
            continue
        
        if len(kg) <= 0:
            print("Text length not up content, Continue ....., actual length = ",len(kg))
            countnoc =  countnoc + 1
            continue
        answerr = answers [i]
        #print(answerr)
        answertokens = answerr
        
        #if I am using the whole answer span, I dont need the answer token
        #answertokens = list(set(process_questions(answerr)))
        
        #print(answerr)
        #print(answertokens)
        for keys in kg:
            texts = gettriplesforEntity(kg, keys)
            for tiples,ent in texts:
                #print(tiples)
                entities,sp,pr = ent
                allent = '|'.join(entities)
                #tiptokens = list(set(process_questions(tiples)))
                tiptokens = [x.lower().strip() for x in tiples] # use this instead of tokens
                #print(tiples)
                tiples2 = ' '.join(tiples)
                if bool(set(tiptokens) & set(answertokens)) == True:
                    #this is a positive example
                    positive_ex.append((qno[i],keys,allent,sp,pr, questn, tiples2, 1))
                else:
                    #negative example
                    positive_ex.append((qno[i],keys,allent,sp,pr, questn, tiples2, 0))
    df = pd.DataFrame(positive_ex, columns=['questionId','NEDEntityId','TripleEntities','subject','predicate','question', 'context', 'label'])
    df.to_csv(outputfile, index = False)


#read the file containing the paragraphs
def read_kg(dirs):
    verbfacts = []
    with open(dirs, 'rb') as fp:  
        triples2 = pickle.load(fp)
    return triples2

def gettriplesforEntity(triples2, key):
    #alltriples =  [triples2[t] for t in triples2]
    verbfacts = []
    flattened_list = triples2[key]
    for t in flattened_list:
        #get all label
        broke = False
        tplabels = [x['label'] for  x in t]
        tpids = [x['id'] for  x in t if entity_pattern.match(x['id'].strip())]
        sp = t[0]['id']
        pr = t[1]['id']
        finallabel =[]
        for x in tplabels:
            if type(x) == list:
                possibleitem = x[0]

                for item in x:
                    possibleitem = x[0].strip('"')
                    if entity_pattern.match(item.strip()):
                        continue
                    elif predicate_pattern.match(item.strip()):
                        continue
                    else:
                        possibleitem = item.strip('"')
                        break
                finallabel.append(possibleitem)

            else:
                broke = True
                #finallabel.append(x.strip('"'))
                break
        #finallabel = [x[-1] if type(x) == list  else x for x in tplabels]
        if broke == False:
            #verbfacts.append((' '.join(finallabel),(tpids,sp,pr)))
            verbfacts.append((finallabel,(tpids,sp,pr)))
    return verbfacts



#tokenize, remove stop words and remove punctuation in the question
def process_questions(quest):
    text_tokens = word_tokenize(quest.lower())
    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words('english') and not word in punc]
    return tokens_without_sw

#tokenize the text / document
def tokenize_text(text):
    text_tokens = word_tokenize(text.lower())
    return text_tokens

def getIndex(context , word):
    indexes = [i for i,x in enumerate(context) if x == word]
    return indexes

def questionWordIndex(context, qw):
    connerstone_pos = [getIndex(context , word) for word in qw]
    return connerstone_pos

def questiontoquestion(connerstone_pos):
    comb = combinations(connerstone_pos, 2) 
    difference = [min(computedifference(x)) for x in comb]
    return difference

def questiontoanswer(comb):
    difference = [min(computedifference(x)) for x in comb]
    return difference

def computedifference(tuplst):
    return list((abs(i-j) for i,j in product(tuplst[0],tuplst[1])))


# Python program to get average of a list 
def Average(lst):
    return sum(lst) / len(lst) 

def answer_present(ans,rang,context):
    #print('answer = ',ans)
    #print(set(context[rang[0]:rang[1]] ))
    return not set(ans).isdisjoint(set(context[rang[0]:rang[1]] ))

def merge(times):
    saved = list(times[0])
    for st, en in sorted([sorted(t) for t in times]):
        if st <= saved[1]:
            saved[1] = max(saved[1], en)
        else:
            yield tuple(saved)
            saved[0] = st
            saved[1] = en
    yield tuple(saved)

if __name__ == "__main__":
    main()
