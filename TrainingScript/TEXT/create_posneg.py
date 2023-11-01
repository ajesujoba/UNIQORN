#import json library
import json
import sys
import pandas as pd
# importing NLTK libarary stopwords 
import argparse
import string
stops = string.punctuation
#import spacy
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
#from spacy.tokens import Span

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
nlp = English()
#nlp = spacy.load('en_core_web_lg')
nlp.max_length = 200000000

from itertools import combinations 
from itertools import product 
from itertools import chain
import pickle

import numpy as np
#punc = ["!","(",")","-","[","]","{","}",";",":","'","\"","\\",","<",">",".","\/","?","@","#","$","%","^","&","*"_~]
punc = '''!()-[]{};:'"\, <>./?@#$%^&*_~'''


#read the file containing the paragraphs
def read_textfile(dirs):
    question = []
    for line in open(dirs,'r'):
        data = json.loads(line)
        question.append(data['text'])
    return question

#tokenize, remove stop words and remove punctuation in the question
def process_questions(my_doc):
    text_tokens = [token.text.lower() for token in my_doc]
    tokens_without_sw = [word.strip() for word in text_tokens if nlp.vocab[word].is_stop == False and word not in punc]
    #text_tokens = word_tokenize(quest.lower())
    #tokens_without_sw = [word for word in text_tokens if not word in stopwords.words('english') and not word in punc]
    return tokens_without_sw

#tokenize, remove stop words and remove punctuation in the question
def process_questions2(quest):
    text_tokens = word_tokenize(quest.lower())
    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words('english') and not word in punc]
    return tokens_without_sw

#tokenize the text / document
def tokenize_text(my_doc):
    text_tokens = [token.text.lower().strip() for token in my_doc]
    #text_tokens = word_tokenize(text.lower())
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

def indexstring(tokens, indexrange):
    span = tokens[indexrange[0]:indexrange[1]+1]
    return span.string



def main(args):
    positive_ex = []
    negative_ex = []
    
    spanvalue = args.spanvalue
    trainfile = args.trainfile
    textdir = args.textdir
    outputfile = args.outputfile

    questionsId = []
    questions = []
    answers = []

    # read the question and passages
    for line in open(trainfile,'r'):
        data = json.loads(line)
        if data['answers'] == None or data['answers']==True or data['answers']==False:
            continue
        questionsId.append(textdir+data['id']+".txt")
        questions.append(data['question'])
        if type(data['answers'][0]) == list:
            if type(data['answers'][0]) == str:
                #print("String")
                answers.append(' '.join(data['answers']))
            elif type(data['answers'][0]) == list or type(data['answers'][0]) == tuple:
                flat_list = [item for sublist in data['answers'] for item in sublist]
                answers.append(' '.join(flat_list))
        else:
            answers.append(' '.join(data['answers']))
    countnop = 0
    countnoc = 0
    foundans = 0
    print("Got IDs for ", len(questions), " dev questions ")
    #questions = questions[0:20]
    #questionsId = questionsId[0:20]
    for i in range(len(questions)):
        questn = questions[i]
        print(questions[i])
        try:
            texts = read_textfile(questionsId[i])
        except IOError:
            countnop = countnop + 1
            continue
        
        if len(texts) <= 0:
            print("Text length not up content, Continue ....., actual length = ",len(texts))
            countnoc =  countnoc + 1
            continue
        
        questn_doc = nlp(questn)
        questiontokens = process_questions(questn_doc)
        #print(questiontokens)
        answerr = answers [i]
        answertokens = list(set(process_questions2(answerr)))
        #print(answertokens)
        
        for t in texts:
            #print(questiontokens)
            text_doc = nlp(t)
            text_tokens = tokenize_text(text_doc)
            #print("text tokens = ",text_tokens)
            #print("anser tokens = ", answertokens)
            question_index = list(filter(None,questionWordIndex(text_tokens, questiontokens)))
            answer_index = list(filter(None,questionWordIndex(text_tokens, answertokens)))
            if len(question_index)<=0:
                continue
            
            if len(question_index)>0 and len(answer_index) <= 0:
                #print("Errorrrrrrrrr!!!!")
                flattened_list = [y for x in question_index for y in x]
                indexes = [ (max(0,x-spanvalue-1),x+spanvalue+1) for x in flattened_list]
                indexes = list(merge(indexes))
                positive_ex.extend([(questn,indexstring(text_doc,(x[0],x[1])),0) for x in  indexes])
                continue
            elif len(answer_index)<=0:
                continue
            
            #print("questiion tokens = ",questiontokens)
            #print(questionWordIndex(text_tokens, questiontokens))
            #print(questiontokens)
            #print("answer tokens = ",answertokens)
            
            flattened_list = [y for x in question_index for y in x]
            indexes = [ (max(0,x-spanvalue-1),x+spanvalue+1) for x in flattened_list]
            indexes = sorted(indexes, key=lambda tup: tup[0])
            indexes = list(merge(indexes))
            #indexes = sorted(indexes, key=lambda tup: tup[0])
            states = [answer_present(answertokens,x,text_tokens) for x in indexes]
            truein = [k for k,v in enumerate(states) if v == True]
            falsein = [k for k,v in enumerate(states) if v == False]
            #
            #positives = 
            positive_ex.extend([(questn,indexstring(text_doc,(indexes[x][0],indexes[x][1])),1) for x in  truein])
            #negatives = 
            positive_ex.extend([(questn,indexstring(text_doc,(indexes[x][0],indexes[x][1])),0) for x in  falsein])
    
    df = pd.DataFrame(positive_ex, columns=['question', 'context', 'label'])
    df.to_csv(outputfile, index = False)
    #print("Found answers for ", foundans, " questions ")
    #print("File without things ", countnoc)
    print("Got IDs for ", len(questions), " dev questions ")
    #print("Percentage covered = ", str((float(foundans)/float(len(questions)))*100))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Your Script Description')
    parser.add_argument('--spanvalue', type=int, default=50, help='An integer representing span value. The default is 50. ')
    parser.add_argument('--trainfile', type=str, help='Path to the training file')
    parser.add_argument('--textdir', type=str, help='Path to a directory containing text files extracted from Google.')
    parser.add_argument('--outputfile', type=str, help='Path to the output csv file')
    args = parser.parse_args()
    main(args)
