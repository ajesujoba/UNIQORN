#here is the code to extract triples and hearst from snippets either during training or testing
#from .hearstPatterns.hearstPatterns import HearstPatterns
from hearst_patterns_python.hearstPatterns.hearstPatterns import HearstPatterns
import string
import re
import spacy
from spacy.tokens import Span
from spacy.matcher import Matcher
#from BERT.sp_classifier import evaluate as ev
nlp = spacy.load('en_core_web_lg')
punc = '''!()-[]{};:'"\, <>./?@#$%^&*_~'''
nlp.max_length = 10000000000
pronoms = ['he','him','his','she','her','hers']
from string import punctuation
#import neuralcoref
#neuralcoref.add_to_pipe(nlp)

h = HearstPatterns(extended=True)
#h.__spacy_nlp =  nlp
#print(h.__spacy_nlp)
#h.__spacy_nlp.max_length = 10000000000


def remove_punct(text):
    return text.translate(str.maketrans('', '', string.punctuation))

#tokenize, remove stop words and remove punctuation in the question
def process_questions(my_doc):
    text_tokens = [token.text.lower() for token in my_doc]
    tokens_without_sw = [word.strip() for word in text_tokens if nlp.vocab[word].is_stop == False and word not in punc]
    #text_tokens = word_tokenize(quest.lower()) 
    #tokens_without_sw = [word for word in text_tokens if not word in stopwords.words('english') and not word in punc]
    return tokens_without_sw 

#tokenize the text / document
def tokenize_text(my_doc):
    text_tokens = [token.text.lower().strip() for token in my_doc]
    #text_tokens = word_tokenize(text.lower())
    return text_tokens

def truecase(doc):
    tagged_sent = [(w.text, w.tag_) for w in doc]
    normalized_sent = [w.capitalize() if t in ["NN","NNS",'NNP'] else w for (w,t) in tagged_sent] #truecase every noun
    normalized_sent[0] = normalized_sent[0].capitalize()
    string = re.sub(" (?=[\.,'!?:;])", "", ' '.join(normalized_sent))
    return string


def getIndex(context , word):
    indexes = [i for i,x in enumerate(context) if x == word]
    return indexes

def questionWordIndex(context, qw):
    connerstone_pos = [getIndex(context , word) for word in qw]
    return connerstone_pos


#to extract the predicate from sentences/snippets 
def getpredicates(doc):
    prespan = []
    entity_patterns = [[{'POS':'VERB'}], [{'POS':'VERB'},{'POS':'ADP'}],[{'POS':'NOUN'},{'POS':'ADP'}] ]
    #,[{'POS':'NOUN'},{'POS':'ADP'},{'POS':'NOUN'}]
    matcherx  =  Matcher(nlp.vocab)
    for i in range(len(entity_patterns)):
        matcherx.add(str(i),None,entity_patterns[i])
    doc_entity = []
    matches = matcherx(doc)
    prespan = [(start, end) for match_id, start, end in matches]
    return prespan

#merge overlappping tuples, for example [(1,6), (2,4),(7,10)] would become [(1,6),(7,10)]
def merge(times):
    if times==None or times == []:
        return []
    saved = list(times[0])
    for st, en in sorted([sorted(t) for t in times]):
        if st <= saved[1]:
            saved[1] = max(saved[1], en)
        else:
            yield tuple(saved)
            saved[0] = st
            saved[1] = en
    yield tuple(saved)


#get named entities from the document , avoid just numbers, ordinals and cadinals
def getNERspan(docx):
    blacklist = ['NUM','ORDINAL','CARDINAL']
    nerspan = []
    for ent in docx.ents:
        #print(ent)
        if ent.label_ not in blacklist:
            prev_token = docx[ent.start - 1]
            #print(prev_token)
            i = 1
            #for the named entity, if the preceeding words are Adjective, propernpoun, noun or number append them
            while prev_token.pos_ in ['ADJ','PROPN','NOUN','NUM'] and max(ent.start - i,-1) >= 0:
                #print(i)
                i = i + 1
                prev_token = loopback(docx,ent.start,i)
            new_ent = Span(docx, ent.start - i+1, ent.end, label=ent.label)
            nerspan.append((new_ent.start, new_ent.end))
    return nerspan

#this function enables getNERspan to look back to the left of the entity extarcted with NER
def loopback(doc, current_ent,i):
    prev_token = doc[current_ent - i]
    return prev_token

def checkoverlap(idx, test_list2):
    #print(idx)
    for idy in test_list2:
        if idx[0] >= idy[0] and idx[0] < idy[1]:
            #print(idx,idy,True)
            return True
    return False

def getentitiesPOSspan(doc):
    nerspan = []
    entity_patterns = [[{'POS':'PROPN'},{'POS':'PROPN'}], [{'POS':'PROPN'},{'POS':'PROPN'},{'POS':'PROPN'}],[{'POS':'PROPN'},{'POS':'PROPN'},{'POS':'PROPN'}],[{'POS':'NOUN'},{'POS':'NOUN'}], [{'POS':'NOUN'},{'POS':'NOUN'},{'POS':'NOUN'}],[{'POS':'NOUN'},{'POS':'NOUN'},{'POS':'NOUN'}], [{'POS':'ADJ'},{'POS':'NOUN'}],[{'POS':'ADJ'},{'POS':'PROPN'}],[{'POS':'NUM'},{'POS':'NOUN'}],[{'POS':'NUM'},{'POS':'PROPN'}],[{'POS':'NUM'},{'POS':'ADJ'},{'POS':'NOUN'}],[{'POS':'NUM'},{'POS':'ADJ'},{'POS':'PROPN'}],[{'POS':'ADJ'},{'POS':'NUM'},{'POS':'NOUN'}],[{'POS':'ADJ'},{'POS':'NUM'},{'POS':'PROPN'}],[{'POS':'PROPN'}]]
    matcherx  =  Matcher(nlp.vocab)
    for i in range(len(entity_patterns)):
        matcherx.add(str(i),None,entity_patterns[i])
    doc_entity = []
    matches = matcherx(doc)
    nerspan = [(start, end) for match_id, start, end in matches]
    return nerspan


def parsetextspacy(strs):
    return nlp(strs)

#given a document, extract all the entity span and predicate span 
def extractallEPs(docx):
    #docx = nlp(strs)
    entity_indexes_span = []
    doctoken = [t.string.lower().strip() for t in docx]
    extracted_entities_span  = []

    justnp = getNERspan(docx)
    extracted_entities_span.extend(getNERspan(docx))
    extracted_entities_span.extend(getentitiesPOSspan(docx))
    extracted_entities_span = sorted(extracted_entities_span, key=lambda tup: tup[0])
    if len(extracted_entities_span) > 0:  
        entity_indexes_span = list(merge(extracted_entities_span))
    
    extracted_predicates_span  = []
    predicates_indexes_span = []
    extracted_predicates_span.extend(getpredicates(docx))
    #print(entity_indexes_span)
    extracted_predicates_span = sorted(extracted_predicates_span, key=lambda tup: tup[0])
    if len(extracted_predicates_span) > 0:
        #remove redicate  spans that overlap with an  entity
        res = [checkoverlap(idx, entity_indexes_span) for idx in extracted_predicates_span] 
        index_span = [i for i, x in enumerate(res) if x]
        predicates_indexes_span = [e for i, e in enumerate(extracted_predicates_span) if i not in index_span]
        #merge overlapping predicate spans seems useless
        predicates_indexes_span = list(merge(predicates_indexes_span))
       
    #print(entity_predicates_span)
    return entity_indexes_span, predicates_indexes_span,justnp

def getEntityWithinSpan(cspan, entity_indexes_span):
    x = [tpl for tpl in entity_indexes_span if cspan[0]<=tpl[0] and cspan[1]>=tpl[1]]
    return x

def getPredicatesWithinSpan(cspan, entity_predicates_span):
    y = [tpl for tpl in entity_predicates_span if cspan[0]<=tpl[0] and cspan[1]>=tpl[1]]
    return y




def getthetriples(cspan,docx,entity_indexes_span, entity_predicates_span):
    #given a span range this is cspan, this method extracts precicates and entities within this span
    x = [tpl for tpl in entity_indexes_span if cspan[0]<=tpl[0] and cspan[1]>=tpl[1]]
    y = [tpl for tpl in entity_predicates_span if cspan[0]<=tpl[0] and cspan[1]>=tpl[1]]
    i = 0
    j = 0
    result = list()
    triples = list()
    #x = entity_indexes_span
    #y = entity_predicates_span
    #print("predicate span =",y)
    
    while i < len(x) - 1 and j < len(y):
        if y[j][0] > x[i][0] and y[j][1] < x[i + 1][1]:# and x[i][1] != y[j][0] and x[i+1][1] != y[j][1]:
            result.append((x[i], y[j], x[i + 1]))
            triples.append((remove_punct(docx[x[i][0]:x[i][1]].text),remove_punct(docx[y[j][0]:y[j][1]].text),remove_punct(docx[x[i + 1][0]:x[i + 1][1]].text)))
            j += 1
        else:
            i += 1
    #print('result = ',result)
    return triples

def getHearst(sts):
    hyps1 = h.find_hyponyms(sts)
    return hyps1

def getHearstfromtext(strs):
    try:
        hearst = getHearst(strs)
    except IndexError:
        hearst = []
    return hearst


def getthehearsts(cspan,docx):
    #print(cspan)
    #print((docx[cspan[0]:cspan[1]].text))
    #given a span range this is cspan, this method extracts precicates and entities within this span
    return getHearst(docx[cspan[0]:cspan[1]].text.strip())
    #return None




def pronoun_coref(doc):
    pronouns = [(tok, tok.i) for tok in doc if (tok.tag_ == "PRP") or tok.text=='him' or tok.text =='her'  or tok.text=='his' or tok.text =='hers']
    pronouns1 = [(tok, tok.i, tok.tag_) for tok in doc]
    #print(pronouns)
    pronouns = [(i,j) for (i,j) in pronouns if str(i).lower() in pronoms]
    #print(pronouns)
    names = [(ent.text, ent[0].i) for ent in doc.ents if ent.label_ == 'PERSON']
    doc = [tok.text_with_ws for tok in doc]
    for p in pronouns:
        replace = max(filter(lambda x: x[1] < p[1], names),key=lambda x: x[1], default=False)
        if replace:
            replace = replace[0]
            if doc[p[1] - 1] in punctuation:
                replace = ' ' + replace
            try:
                if doc[p[1] + 1] not in punctuation:
                    replace = replace + ' '
                    doc[p[1]] = replace
            except IndexError:
                continue
    doc = ''.join(doc)
    return doc



def corefer(doc):
    if doc._.has_coref:
        return doc._.coref_resolved
    else:
        return doc.text


def createdict(x,y,example):
    #x : a list of tuple
    #y : a list of numbers
    #example : a dictionary with tuple as key and list of numbers as values
    for i in range(len(x)):
        for j in x[i]:
            example.setdefault(j, []).append(y[i])
    return example



def createdict2(x,y,example):
    #x : a list of tuple
    #y : a list of numbers
    #example : a dictionary with tuple as key and list of numbers as values
    for i in range(len(x)):
        example.setdefault(x[i], []).append(y)
    return example

def createdict3(x2,y,example):
    #x : a list of tuple
    #y : a list of numbers
    #example : a dictionary with tuple as key and list of numbers as values
    k = x2[0]
    x = x2[1]
    for i in range(len(x)):
            example.setdefault((k,x[i]), []).append(y)
    return example






#def getBERTmodel():
#    MODEL = ev.getBERTmodel()
#    return MODEL








def getsimilarity(q1,q2,model):
    return sentence_pair_prediction(q1,q2,model)

def getspantext(cspan,docx):
    return (docx[cspan[0]:cspan[1]].text.strip())


import config
import copy
import dataset
import torch
from model import BERTBaseUncased
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # torch.device("cuda")

def getBERTmodel():
    MODEL = BERTBaseUncased()
    #MODEL = nn.DataParallel(MODEL)
    #model_old = torch.load('model15105e5.bin')
    model_old = torch.load('./BERT/model3.bin', map_location=device)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in model_old.items():
        name = k[7:] # remove `module.`
        if name == 'bert.embeddings.position_ids':
            continue

        new_state_dict[name] = v
        # load params
    MODEL.load_state_dict(new_state_dict)
    
    #MODEL.load_state_dict(torch.load(config.MODEL_PATH),strict=False)
        #MODEL.load_state_dict(copy.deepcopy(torch.load(config.MODEL_PATH,device)))                                                                                   
    #MODEL.to(device)'''
    MODEL.eval()
    return MODEL

def sentence_pair_prediction(q1,q2,model):
    tokenizer = config.TOKENIZER
    max_len = config.MAX_LEN
    q1 = str(q1)
    q2 = str(q2)
    inputs = tokenizer.encode_plus(q1,q2,add_special_tokens = True, max_length=max_len, padding='longest',truncation=True)
    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]
    padding_length = max_len - len(ids)
    ids = ids + ([0] * padding_length)
    mask = mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([0] * padding_length)
    #unsqueeze to make a batch of 1
    ids = torch.tensor(ids,dtype=torch.long).unsqueeze(0)
    mask = torch.tensor(mask, dtype = torch.long).unsqueeze(0)
    token_type_ids = torch.tensor(token_type_ids, dtype = torch.long).unsqueeze(0)
    #'targets' : torch.tensor(self.target[item], dtype=torch.long)}
    ids = ids.to(device, dtype = torch.long)
    token_type_ids = token_type_ids.to(device, dtype = torch.long)
    mask = mask.to(device, dtype = torch.long)
    #targets = targets.to(device, dtype = torch.float)
    outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
    outputs =  torch.sigmoid(outputs).cpu().detach().numpy()
    return outputs[0][0]



def co_occur2(doc,entity,predicates):
    #extract triples refer to co-occur if entity exist in the form  ((2,3),(3,4)), then they co-occur. If (2,3),(4,5) and no preficate exist between them, then they co-occur
    lst = []  

    for i in range(len(entity)):
        if i+1 < len(entity):
            if abs(entity[i + 1][0] - entity[i][1])>1:
                continue
            context = entity[i],entity[i+1]
            cond = True
            if abs(context[1][0] - context[0][1])==0:
                lst.append((remove_punct(getspantext(context[0],doc)),'cooccur', remove_punct(getspantext(context[1],doc))))
            '''
            
          for j in predicates:
                #if j[0]<context[1]:
                #break
                if abs(context[1][0] - context[0][1])==0:
                    cond = cond and True
                elif abs(context[1][0] - context[0][1])==1:
                    if j[0]>=context[0][1] and j[0]<context[1][0]:
                        cond = cond and True
                        #else:
                        #cond = cond and False
                else:
                    cond = cond and False
            if cond == True:   
                lst.append((remove_punct(getspantext(context[0],doc)),'cooccur', remove_punct(getspantext(context[1],doc))))
    #print(lst)'''
    return lst
                                                                                                                                            
                                                                                                                                                 
def co_occur(doc,entity,predicates):
    lst = []
    for i in range(len(entity)):
        if i+1 < len(entity):
            context = entity[i],entity[i+1]
            if abs(entity[i + 1][0] - entity[i][1])>1:
                continue
            elif abs(entity[i + 1][0] - entity[i][1])==0:
                lst.append((remove_punct(getspantext(context[0],doc)),'cooccur', remove_punct(getspantext(context[1],doc))))
                #lst.append((context[0],'cooccur', context[1]))
            else:
                cond = True
                for j in predicates:
                    if j[0]>=context[0][1] and j[0]<context[1][0]:
                        cond = cond and False
                if cond == True:
                    #lst.append((context[0],'cooccur', context[1]))
                    lst.append((remove_punct(getspantext(context[0],doc)),'cooccur', remove_punct(getspantext(context[1],doc))))

    return lst


#get tokenized version of the text - paragraphs
#text_tokens = tokenize_text(text_doc)

#get tokenized question without 
#questiontokens = process_questions(questn_doc)


#get the question tokens in the passage
#question_index = list(filter(None,questionWordIndex(text_tokens, questiontokens))) #get the index of the question word in the 
#flattened_list = [y for x in question_index for y in x]
#indexes = [ (max(0,x-spanvalue-1),x+spanvalue+1) for x in flattened_list]
#indexes = sorted(indexes, key=lambda tup: tup[0])
#indexes = list(merge(indexes))

#from all the indexes extract triples
#triples = [getthetriples(indexpair, entity_indexes_span, entity_predicates_span) for indexpair in indexes]
