import requests
import pickle
import json
import re
#import urllib2
import urllib.request as urllib2
from bs4 import BeautifulSoup
import signal
import time

import sys
_RE_COMBINE_WHITESPACE = re.compile(r"\s+")

def writefile(data_file, outjson, id='test_1'):
    fp = open(outjson+id+'.json','w')
    for i in range(len(data_file['paragraphs'])):
        fp.write(json.dumps({'text':process_text(data_file['paragraphs'][i]),'id':id,'link':data_file['links'][i]}))
        fp.write('\n')
    fp.close()

def process_text(text):
    sample_text = BeautifulSoup(text, "lxml").text
    word = re.sub(r'\n+', '\n', sample_text).strip()
    word = re.sub(r'\t+', '\t', word).strip()
    #word = re.sub(r'(?!\n)\s+',' ',word).strip()
    word = re.sub(r'\n+', '\n', word).strip()
    #word = re.sub(r'\s+', ' ', word).strip()

    word = _RE_COMBINE_WHITESPACE.sub(" ", word).strip()
    return word

def test_request(arg=None):
    """Your http request."""
    time.sleep(2)
    return arg
 
class Timeout():
    """Timeout class using ALARM signal."""
    class Timeout(Exception):
        pass
 
    def __init__(self, sec):
        self.sec = sec
 
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.raise_timeout)
        signal.alarm(self.sec)
 
    def __exit__(self, *args):
        signal.alarm(0)    # disable alarm
 
    def raise_timeout(self, *args):
        raise Timeout.Timeout()


def visible(element):
	if element.parent.name in ['style', 'script', '[document]', 'head', 'title']:
		return False
	elif re.match('<!--.*-->', str(element.encode('utf-8'))):
		return False
	return True

def process(result):
	result1=''
	for elem in result:
		if elem.count(' ') <= 1:
			continue
		elem=elem.strip('\n')
		elem=elem.strip('\t')
		elem=elem.strip('\r')	
		result1+=' '+elem
	return result1
		
#links=pickle.load(open('google_search_links','r'))

def getQuestion(qir):
    question = {}
    qid =  []
    for line in open(qir,'r'):
        data = json.loads(line)
        if data['answers'] == None or type(data['answers']) == bool:
            continue

        question[data['id']]={}
        qid.append(data['id'])

        question[data['id']]['text']=data['question']
        #print(data['answers'])
        if data['answers'] == None:
            continue
            question[data['id']]['GT']=set([None])
            continue

        if type(data['answers']) == bool:
            continue
            question[data['id']]['GT']=set([data['answers']])
            continue

        if type(data['answers']) == list and type(data['answers'][0]) ==  list:
            question[data['id']]['GT']=set([y for x in data['answers'] for y in x])
            continue

        question[data['id']]['GT']=set(data['answers'])

    return qid, question

'''
fp=open('LCQuAD_link_wise_docs.json','r')

done_links=set()
ct=0
for line in fp:
	ct+=1
print ct
fp.close()
'''
#REMEMBER TO REMOVE THE LAST LINE IF RESTARTED

#REMEMBER TO REMOVE THE LAST LINE IF RESTARTED

#REMEMBER TO REMOVE THE LAST LINE IF RESTARTED

'''
fp=open('LCQuAD_link_wise_docs.json','r')	
for line in fp:
	l=json.loads(line.strip())
	ct-=1
	#print ct
	link=l['link']
	doc=l['doc']
	done_links.add(link)
	if ct==0:
		break
print len(done_links)

fp.close()
print "done links ",len(done_links)
'''

def main():

    inputfile = sys.argv[1]
    outputjson = sys.argv[2]
    filejson = inputfile 
    links={}
    done_links=set()

    _, qid = getQuestion(filejson)
    #links=pickle.load(open('google_search_links_LCQuAD','r'))
    #print "Done links, questions ",len(done_links),len(links)
    # fp=open(outputjson,'a+') #MAKe it a+ if restarted
    #not_crawled=pickle.load(open('Not_crawled_ques','r'))
    #with open(inputfile) as filex:
    #not_crawled = [line.strip() for line in filex]
    
    line_count=0
    #for q in links:
    for line in qid:#open('LC-QuAD_2_Selected_Ques_New1.txt'):
        # fp=open(outputjson,'a+') #MAKe it a+ if restarted
        line1=qid[line]['text'].strip()
        line_count+=1
        print ("\n===============================================================>\n")
        print ("Current line ",line1)
        q=line1
        if q in links:
            continue
        links[q]=[]
        start=1
        print ("Crawling for ---> ",q)
        crawled=0
        docst =[]
        paras = []
        para_links = []
        while crawled<10 and start<=31:
            qflag=0
            query='https://www.googleapis.com/customsearch/v1?key=<INSERT YOUR KEY HERE>&q='+q+'&start='+str(start)
            try:
                with Timeout(10):
                    result = requests.get(query)
                    r=result.json()
                    qflag=1
            except Timeout.Timeout:
                print ("\n\nTimeout Google\n\n")
            except:
                print ("\n\nCould not crawl google... ",start,q)		
            
            print ("done link")
            ct=0
            if qflag==1 and 'items' in r and r['items'][0]['kind']!='':
                for doc in r['items']:
                    print ("link  ",start+ct, doc['link'])
                    if doc['link'] not in done_links:	
                        flag=0
                        try:
                            with Timeout(15):
                                html = urllib2.urlopen(doc['link']).read() #urllib.request.urlopen(link)
                                flag=1
                        except Timeout.Timeout:
                            print ("\n\nTimeout")
                        except:
                            print ("\n\nCould not crawl ... ",start+ct,doc['link'])
                        if flag==1:
                            done_links.add(doc['link'])			
                            links[q].append(doc['link'])
                            crawled+=1
                            soup = BeautifulSoup(html,"lxml")
                            data = soup.findAll(text=True)
                            result = list(filter(visible, data))
                            result1=process(result)
                            
                            #body = soup.find('body')
                            #content = body.findChildren()
                            print ('Successfully crawled link ',start+ct, doc['link'],crawled)
                            #print 'Text ',result1.encode('utf-8')
                            #fp.write(json.dumps({'link':doc['link'],'doc':result1.encode('utf-8')}))
                            #fp.write('\n')
                            docst.append({'link':doc['link'],'doc':result1.encode('utf-8')})
                            paras.append(result1) #.encode('utf-8'))
                            para_links.append(doc['link'])
                            print ("Done writing")
                            if crawled==10:
                                #towrite = {'question':q,'value':docts}
                                #fp.write(json.dumps(towrite))
                                #fp.write('\n')
                                break
                    else:
                        crawled+=1
                        links[q].append(doc['link'])
                    ct+=1
            start+=10
            print ( "Start ",start,crawled)
        #towrite = {'question':q, 'value':docst}
        towrite = {'question':q, 'links':para_links, 'paragraphs': paras}
        writefile(towrite, outputjson, id=line)
        #fp.write(json.dumps(towrite))
        #fp.write('\n')
        #fp.close()

        if line_count%10==0:
            # pickle.dump(links,open('google_search_links_LCQuAD_part4','w'))
            # if you want to keep a log of links
            pass
    #print (links)
    #pickle.dump(links,open('google_search_links_LCQuAD_part4','w'))
if __name__ == '__main__':
    main()
