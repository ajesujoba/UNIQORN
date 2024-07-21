# code for automatically checking for hallucinations in gpt-4o outputs
# for kg+text, update filename and "evidence" in L47
# for kg, update filename and "triples" in L47
# for text, update filename and "snippets" in L47

import json
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import nltk
nltk.download('stopwords')
nltk.download('punkt')


def load_json(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data


def preprocess_text(text):
    text = text.lower()  # convert to lowercase
    text = re.sub(r'[^\w\s]', ' ', text)  # replace all punctuation with spaces
    text = re.sub(r'\s+', ' ', text)  # normalize whitespace to single space
    text = text.strip()  # remove leading and trailing whitespace
    words = word_tokenize(text)  # tokenize the text
    return words


def check_hallucinations(data):
    stop_words = set(stopwords.words('english'))
    additional_stopwords = {
        'context', 'length', 'exceeded', 'sorry', 'id', 'origin', 'current',
        'specified', 'provide', 'provided', 'given', 'unknown', 'none',
        'explicitly', 'information', 'available', 'mentioned', 'text',
        'specific', 'directly', 'direct', 'nothing', 'producing',
        'index', 'contain', 'containing', 'question', 'answer', 'could',
        'please', 'specify', 'include', 'following', 'unable', 'starting',
        'applicable', 'mention', 'evidence', 'input'
    }
    stop_words.update(additional_stopwords)

    hallucination_entries = []
    
    for entry_id, entry in data.items():
        rag_answer_words = preprocess_text(entry["rag_answer"])
        evidence_words = preprocess_text(entry["snippets"])
        question_words = preprocess_text(entry["question"])
        
        # Remove stopwords
        rag_answer_words = [
            word for word in rag_answer_words 
            if word not in stop_words
        ]
        evidence_words_set = set(evidence_words)
        question_words_set = set(question_words)
        
        # Check for hallucinations
        hallucinations = [
            word for word in rag_answer_words 
            if (word not in evidence_words_set 
                and word not in question_words_set)
        ]
        if hallucinations:
            print(f'Entry ID: {entry_id}, Hallucinations: {hallucinations}')
            hallucination_entries.append(entry_id)
    
    print(f'Total entries with hallucinations: {len(hallucination_entries)}')


# Load JSON data
data = load_json('answers-gpt4o-rag-kg-text.json')
# data = load_json('answers-gpt4o-rag-text.json')
# data = load_json('answers-gpt4o-rag-text.json')

# Perform hallucination check
check_hallucinations(data)
