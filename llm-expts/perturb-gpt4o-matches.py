# code to prepare synthetic data for stress-testing RAG

# important: 
# for kg+text, use 'evidence' in L64 and L68
# for kg, use 'triples' in L64 and L68
# for text, use 'snippets' in L64 and L68

import json
import re
import string


# Function to normalize and tokenize the text
def normalize(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    return tokens


# Function to check if any token from answers is in evidence tokens
def find_and_replace(evidence, answers):
    evidence_tokens = normalize(evidence)
    answer_tokens_list = []
    
    for answer in answers:
        if isinstance(answer, list):
            for sub_answer in answer:
                answer_tokens_list.append(normalize(sub_answer))
        else:
            answer_tokens_list.append(normalize(answer))
    
    # Find matches
    for answer_tokens in answer_tokens_list:
        for token in answer_tokens:
            if token not in stopwords and token in evidence_tokens:
                pattern = r'\b' + re.escape(token) + r'\b'
                evidence = re.sub(pattern,
                                  'Seqret Uniquorn',
                                  evidence,
                                  flags=re.IGNORECASE)
    
    return evidence


# Define a list of common stopwords
stopwords = set([
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'to', 'in', 'of', 'on', 'for',
    'with', 'at', 'by', 'from', 'about', 'as', 'into', 'like', 'through',
    'after', 'over', 'between', 'out', 'against', 'during', 'without', 
    'before', 'under', 'around', 'among'
])

# Load the JSON file
with open('matches-gpt4o-rag-kg-text.json', 'r') as file:
    # with open('matches-gpt4o-rag-kg.json', 'r') as file:
    # with open('matches-gpt4o-rag-text.json', 'r') as file:
    data = json.load(file)

# Process each entry in the JSON file
for entry in data.values():
    answers = entry.get('answers', [])
    evidence = entry.get('evidence', '')
    
    if evidence and answers:
        updated_evidence = find_and_replace(evidence, answers)
        entry['evidence'] = updated_evidence
    
    # Remove the rag_answer field if it exists
    if 'rag_answer' in entry:
        del entry['rag_answer']

# Save the updated JSON to a new file
output_file = 'matches-gpt4o-rag-kg-text-perturbed.json'
# output_file = 'matches-gpt4o-rag-kg-perturbed.json'
# output_file = 'matches-gpt4o-rag-text-perturbed.json'
with open(output_file, 'w') as file:
    json.dump(data, file, indent=4)

print(f"Updated JSON file saved to {output_file}")
