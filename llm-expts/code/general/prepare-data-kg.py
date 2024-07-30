# code to create merged QA data file for kg with questions, gold answers,
# rag contexts (triples) with bert from individual pickle and json files

import pickle
import json
import re


# Paths to the files
pickle_file_path = 'top5-facts-kg.pkl'  
json_file_path = 'dev.json'
merged_file_path = 'merged-file-kg.json' 

# Step 1: Load the pickle file
with open(pickle_file_path, 'rb') as pickle_file:
    pickle_data = pickle.load(pickle_file)

# Check the type of loaded pickle data
print(f'Type of pickle_data: {type(pickle_data)}')
if isinstance(pickle_data, dict):
    print(f'Sample of pickle_data: {list(pickle_data.items())[:3]}')
    # Print first 3 items as a sample

# Step 2: Load the JSON file
with open(json_file_path, 'r', encoding='utf-8') as json_file:
    json_data = [json.loads(line) for line in json_file]

# Convert JSON data to a dictionary with 'id' as the key for faster lookup
json_dict = {item['id']: item for item in json_data}

# Step 3: Merge the data
if isinstance(pickle_data, dict):
    for question_id, entry in pickle_data.items():
        if question_id in json_dict:
            entry['question'] = json_dict[question_id]['question']
            entry['answers'] = json_dict[question_id]['answers']
else:
    print('Error: pickle_data is not a dictionary of dictionaries.')

# Step 4: Post-process the 'triples' field
pattern = re.compile(r'^\d+\.\d+ #### ') 
# Pattern to match the number, space, and 4 hashes
for entry in pickle_data.values():
    if 'triples' in entry:
        # Remove the floating point number and hashes
        cleaned_triples = \
            [pattern.sub('', triple) for triple in entry['triples']]
        # Merge the list into a single string and remove instances of '###'
        merged_triples = '. '.join(cleaned_triples).replace('###', '')
        entry['triples'] = merged_triples

# Step 5: Save the merged data to a JSON file
with open(merged_file_path, 'w', encoding='utf-8') as merged_file:
    json.dump(pickle_data, merged_file, ensure_ascii=False, indent=4)

# Step 6: Print a sample of the merged data to verify
for i, (key, item) in enumerate(pickle_data.items()):
    # Print the first 3 items as a sample
    if i >= 10:
        break
    print(f'Item {i}:')
    print(f'  ID: {key}')
    for sub_key, value in item.items():
        print(f'    {sub_key}: {value}')
    print('\n')

print(f'The merged data has been saved to {merged_file_path}')
