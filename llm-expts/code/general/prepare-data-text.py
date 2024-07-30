# code to create merged QA data file for text with questions, gold answers,
# rag contexts (snippets) with bert from individual pickle and json files

import pickle
import json


# Paths to the files
pickle_file_path = 'top5-snippets-text.pkl' 
json_file_path = 'dev-qa-pairs.json'
merged_file_path = 'merged-file-text.json' 

# Step 1: Load the pickle file
with open(pickle_file_path, 'rb') as pickle_file:
    pickle_data = pickle.load(pickle_file)

# Check the type of loaded pickle data
# print(f'Type of pickle_data: {type(pickle_data)}')
# if isinstance(pickle_data, dict):
#     print(f'Sample of pickle_data: {list(pickle_data.items())[:3]}')
#     # Print first 3 items as a sample

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

ctr = 0
# Step 4: Join the entries 'snippets' in the snippets list
for entry in pickle_data.values():
    if 'snippets' in entry:
        # Merge the list into a single string 
        print(type(entry['snippets']))
        merged_snippets = '. '.join(entry['snippets'])
        entry['snippets'] = merged_snippets
        print(type(entry['snippets']))
        print(f"\n{entry['snippets']}\n")
        ctr += 1
print(ctr)

# Step 5: Save the merged data to a JSON file
with open(merged_file_path, 'w', encoding='utf-8') as merged_file:
    json.dump(pickle_data, merged_file, ensure_ascii=False, indent=4)

print(f'The merged data has been saved to {merged_file_path}')
