# code to create merged data file for kg and text for dev QA pairs

import json

# Load JSON files
with open('merged-file-kg.json', 'r') as file:
    merged_file_kg = json.load(file)

with open('merged-file-text.json', 'r') as file:
    merged_file_text = json.load(file)

# Create new merged data
merged_data = {}

for key in merged_file_text:
    if key in merged_file_kg:
        merged_data[key] = {
            "answers": merged_file_text[key]["answers"],
            "evidence": (merged_file_kg[key]["triples"] + '. ' +
                         merged_file_text[key]["snippets"]),
            "question": merged_file_text[key]["question"]
        }


# Save to new JSON file
with open('merged-file-kg-text.json', 'w') as file:
    json.dump(merged_data, file, indent=4)

print("Merged file 'merged-file-kg-text.json' created successfully.")
