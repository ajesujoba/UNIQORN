# code to create merged data file with uniqorn answers for kg+text, kg, text

import json

# KG+Text data
# Load the JSON files
with open('answers-gpt4o-rag-kg-text.json', 'r') as gpt4o_file:
    gpt4o_data = json.load(gpt4o_file)

with open('answers-uniqorn-kg-text-dev.json', 'r') as uniqorn_file:
    uniqorn_data = [json.loads(line) for line in uniqorn_file]

# Create a dictionary for fast lookup of uniqorn data by id
uniqorn_dict = {item["id"]: item["Predictions"] for item in uniqorn_data}

# Merge the data
for id, gpt4o_entry in gpt4o_data.items():
    if id in uniqorn_dict:
        gpt4o_entry["rag_answer"] = uniqorn_dict[id]

# Write the merged data to a new file
with open('answers-uniqorn-rag-kg-text.json', 'w') as output_file:
    json.dump(gpt4o_data, output_file, indent=4)

print("Merged file created successfully.")

# KG data
# # Load the JSON files
# with open('answers-gpt4o-rag-kg.json', 'r') as gpt4o_file:
#     gpt4o_data = json.load(gpt4o_file)

# with open('answers-uniqorn-kg-dev.json', 'r') as uniqorn_file:
#     uniqorn_data = [json.loads(line) for line in uniqorn_file]

# # Create a dictionary for fast lookup of uniqorn data by id
# uniqorn_dict = {item["id"]: item["Predictions"] for item in uniqorn_data}

# # Merge the data
# for id, gpt4o_entry in gpt4o_data.items():
#     if id in uniqorn_dict:
#         gpt4o_entry["rag_answer"] = uniqorn_dict[id]

# # Write the merged data to a new file
# with open('answers-uniqorn-rag-kg.json', 'w') as output_file:
#     json.dump(gpt4o_data, output_file, indent=4)

# print("Merged file created successfully.")

# # Text data
# # Load the JSON files
# with open('answers-gpt4o-rag-text.json', 'r') as gpt4o_file:
#     gpt4o_data = json.load(gpt4o_file)

# with open('answers-uniqorn-text-dev.json', 'r') as uniqorn_file:
#     uniqorn_data = [json.loads(line) for line in uniqorn_file]

# # Create a dictionary for fast lookup of uniqorn data by id
# uniqorn_dict = {item["id"]: item["Predictions"] for item in uniqorn_data}

# # Merge the data
# for id, gpt4o_entry in gpt4o_data.items():
#     if id in uniqorn_dict:
#         gpt4o_entry["rag_answer"] = uniqorn_dict[id]

# # Write the merged data to a new file
# with open('answers-uniqorn-rag-text.json', 'w') as output_file:
#     json.dump(gpt4o_data, output_file, indent=4)

# print("Merged file created successfully.")
