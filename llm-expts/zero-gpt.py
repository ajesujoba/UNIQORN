# code to obtain zero-shot output of gpt4o on dev set questions

import json
from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI(
    api_key='your-openai-api-key-here')

# Path to the merged JSON file - doesn't matter if this is kg or text
# only the question is used
merged_file_path = 'merged-file-kg.json'
output_file_path = 'answers-gpt4o-zero.json' 

# Load the merged JSON data
with open(merged_file_path, 'r', encoding='utf-8') as merged_file:
    data = json.load(merged_file)


# Function to generate answers using GPT-4o
def generate_answer(question):
    prompt = (
        "Please provide a crisp answer to the following question. "
        "Your response should ideally be short strings (or lists of short "
        "strings). These strings could be entity labels of names and places, "
        f"numbers, dates or other strings like quotations, etc.\n\n{question}."
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=50,  
        temperature=0.0  
    )
    return response.choices[0].message.content.strip()


# Extract each question, generate an answer, and update the JSON data
total_questions = len(data)
for i, entry in enumerate(data.values(), 1):
    if 'question' in entry:
        question = entry['question']
        print(f"Processing question {i}/{total_questions} ...")
        print(f"Question: {question}")
        answer = generate_answer(question)
        print(f"Generated answer: {answer}")
        entry['generated_answer'] = answer

# Save the updated JSON data to a new file
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    json.dump(data, output_file, ensure_ascii=False, indent=4)

print(f"The updated data has been saved to {output_file_path}")
