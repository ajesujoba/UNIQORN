# code to run gpt-4o in rag setting for kg

import json
from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI(
    api_key='your-openai-api-key')

# Path to the merged JSON file
merged_file_path = 'merged-file-kg.json'
output_file_path = 'answers-gpt4o-rag-kg.json'

# merged_file_path = 'matches-gpt4o-rag-kg-perturbed.json'
# output_file_path = 'answers-gpt4o-rag-kg-perturbed.json'

# Load the merged JSON data
with open(merged_file_path, 'r', encoding='utf-8') as merged_file:
    data = json.load(merged_file)


# Function to generate answers using GPT-4
def generate_answer(question, context):
    prompt = (
        "Please provide a crisp answer to the following question. "
        "Your response should ideally be short strings (or lists of short "
        "strings). These strings could be entity labels of names and places, "
        f"numbers, dates or other strings like quotations, etc.\n\n{question}."
        f"You can only use this specified context for answering: {context}."
        "If the provided context does not contain the answer, please output"
        "the following string: The given evidence does not contain the answer "
        "to the input question. Please do not generate answers from your "
        "parametric memory or world knowledge."
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
save_interval = 10  # Save to file after every 10 questions
for i, (key, entry) in enumerate(data.items(), 1):
    if 'question' in entry:
        question = entry['question']
        print(f"\nProcessing question {i}/{total_questions} ...")
        print(f"\nQuestion: {question}")
        context = entry['triples']  # for kg
        context_length = len(context)

        if context_length > 25000:
            print("\nContext length exceeded")
            answer = "Context length exceeded"
        else:
            print(f"\nContext: {context}")
            answer = generate_answer(question, context)
        
        print(f"\nGenerated answer (RAG): {answer}")
        entry['rag_answer'] = answer
        gold_answer = entry['answers']
        print(f"\nGold answers: {gold_answer}")

        # Save to file at regular intervals
        if i % save_interval == 0:
            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                json.dump(data, output_file, ensure_ascii=False, indent=4)
            print(f"\nProgress saved to {output_file_path} at question {i}")

# Final save to ensure the last few entries are saved
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    json.dump(data, output_file, ensure_ascii=False, indent=4)

print(f"\nThe updated data has been saved to {output_file_path}")
