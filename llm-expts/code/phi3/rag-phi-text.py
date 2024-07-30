# code to get phi3 outputs over text snippets (rag)

import json
from transformers import AutoModelForCausalLM, AutoTokenizer


# Initialize the model and tokenizer
model_name = "microsoft/Phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Path to the merged JSON file
merged_file_path = 'merged-file-text.json'
output_file_path = 'answers-phi3-rag-text.json'

# Load the merged JSON data
with open(merged_file_path, 'r', encoding='utf-8') as merged_file:
    data = json.load(merged_file)


# Function to generate answers using the Phi-3-Mini-4K-Instruct model
def generate_answer(question, context):
    prompt = (
        "Please provide a crisp answer to the following question. "
        "Your response should ideally be short strings (or lists of short "
        "strings). These strings could be entity labels of names and places, "
        f"numbers, dates or other strings like quotations, etc.\n\n{question}."
        f"You can only use this specified context for answering: {context}."
        "Please do not use any of your existing world knowledge."
    )

    inputs = tokenizer(prompt,
                       return_tensors="pt",
                       max_length=512,
                       truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=50)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return answer


# Extract each question, generate an answer, and update the JSON data
total_questions = len(data)
save_interval = 10  # Save to file after every 10 questions
for i, (key, entry) in enumerate(data.items(), 1):
    if 'question' in entry:
        question = entry['question']
        print(f"\nProcessing question {i}/{total_questions} ...")
        print(f"\nQuestion: {question}")
        context = entry['snippets']  # for kg
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
