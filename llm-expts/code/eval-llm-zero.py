# code for evaluating zero-shot results of llms against ground-truth answers 
# using gpt4o

import json
from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI(
    api_key='your-openai-api-key-here')

# Path to the output JSON file
# output_file_path = 'answers-gpt4o-zero.json'
# eval_output_file_path = 'eval-gpt4o-zero.txt'

output_file_path = 'answers-phi-3-zero.json'
eval_output_file_path = 'eval-phi-3-zero.txt'

# Load the JSON data
with open(output_file_path, 'r', encoding='utf-8') as output_file:
    data = json.load(output_file)


# Function to compare answers using GPT-4o
def compare_answers(gold_answers, generated_answer):
    prompt = (
        "Please compare the following answers and determine if "
        "the generated answer matches any of the gold answers semantically. "
        "For semantic matches (i.e., for getting a 1, the answers "
        "may not match exactly, but the two answers should refer to the same "
        "entity, date, or other constant. "
        "Respond with '1' if they match and '0' if they do not.\n\n"
        f"Gold answers: {gold_answers}\n"
        f"Generated answer: {generated_answer}"
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


# Compare the answers and record the results
total_entries = len(data)
match_count = 0
mismatch_count = 0

# Open the evaluation output file
with open(eval_output_file_path, 'w', encoding='utf-8') as eval_file:
    for entry in data.values():
        if 'answers' in entry and 'generated_answer' in entry:
            gold_answers = entry['answers']
            generated_answer = entry['generated_answer']
            result = compare_answers(gold_answers, generated_answer)

            if result == '1':
                eval_file.write("\nMatch found:\n")
                eval_file.write(f"Gold answers: {gold_answers}\n")
                eval_file.write(f"Generated answer: {generated_answer}\n")
                match_count += 1
            else:
                mismatch_count += 1
                eval_file.write("\nMismatch found:\n")
                eval_file.write(f"Gold answers: {gold_answers}\n")
                eval_file.write(f"Generated answer: {generated_answer}\n")

    # Final output of the results
    eval_file.write(f"\nTotal matches (1s): {match_count}\n")
    eval_file.write(f"Total mismatches (0s): {mismatch_count}\n")

print(f"Evaluation completed and results saved to {eval_output_file_path}")
