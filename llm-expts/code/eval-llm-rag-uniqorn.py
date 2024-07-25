import json
from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI(
    api_key='your-openai-api-key')

# Path to the output JSON file
output_file_path = 'answers-uniqorn-rag-kg-text.json'
eval_output_file_path = 'eval-uniqorn-rag-kg-text.txt'

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
        if 'answers' in entry and 'rag_answer' in entry:
            gold_answers = entry['answers']
            generated_answers = entry['rag_answer']
            match_found = False

            # Iterate over the list of generated answers
            for generated_answer in generated_answers:
                result = compare_answers(gold_answers, generated_answer)

                if result == '1':
                    match_found = True
                    eval_file.write("\nMatch found:\n")
                    eval_file.write(f"Gold answers: {gold_answers}\n")
                    eval_file.write(f"Generated answer: {generated_answer}\n")
                    match_count += 1
                    break

            if not match_found:
                mismatch_count += 1
                eval_file.write("\nMismatch found:\n")
                eval_file.write(f"Gold answers: {gold_answers}\n")
                eval_file.write(f"Generated answers: {generated_answers}\n")

    # Final output of the results
    eval_file.write(f"\nTotal matches (1s): {match_count}\n")
    eval_file.write(f"Total mismatches (0s): {mismatch_count}\n")

print(f"Evaluation completed and results saved to {eval_output_file_path}")
