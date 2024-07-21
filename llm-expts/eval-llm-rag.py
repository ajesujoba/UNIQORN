# this code finds cases where an llm (gpt-4o/phi-3) correctly answered
# a question
# writes these matched cases to a file
# addl: these matched cases can then be perturbed for stress-testing

import json
from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI(
    api_key='your-openai-api-key')

# output_file_path = 'answers-gpt4o-rag-kg-text.json'
# eval_output_file_path = 'eval-gpt4o-rag-kg-text.txt'
# matches_output_file_path = 'matches-gpt4o-rag-kg-text.json'

# output_file_path = 'answers-gpt4o-rag-kg.json'
# eval_output_file_path = 'eval-gpt4o-rag-kg.txt'
# matches_output_file_path = 'matches-gpt4o-rag-kg.json'

# output_file_path = 'answers-gpt4o-rag-text.json'
# eval_output_file_path = 'eval-gpt4o-rag-text.txt'
# matches_output_file_path = 'matches-gpt4o-rag-text.json'

# output_file_path = 'answers-gpt4o-rag-kg-text-perturbed.json'
# eval_output_file_path = 'eval-gpt4o-rag-kg-text-perturbed.txt'
# matches_output_file_path = 'matches-gpt4o-rag-kg-text-perturbed.json'

# output_file_path = 'answers-gpt4o-rag-kg-perturbed.json'
# eval_output_file_path = 'eval-gpt4o-rag-kg-perturbed.txt'
# matches_output_file_path = 'matches-gpt4o-rag-kg-perturbed.json'

output_file_path = 'answers-gpt4o-rag-text-perturbed.json'
eval_output_file_path = 'eval-gpt4o-rag-text-perturbed.txt'
matches_output_file_path = 'matches-gpt4o-rag-text-perturbed.json'

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

matches = {}

# Open the evaluation output file
with open(eval_output_file_path, 'w', encoding='utf-8') as eval_file:
    for i, (key, entry) in enumerate(data.items()):
        if 'answers' in entry and 'rag_answer' in entry:
            gold_answers = entry['answers']
            generated_answer = entry['rag_answer']
            result = compare_answers(gold_answers, generated_answer)

            if result == '1':
                eval_file.write("\nMatch found:\n")
                eval_file.write(f"Gold answers: {gold_answers}\n")
                eval_file.write(f"Generated answer: {generated_answer}\n")
                match_count += 1
                matches[key] = entry  # Store the key and entry in matches dict
            else:
                mismatch_count += 1
                eval_file.write("\nMismatch found:\n")
                eval_file.write(f"Gold answers: {gold_answers}\n")
                eval_file.write(f"Generated answer: {generated_answer}\n")

        # Save progress every 10 entries
        if (i + 1) % 10 == 0:
            eval_file.write(f"\nProcessed {i + 1} entries...\n")
            with open(matches_output_file_path,
                      'w',
                      encoding='utf-8') as matches_file:
                json.dump(matches, matches_file, indent=4, ensure_ascii=False)
            print(f"Processed {i + 1} entries and saved intermediate results.")

    # Final output of the results
    eval_file.write(f"\nTotal matches (1s): {match_count}\n")
    eval_file.write(f"Total mismatches (0s): {mismatch_count}\n")

# Save the final matched entries to a separate JSON file
with open(matches_output_file_path, 'w', encoding='utf-8') as matches_file:
    json.dump(matches, matches_file, indent=4, ensure_ascii=False)

print(f"Evaluation completed and results saved to {eval_output_file_path}")
print(f"Matched entries saved to {matches_output_file_path}")
