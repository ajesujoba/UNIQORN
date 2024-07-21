# this code counts how often "Seqret Uniquorn" is part of 
# a generated answer in RAG (perturbation experiments)

def count_matches(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        
    match_count = 0
    
    for line in lines:
        if ("Seqret" in line or "Uniquorn" in line):
            # if ("Generated answer" in line and 
            #    ("Seqret" in line or "Uniquorn" in line)):
            match_count += 1
            print(line.strip())
    
    print(f"\nTotal matches: {match_count}")


filename = "eval-gpt4o-rag-kg-text-perturbed.txt"
# filename = "eval-gpt4o-rag-kg-perturbed.txt"
# filename = "eval-gpt4o-rag-text-perturbed.txt"

count_matches(filename)
