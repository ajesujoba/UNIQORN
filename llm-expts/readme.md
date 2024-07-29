# Guide to scripts, data, and results for the LLM experiments

Below we provide a short guide to the various scripts and data used in the LLM experiments (zero-shot and RAG) for UNIQORN, and the result files they generated. The files are organized into topical directories, but we provide the short descriptions of what each file does, as a flat list below for easy reference. This should be helpful for reproducibility of the desired experiments. Note that the backend LLMs are constantly updated (espcially GPT-4o accessed via the API), so exact repeatability of the reported values is not guaranteed. But we expect overall trends of the experiments to hold good for some time. The original experiments were conducted in July 2024.

## Scripts

* check-hallu-gpt4o.py: code checks for hallucinations in gpt-4o outputs

* count-seqret.py: code counts how often "Seqret Uniquorn" is part of a generated answer in RAG (perturbation experiments)

* eval-llm-rag-uniqorn.py: code compares "rag" outputs with uniqorn (default 
uniqorn configuration) with gold answers via gpt4o

* eval-llm-rag.py: code compares rag outputs with gpt4o/phi3 with gold answers via gpt4o

* eval-llm-zero.py: code compares zero-shot outputs with gpt4o/phi3 with gold answers via gpt4o

* perturb-gpt4o-matches.py: code to prepare synthetic data for stress-testing RAG, with passage perturbation

* prepare-data-kg-text.py: code to create merged data file for kg and text for dev QA pairs

* prepare-data-kg.py: code to create merged QA data file for kg with questions, gold answers, rag contexts (triples) with bert from individual pickle and json files

* prepare-data-text.py: code to create merged QA data file for text with questions, gold answers, rag contexts (snippets) with bert from individual pickle and json files

* prepare-uniqorn-rag-data.py: code to create merged data file with uniqorn answers for kg+text, kg, text

* rag-gpt-kg-text.py: code for generating answers in rag setup by gpt4o over kg and text

* rag-gpt-kg.py: code for generating answers in rag setup by gpt4o over kg

* rag-gpt-text.py: code for generating answers in rag setup by gpt4o over text

* rag-phi-kg-text.py: code for generating answers in rag setup by phi3 over kg and text

* rag-phi-kg.py: code for generating answers in rag setup by phi3 over kg

* rag-phi-text.py: code for generating answers in rag setup by phi3 over text

* zero-gpt.py: code for generating zero-shot results with got4o

* zero-phi.py: : code for generating zero-shot results with phi3

## Data

* dev-qa-pairs.json
* merged-file-kg-perturbed.json
* merged-file-kg-text-perturbed.json
* merged-file-kg-text.json
* merged-file-kg.json
* merged-file-text-perturbed.json
* merged-file-text.json
* top5-facts-kg.pkl
* top5-snippets-text.pkl

## Results

* answers-gpt4o-rag-kg-hallu.json
* answers-gpt4o-rag-kg-perturbed-full.json
* answers-gpt4o-rag-kg-perturbed.json
* answers-gpt4o-rag-kg-text-perturbed-full.json
* answers-gpt4o-rag-kg-text-perturbed.json
* answers-gpt4o-rag-kg-text.json
* answers-gpt4o-rag-kg.json
* answers-gpt4o-rag-text-hallu.json
* answers-gpt4o-rag-text-perturbed-full.json
* answers-gpt4o-rag-text-perturbed.json
* answers-gpt4o-rag-text.json
* answers-gpt4o-zero.json
* answers-phi3-rag-kg-hallu.json
* answers-phi3-rag-kg-text.json
* answers-phi3-rag-kg.json
* answers-phi3-rag-text-hallu.json
* answers-phi3-rag-text.json
* answers-phi3-zero.json
* answers-uniqorn-rag-kg-text.json
* answers-uniqorn-rag-kg.json
* answers-uniqorn-rag-text.json
* eval-gpt4o-rag-kg-perturbed-full.txt
* eval-gpt4o-rag-kg-perturbed.txt
* eval-gpt4o-rag-kg-text-perturbed-full.txt
* eval-gpt4o-rag-kg-text-perturbed.txt
* eval-gpt4o-rag-kg-text.txt
* eval-gpt4o-rag-kg.txt
* eval-gpt4o-rag-text-perturbed-full.txt
* eval-gpt4o-rag-text-perturbed.txt
* eval-gpt4o-rag-text.txt
* eval-gpt4o-zero.txt
* eval-phi3-rag-kg-text.txt
* eval-phi3-rag-kg.txt
* eval-phi3-rag-text.txt
* eval-phi3-zero.txt
* eval-uniqorn-rag-kg-text.txt
* eval-uniqorn-rag-kg.txt
* eval-uniqorn-rag-text.txt
* matches-gpt4o-rag-kg-perturbed-full.json
* matches-gpt4o-rag-kg-text-perturbed-full.json
* matches-gpt4o-rag-kg-text-perturbed.json
* matches-gpt4o-rag-kg-text.json
* matches-gpt4o-rag-kg.json
* matches-gpt4o-rag-text-perturbed-full.json
* matches-gpt4o-rag-text-perturbed.json
* matches-gpt4o-rag-text.json
