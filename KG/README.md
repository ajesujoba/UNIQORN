## UNIQORN KG Setup

To run UNIQORN in the KG setup requires following major steps.

### 1. INPUT file format
Format the questions in a json file (questions.json) with or without `answers` in the following way:
```
{"id": "train_7308", "question": "Who is the child of Walter Raleigh?", "answers": ["Carew Raleigh", "Walter Ralegh"]}
```
and place it in the 'Files' folder. A sample file is kept there as reference.


### 2. NERD over the question. 
Perform NERD over the input questions present in questions.json to obtain seed entities. In the UNIQORN setup, we used the combination of two NEW tools - TAGME and ELQ. We have provided the code to extract the seed entities using both TAGME and ELQ in the `NED` folder. Use the `NED_ENV.yml` provided inside the `NED` folder to create a conda environment and execute the codes after activating that.
```
cd NED
```
#To get the seed entities from Questions present in questions.json using TAGME and store them in TagmeEntity.pkl
```
python get_seed_entities_TAGME.py --inputfile ./../Files/questions.json --outputfile ./../Files/TagmeEntity.pkl
```
#To get the seed entities from Questions present in questions.json using TAGME and store them in ElqEntity.pkl
```
python get_seed_ELQ.py --inputfile ./../Files/questions.json --outputfile ./../Files/ElqEntity.pkl
```
#Note: Before running the ELQ, you need to download the BLINK repository including models from the GITHUB (https://github.com/facebookresearch/BLINK/tree/main/) and set the $PYTHONPATH variable accordingly (i.e export PYTHONPATH=$PYTHONPATH:[Path to the BLINK-main folder]). 

#To combine the seed entities obtained from both ELQ and TAGME
```
python merge_WikiIDs_TAGME_ELQ.py --inputfile ./../Files/questions.json --tagmefile ./../Files/TagmeEntity.pkl --elqfile ./../Files/ElqEntity.pkl --outputfile ./../Files/ElqTagme.pkl
```

### 3. Extract Triples for the seed entities from WIKIDATA 
Having obtained the seed entities from `2.`, obtain the triples(facts) for those seed entities from WIKIDATA. For UNIQORN we used the CLOCQ API (https://github.com/PhilippChr/CLOCQ/tree/master/) to extract the triples. Furthermore, UNIQORN requires some meta data in answering question, they include, seed entity names (NERD entity names), Entity Types (occupation and instance), and Predicate Aliases. All this can be carried out using CLOCQ. You can continue using the same `NED_ENV.yml` python environment for executing these codes.
```
cd GET_TRIPLES
```
#To extract the facts from wikidata for the seed entity
```
python ExtractCLOCQSPO.py --inputfile ./../Files/questions.json --seedfile ./../Files/ElqTagme.pkl --outputfile ./../Files/WikiTriplesUniqorn/
```
#To get all entity Types
```
python getAllEntityType.py --inputfile ./../Files/questions.json --triplefile ./../Files/WikiTriplesUniqorn/ --outputfile ./../Files/WikiTriplesUniqorn/
```
#To obtain the names of the Seed entities
```
python getSeedTerms.py --inputfile ./../Files/questions.json --seedfile ./../Files/ElqTagme.pkl --outputfile ./../Files/ScoreTriples/
```
#To obtain the Predicate aliases
```
python getAllAliases.py --inputfile ./../Files/questions.json --triplefile ./../Files/WikiTriplesUniqorn/ --outputfile ./../Files/ScoreTriples/
```
We are also retrieveing 1/2-hop(s) paths betweeen seed entities, we score those paths and we pick the most relevant path to the question using cosine similarity between the question and thr question vector representation obtained from BERT

#To get the Paths (triples) between the seed entities using CLOCQ
```
python extractPath.py --inputfile ./../Files/questions.json --seedfile ./../Files/ElqTagme.pkl --outputfile ./../Files/WikiTriplesUniqorn/
```
#To extract the Label (Surface form of the paths)
```
python extractPathLabel.py --inputfile ./../Files/questions.json --pathfile ./../Files/WikiTriplesUniqorn/ --outputfile ./../Files/WikiTriplesUniqorn/
```
#To score the triples based on their relevance to the question 
```
python scorepath.py \
	--inputfile ./../Files/questions.json \
	--bertdir ./BERT/ \
	--pathfile ./../Files/WikiTriplesUniqorn/ \
	--outputfile ./../Files/WikiTriplesUniqorn/
```
#Note that before executing, you need to download the BERT model `pytorch_model.bin` from here - https://drive.google.com/file/d/1WTZVvHV5cZFdAa5vJIcrPa-QROstB861/view?usp=sharing and place it in the GET_TRIPLES/BERT folder.	

#Get the most relevant path (top scoring path)
```
python getTopPaths.py \
	--inputfile ./../Files/questions.json \
	--pathfile ./../Files/WikiTriplesUniqorn/ \
	--outputfile ./../Files/ScoreTriples/
```
#Get the predicate aliases for the seed connection paths and update the previous aliases.
```
python getAllAliases_e2e.py \
	--inputfile ./../Files/questions.json \
	--pathfile ./../Files/WikiTriplesUniqorn/ \
	--outputfile ./../Files/ScoreTriples/

```
### 3. Facts/Triples scoring with BERT
The next step is to score the SPO Triples and Types from CLOCQ using our pretrained BERT model. For these you need to use a different conda environment to be created from `BERT_ENV.yml` provided inside the `TRIPLE_SCORER` folder.
```
cd TRIPLE_SCORER
```
#To score the facts
```
python scoreTriple.py \
	--inputfile ./../Files/questions.json \
	--triplefile ./../Files/WikiTriplesUniqorn/ \
	--outputfile ./../Files/ScoreTriples/ \
	--modeldir ./BERT/model3.bin
```
#Note that before executing, you need to download the BERT model `model3.bin`from here -https://drive.google.com/file/d/1WANHWActydziT4o1IoHZqZiVI-n6phPS/view?usp=sharing and place it in the TRIPLE_SCORER/BERT folder.	

#To score the facts 
```
python scoreTypes.py \
	--inputfile ./../Files/questions.json \
	--triplefile ./../Files/WikiTriplesUniqorn/ \
	--outputfile ./../Files/ScoreTriples/ \
	--modeldir ./BERT/model3.bin
```

### 4. Answer Questions
Answer questions over the scored Triples and Meta Data. Please continue using the same `BERT_ENV` for executing these codes. The results would be stored in `Files/Results/5_10/_ANSWER_5_10/LcQUAD_Answer_list_<question_id>_<ranking_scheme>` pickle files. 
```
cd GET_ANSWER
```
#Answer questions
```
python answer.py \
	--write_dir  ./../Files/Results/ \
	--kgcorpus ./../Files/ScoreTriples/ \
	--inputfile ./../Files/questions.json \
	--topk 5 \
	--n_GST 10 \
	--addType 1 --connectseed 1 \
	--chain_join_flag 1 \
	--logfile log_answer
```

