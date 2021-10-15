## UNIQORN KG Setup

To run UNIQORN in the KG setup requires following three major steps.

### 1. NERD over the question. 
Perform NERD over the questions to obtain seed entities. In the UNIQORN setup, we used the combination of both TAGME and ELQ. We have provided the code to extract the seed entities using both TAGME and ELQ in the `NERD` folder. ...
```
cd NED
bash get_Elq_Entities.sh #get the seed entities from Questions using ELQ
bash get_Tagme_Entities.sh #get the seed entities from Questions using TAGME
bash get_Merge_ELQTAGME.sh #Combine the seed entities from both ELQ and TAGME
```
### 2. Extract Triples for the seed entities from WIKIDATA 
Having obtained the seed entities from `1.`, obtain the triples(facts) for those seed entities from WIKIDATA. For UNIQORN we used the CLOCQ API to extract the triples. Furthermore, UNIQORN requires some meta data in answering question, they include, seed entity names (NERD entity names), Entity Types (occupation and instance), and Predicate Aliases. All this can be carried out using CLOCQ. 
```
cd Meta
# To extract the facts from wikidata for the seed entity
bash ExtractCLOCQSPO.sh
# To get all entity Types ()
getCLOCQTypes.sh
# To obtain the names of the Seed entities
bash getCLOCQSeeds.sh
# To obtain the Predicate aliases
bash getCLOCQAliases.sh

```
### 3. Facts/Triples scoring with BERT
The next step is to score the SPO Triples and Types from CLOCQ using our pretrained BERT model. 
```
cd TripleScorer
# To score the facts 
bash score_triple.sh
# To score the facts 
bash score_type.sh

```
### 4. Get the highscoring path between Seed entities
The next step is to get the paths between the seed en
```
cd TripleScorer
# To score the facts
bash score_triple.sh
# To score the facts
bash score_type.sh

```
### 5. Answer Questions
Answer questions over the scored Triples and Meta Data
```
cd uniqorn
#Answer questions
bash runanswer.sh 
```
### Notes
1. Download the finetuned BERT model from <a href = "qa.mpi-inf.mpg.de/uniqorn/models/uniqorn.zip"> here </a>
2. The CLOCQ code for the extraction of triples is not yet online, this should be available in the coming week. 

