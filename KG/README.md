## UNIQORN KG Setup

To run UNIQORN in the KG setup requires following three major steps.

### 1. NERD over the question. 
Perform NERD over the questions to obtain seed entities. In the UNIQORN setup, we used the combination of both TAGME and ELQ. We have provided the code to extract the seed entities using both TAGME and ELQ in the `NERD` folder. ...
```
bash get_Elq_Entities.sh #get the seed entities from Questions using ELQ
bash get_Tagme_Entities.sh #get the seed entities from Questions using TAGME
bash get_Merge_ELQTAGME.sh #Combine the seed entities from both ELQ and TAGME
```
### 2. Extract Triples for the seed entities from WIKIDATA 
Having obtained the seed entities from `1.`, obtain the triples(facts) for those seed entities from WIKIDATA. For UNIQORN we used the CLOCQ API to extract the triples. Furthermore, UNIQORN requires some meta data in answering question, they include, seed entity names (NERD entity names), Entity Types (occupation and instance), and Predicate Aliases. All this can be carried out using CLOCQ. 
```
# To extract the facts from wikidata for the seed entity
bash ExtractCLOCQSPO.sh
# To get all entity Types ()
getCLOCQTypes.sh
# To obtain the names of the Seed entities
bash getCLOCQSeeds.sh
# To obtain the Predicate aliases
bash getCLOCQAliases.sh

```
3. 
4. Score the SPO Triples and Types from CLOCQ using the pretrained BERT model. 
5. Answer questions over the scored Triples and Meta Data
