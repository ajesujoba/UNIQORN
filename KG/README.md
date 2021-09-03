## UNIQORN KG Setup

To run UNIQORN in the KG setup requires following three major steps.

### 1. NERD over the question. 
Perform NERD over the questions to obtain seed entities. In the UNIQORN setup, we used the combination of both TAGME and ELQ. We have provided the code to extract the seed entities using both TAGME and ELQ in the `NERD` folder. ...
```
bash get_Elq_Entities.sh #get the seed entities from Questions using ELQ
bash get_Tagme_Entities.sh #get the seed entities from Questions using TAGME
bash #Combine the seed entities from both ELQ and TAGME
```
2. Extract the Meta data need to answer the questions, this includes, the SPO, NED entity names, Types (occupation and instance), and Aliases. This can be carried out using CLOCQ. 
3. Score the SPO Triples and Types from CLOCQ using the pretrained BERT model. 
4. Answer questions over the scored Triples and Meta Data
