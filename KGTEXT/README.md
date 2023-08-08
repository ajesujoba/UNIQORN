## UNIQORN HETEROGENEOUS (KG+TEXT) Setup

To run UNIQORN in the HETEROGENEOUS setup requires following major steps.

### 1. INPUT file format
Format the questions in a json file (questions.json) with or without `answers` in the following way:
```
{"id": "train_7308", "question": "Who is the child of Walter Raleigh?", "answers": ["Carew Raleigh", "Walter Ralegh"]}
```
and place it in the 'Files' folder. A sample file is kept there as reference. 

### 2. Generate input files from the KG Side.
The KG+TEXT setup utilises several files generated for the KG and TEXT setups. From the KG side, we require to do the following steps with minor modifications as mentioned below,

a) ```NERD over the question``` - No modification in this step, run following the instructions as mentioned in the ReadMe for KG setup. However, it need not be re-run in case you have already run for the KG setup.

b) ```Extract Triples for the seed entities from WIKIDATA``` - No modification in this step also, run following the instructions as mentioned in the ReadMe for KG setup. However, it need not be re-run in case you have already run for the KG setup.

c) ```Facts/Triples scoring with BERT``` - This step requires two modifications. First, the BERT model to be used in this step, needs to be updated. You need to download the BERT model `model3.bin` from https://drive.google.com/file/d/1i901TbjejHw0mArcSGNG2DGpAzQC4nYW/view?usp=sharing and place (replace the `model3.bin` of KG setup if already present) it in the KG/TRIPLE_SCORER/BERT folder. Second, run the `scoreTriple_HYB.py` (also stored in KG/TRIPLE_SCORER folder) instead of scoreTriple.py. Re-run all the codes under this step following the ReadMe for KG setup.

d) ```Answer Questions``` - No need to run this step.

### 3. Generate input files from the TEXT Side.
Similar to KG side, from the TEXT side, we require to do the following steps with minor modifications as mentioned below,

a) ```GET DOCUMENTS from GOOGLE for each question``` - No modification in this step, run following the instructions as mentioned in the ReadMe for TEXT setup. However, it need not be re-run in case you have already run for the TEXT setup.

b) ```Score Triples``` - This step requires only one modification. The BERT model to be used in this step, needs to be updated. You need to download the BERT model `model3.bin` from <google drive link> and place (replace the `model3.bin` of TEXT setup if already present) it in the TEXT/SCORER/BERT folder. Re-run all the codes under this step following the ReadMe for TEXT setup.

c) ```Create alignment links``` - No need to run this step.

d) ```Answer Questions``` - No need to run this step.


### 4. Create alignment links
This code helps to create alignment links between the entity, predicate and type nodes in the graph across KG-TEXT nodes pairs as well as TEXT-TEXT node pairs. For these you need to use a conda environment to be created from `BERT_ENV.yml` provided inside the `CREATE_ALIGN` folder.
```
cd CREATE_ALIGN
python createTable_HETERSd.py questions.json alignlog 0 6
```
#0 and 6 are the start and end indices as there are six questions in questions.json.

#Before executing the code, you need to download the embedding files for `Word2Vec (GoogleNews-vectors-negative300.bin.gz)` or `Wikipedia2Vec (enwiki_20180420_100d.txt)` from https://drive.google.com/drive/folders/1b0OPw1u2AIFeUQby24GZgTwDrDyi7ciQ?usp=sharing and place them in the CREATE_ALIGN/EMBED folder. You need to set the `embedding` variable in the config file according to your choice of embedding.

### 5. Answer Questions
Answer questions over the scored Triples and Meta Data. Please continue using the same environment for executing these codes. The results would be stored in `Files/Results/0.8_0.7_0.7/_ANSWER_0.8_0.7_0.7/LcQUAD_Answer_list_<question_id>_<ranking_scheme>` pickle files. 
```
cd GET_ANSWER
python test_answer.py questions.json hetlog 0 6
```
#0 and 6 are the start and end indices as there are six questions in questions.json


