## UNIQORN TEXT Setup

To run UNIQORN in the TEXT setup requires following major steps.

### 1. INPUT file format
Format the questions in a json file (questions.json) with or without `answers` in the following way:
```
{"id": "train_7308", "question": "Who is the child of Walter Raleigh?", "answers": ["Carew Raleigh", "Walter Ralegh"]}
```
and place it in the 'Files' folder. A sample file is kept there as reference. 

### 2. GET DOCUMENTS from GOOGLE for each question. 
Run the following code to crawl the top 10 most relevant documents for each question in questions.json using Google search API. You need to create a conda environment from the `GET_DOCS/UNI_TEXT.yml` file and activate. 
```
cd GET_DOCS
python get_docs.py ./../Files/questions.json ./../Files/
```

### 3. Score Triples. 
Next, we create SPO triples from the crawled documents and score these triples using BERT. The code also add TYPE nodes using Hearst Patterns. You can continue using the same python environment for running these command.
```
cd SCORER
python extractFinal.py questions.json scoreFinal.txt

# Note that before executing, you need to download the BERT model `model3.bin` and `pytorch_model.bin` from https://drive.google.com/file/d/1f4iR51N9IEs_TFZuQShSR-oJ8DkkCbnS/view?usp=sharing and https://drive.google.com/file/d/1WTZVvHV5cZFdAa5vJIcrPa-QROstB861/view?usp=sharing respectively and place them in the SCORER/BERT folder.
```

### 4. Create alignment links
This code helps to create alignment links between the entity. predicate and type nodes in the graph. You can continue using the same python environment for running these command.
```
cd CREATE_ALIGN
python create_align.py questions.json align_log 0 6
```
#0 and 6 are the start and end indices as there are six questions in questions.json
#Before executing the code, you need to download the embedding files for `Word2Vec (GoogleNews-vectors-negative300.bin.gz)` or `Wikipedia2Vec (enwiki_20180420_100d.txt)` from [<google drive link>](https://drive.google.com/drive/folders/1b0OPw1u2AIFeUQby24GZgTwDrDyi7ciQ?usp=sharing) and place them in the CREATE_ALIGN/EMBED folder. You need to set the `embedding` variable in the config file according to your choice of embedding.

### 5. Answer Questions
Answer questions over the scored Triples and Meta Data. Please continue using the same environment for executing these codes. The results would be stored in `Files/Results/0.5_0.9_0.9/_ANSWER_0.5_0.9_0.9/LcQUAD_Answer_list_<question_id>_<ranking_scheme>` pickle files. 
```
cd GET_ANSWER
python answer_test.py questions.json textlog 0 6
#0 and 6 are the start and end indices as there are six questions in questions.json
```

