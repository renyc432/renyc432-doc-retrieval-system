import os
import json

from datetime import datetime


path = 'C:\\Users\\roy79\\Desktop\\Research\\question answering system\\code\\'
os.chdir(path)


# Create Index on SQuAD dataset

squad_train = json.loads(open('.//data//SQuAD_train-v2.0.json').read())
squad_train = squad_train['data']

# Transform the data into required format
documents = []
para_id = 0
for article_id in range(len(squad_train)):
    article = squad_train[article_id]
    
    paragraphs = article['paragraphs']
    for para in paragraphs:
        document = {'id':str(article_id)+'_'+str(para_id), 'contents':para['context'], 'article_id':article_id}
        documents.append(document)
        para_id += 1

with open('squad_train.json', 'w') as out:
    json.dump(documents, out)


# =============================================================================
# GENERATE ANSERINI INDEX
# python -m pyserini.index -collection JsonCollection -generator DefaultLuceneDocumentGenerator \
#  -threads 1 -input integrations/resources/squad_train \
#  -index indexes/squad_train -storePositions -storeDocvectors -storeRaw
# =============================================================================

from pyserini.search import SimpleSearcher
import info_retriever

searcher_squad = SimpleSearcher('indexes/squad_train')
searcher = info_retriever.Searcher(searcher_squad, is_debug=True)

question = 'After her second solo album, what other entertainment venture did Beyonce explore?'

hits = searcher.search(question)
results = searcher.get_results(hits)

start = datetime.now()
results = searcher.update_albert_rerank()
print('rerank took:', datetime.now()-start)


# notes
# single pass takes 2 seconds even at 100 requests
# on contrast, batch processing takes 7 seconds -> do example_to_features() manually could definitely speed it up
 


############################## Build Evaluation Set ############################

questions = []

para_id = 0
for article_id in range(len(squad_train)):
    article = squad_train[article_id]
    
    paragraphs = article['paragraphs']
    
    for para in paragraphs:
        q_id = 0
        for q in para['qas']:
            question = {'id':str(article_id)+'_'+str(para_id), 'q_id':q_id, 'question':q['question'], 'answers': q['answers'] }
            q_id += 1
            questions.append(question)
        para_id += 1


############################# Accuracy/Speed Test ##############################
import random
import numpy as np
import pandas as pd
questions_with_answer = [temp for temp in questions if temp['answers'] != []]

random.seed(87325)
test_questions = random.choices(questions_with_answer,k=5000)
# eliminate questions with no answers

searcher_squad = SimpleSearcher('indexes/squad_train')
searcher = info_retriever.Searcher(searcher_squad, is_debug=True)

is_benchmark = False

top1s = []
top5s = []
top10s = []
top20s = []
top50s = []

start = datetime.now()
for test in test_questions:
    q = test['question']
    para_id = test['id']

    hits = searcher.search(q)
    results = searcher.get_results(hits)
    if not is_benchmark:
        results = searcher.update_albert_rerank()
    
    top1 = results['Title'].head(1).eq(para_id).sum()
    top5 = results['Title'].head(5).eq(para_id).sum()
    top10 = results['Title'].head(10).eq(para_id).sum()
    top20 = results['Title'].head(20).eq(para_id).sum()
    top50 = results['Title'].head(50).eq(para_id).sum()
    
    top1s.append(top1)
    top5s.append(top5)
    top10s.append(top10)
    top20s.append(top20)
    top50s.append(top50)
print(datetime.now()-start)

top1_pt = np.mean(top1s)
top5_pt = np.mean(top5s)
top10_pt = np.mean(top10s)
top20_pt = np.mean(top20s)
top50_pt = np.mean(top50s)
print(top1_pt, top5_pt, top10_pt, top20_pt, top50_pt)









