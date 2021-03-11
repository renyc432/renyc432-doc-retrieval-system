
import json
import pandas as pd
from pyserini.search import SimpleSearcher

from albert_predict import albert_predict

from concurrent.futures import ProcessPoolExecutor

# collection of searchers

# This is a sparse search using BM25 ranking with bag-of-words
#searcher_covid_simple = SimpleSearcher.from_prebuilt_index('trec-covid-r5-full-text')
searcher_covid_simple = SimpleSearcher.from_prebuilt_index('trec-covid-r5-paragraph')

# =============================================================================
# #hybrid search is more effective than dense than sparse
# Covid prebuilt: Not implemented by the pyserini team yet
# from pyserini.search import SimpleSearcher
# from pyserini.dsearch import SimpleDenseSearcher, TCTColBERTQueryEncoder
# from pyserini.hsearch import HybridSearcher
# 
# ssearcher = SimpleSearcher.from_prebuilt_index('trec-covid-r5-full-text')
# encoder = TCTColBERTQueryEncoder('castorini/tct_colbert-msmarco')
# dsearcher = SimpleDenseSearcher.from_prebuilt_index(
#     'msmarco-passage-tct_colbert-hnsw',
#     encoder
# )
# search_covid_hybrid = HybridSearcher(dsearcher, ssearcher)
# =============================================================================

class Searcher:
    
    def __init__(self, index_name, method):
        if (index_name == 'trec-covid-r5-full-text') & (method == 'simple'):
            self.searcher = searcher_covid_simple
#        elif (index_name == 'trec-covid-r5-full-text') & (method == 'hybrid'):
#            self.searcher = searcher_covid_hybrid
        self.max_num_results = 20
        self.score_weight = 0.2
        
        self.albert = albert_predict()
        self.query = ''
        self.rank = []
        self.title = []
        self.author=[]
        self.url = []
        self.abstract = []
        self.date_of_pub = []
        self.arvix_id = []
        self.score = []
        self.answer = []
        self.paragraph = []


    def clear_results(self):
        self.rank.clear()
        self.title.clear()
        self.author.clear()
        self.url.clear()
        self.abstract.clear()
        self.date_of_pub.clear()
        self.arvix_id.clear()
        self.score.clear()
        self.answer.clear()
        self.paragraph.clear()
    
    def search(self, query, k = 100):
        self.query = query
        hits = self.searcher.search(query,k)
        return hits
        
    
    def get_article(self, hit, query=False):
         doc = self.searcher.doc(hit.docid)
         self.title.append(doc.get('title'))
         self.author.append(doc.get('authors'))
         self.url.append(doc.get('url'))
         self.abstract.append(doc.get('abstract'))
         self.date_of_pub.append(doc.get('publish_time'))
         self.arvix_id.append(doc.get('arvix_id'))
         
         #answer = self.extract_answer_albert(doc,self.query)
         #self.answer.append(answer)
         
         #self.rank.append(rank)
         #self.score.append(hit.score)
         return
        
        
    def get_results(self, hits):
        self.clear_results()
        
        i = 0
        for hit in hits:
            i += 1
            self.score.append(hit.score)
            self.get_article(hit)
            self.rank.append(i)
            self.paragraph.append(hit.contents)
            if (i >= self.max_num_results):
                break

        self.result = pd.DataFrame({'Title':self.title, 
                               'Author':self.author,
                               'Link':self.url, 
                               'Hit':self.paragraph,
                               'Abstract':self.abstract, 
                               'Publication Date':self.date_of_pub, 
#                               'arvix_id':self.arvix_id,
                               'BM25 Score':self.score},
                              index=self.rank)
        self.result.index.name = 'Rank'
        return self.result
    
    def extract_answer_albert(self, doc, query):
        context = ''
        doc_json = json.loads(doc.raw())
        for paragraph in doc_json['body_text']:
            context += paragraph['text']
        answer = self.albert.predict(query, context)
        return answer

    def update_albert_rerank(self):
        reranked = self.albert.rerank(self.query, self.result)
        print(len(self.result))
        print(reranked)
        self.result['Albert Answer'] = reranked['answer'].tolist()
        self.result['Albert Score'] = reranked['albert score'].tolist()
        self.result['Score'] = self.result['BM25 Score']*self.score_weight + self.result['Albert Score']*(1-self.score_weight)
                
        self.result.sort_values('Score', ascending=False, inplace=True)
        self.result.index = range(1,len(self.result)+1)
        self.result.index.name = 'Rank'
        
        columns_ordered = ['Title','Author','Albert Answer','Abstract','Publication Date','Score', 'Link']
        
        return self.result[columns_ordered]
        









