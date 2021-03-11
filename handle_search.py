import os
import json

import pandas as pd

import info_retriever

from plot_table import plot_table_to_html

from datetime import datetime

path = 'C:\\Users\\roy79\\Desktop\\Research\\question answering system\\code\\'
os.chdir(path)


# load the searcher before running the server
os.environ['ANSERINI_CLASSPATH'] = "C:/Users/roy79/anaconda3/Lib/site-packages/pyserini/resources/jars/"
doc_index = 'trec-covid-r5-full-text'
search_method = 'simple'
searcher = info_retriever.Searcher(doc_index,search_method)



def handle_search(question):
    
    start = datetime.now()
#    question = "Is covid transmitted by aerisol, droplets, food, close contact, fecal matter, or water"
    hits = searcher.search(question)
    len(hits)    
    results = searcher.get_results(hits)
    results = searcher.update_albert_rerank()
    
    plot_table_to_html(results)
    
    print(datetime.now()-start)
    return











