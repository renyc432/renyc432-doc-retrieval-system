import os

from plot_table import plot_table_to_html

from datetime import datetime

path = 'C:\\Users\\roy79\\Desktop\\Research\\question answering system\\code\\'
os.chdir(path)



#question = "Is covid transmitted by aerisol, droplets, food, close contact, fecal matter, or water"

def handle_search(searcher, question, db):
    
    hits = searcher.search(question)
    results = searcher.get_results(hits)

    start = datetime.now()
    results = searcher.update_albert_rerank()
    print('rerank took:', datetime.now()-start)

    plot_table_to_html(results, db)
    
    return


# =============================================================================
# for DEBUG
# if __name__=="__main__":
#     
#     searcher_covid_simple = SimpleSearcher.from_prebuilt_index('trec-covid-r5-paragraph')
#     searcher = info_retriever.Searcher(searcher_covid_simple)
#     
#     question = 'Is covid transmitted by aerisol, droplets, food, close contact, fecal matter, or water'
#     handle_search(searcher, question, 'covid')
# 
# =============================================================================





