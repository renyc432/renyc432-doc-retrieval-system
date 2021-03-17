import os
import info_retriever

from flask import Flask, render_template
from flask import request

from handle_search import handle_search
#from backend_test import abc

from pyserini.search import SimpleSearcher

app = Flask(__name__)

searcher_covid_simple = SimpleSearcher.from_prebuilt_index('trec-covid-r5-paragraph')
#searcher_wiki_simple = SimpleSearcher.from_prebuilt_index('trec-covid-r5-paragraph')

def load_searcher(db):
    # load the searcher before running the server
    os.environ['ANSERINI_CLASSPATH'] = "C:/Users/roy79/anaconda3/Lib/site-packages/pyserini/resources/jars/"
    if db=='covid':
        searcher = info_retriever.Searcher(searcher_covid_simple)
# =============================================================================
#     elif db=='wiki':
#         searcher = info_retriever.Searcher(searcher_wiki_simple)
# =============================================================================
    return searcher

searcher_covid = load_searcher('covid')
#searcher_wiki = load_searcher('wiki')



@app.route("/")
def home():
    return render_template("home.html")

@app.route("/search", methods=["GET", "POST"])
def search():
    searcher = searcher_covid
    if request.method == 'POST':
        question = request.form.get('question')
        #return 'Testing'
        handle_search(searcher, question, 'covid')
        return render_template("output.html")
    return render_template("search.html")

@app.route("/search_wiki", methods=["GET","POST"])
def search_wiki():
    searcher = searcher_covid
    if request.method == 'POST':
        question = request.form.get('question')
        #return 'Testing'
        handle_search(searcher, question, 'covid')
        return render_template("output.html")
    return render_template("search_wiki.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/test-css")
def test_css():
    return render_template("test-css.html")


if __name__=="__main__":
    app.run(debug=True)

# =============================================================================
# from flask import url_for
# with app.test_request_context():
#     print(f"{ url_for('search') }")
# =============================================================================
    
    
    
    
    

