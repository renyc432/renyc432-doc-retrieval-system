
from flask import Flask, render_template
from flask import request

from handle_search import handle_search
from backend_test import abc

app = Flask(__name__)

# this says when the user goes to the website, they go to the default page
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/search", methods=["GET", "POST"])
def search():
    if request.method == 'POST':
        question = request.form.get('question')
        #return 'Testing'
        handle_search(question)
        return render_template("output.html")
    return render_template("search.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/test-css")
def test_css():
    return render_template("test-css.html")


@app.route("/search-result")
def search_result():
    return render_template("search-result.html")


if __name__=="__main__":
    app.run(debug=True)

# =============================================================================
# from flask import url_for
# with app.test_request_context():
#     print(f"{ url_for('static',     filename='css/template.css') }")
# =============================================================================