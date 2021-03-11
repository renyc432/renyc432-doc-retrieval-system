# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 20:32:49 2021

@author: roy79
"""

# plot table

import pandas as pd

import webbrowser


# =============================================================================
# html_string = '''
# <html>
#   <head><title>HTML Pandas Dataframe with CSS</title></head>
#   <link rel="stylesheet" href="{{{{ url_for('static',     filename='css/df_style.css')}}}}">
#   <body>
#     {table}
#   </body>
# </html>.
# '''
# =============================================================================


html_string = '''
<!DOCTYPE html>
<html lang="en" dir="ltr">
    <head>
        <meta charset="utf-8">
        <title>Search</title>
        <link rel="stylesheet" href="/static/css/search_result_style.css" type="text/css">
        <link rel="stylesheet" href="{{{{ url_for('static',     filename='css/df_style.css')}}}}">
    </head>
    <body>
        {{% extends "template.html" %}}
        {{% block content %}}

        <div class='search-box'>
		    <form action="{{{{ url_for("search")}}}}" method="post"> 
		      	<input type="text" id="question" name="question" placeholder="Please enter your question">
			    <button type="submit">Search</button> 
    	    </form>
        </div>
        <div class='search-result'>
            {table}
        </div>
    {{% endblock %}}
  </body>
</html>

'''


def plot_table_to_html(df):

    pd.set_option('colheader_justify', 'center')
    pd.set_option('display.max_colwidth', 20)
    
    with open("./templates/output.html", "w", encoding="utf-8") as out:
        out.write(html_string.format(table=df.to_html(classes='mystyle', render_links=True, escape=False, justify='left')))


def open_in_browser(file):
    new = 2
    webbrowser.open(file, new=new)