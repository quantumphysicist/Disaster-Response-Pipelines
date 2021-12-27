"""
Creates a web app that can be accessed at http://localhost:3001/. 
The web app visualises the dataset of categorized messages on the homepage. 
When a message is inputted into the app, the message is classified into one or more categories.
"""
import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
#from sklearn.externals import joblib (Note that this is deprecated.)
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    """
    Tokenizes, lemmatizes and case normalizes each word in a piece of text.
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # Data needed for visuals
    
    # Data for graph 1
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index.str.replace("_", " ").str.title())
    print(genre_counts)
    print(genre_names)
    
    # Data for graph 2
    categories = [column for column in df.columns if column not in ['index', 'id', 'message', 'original', 'genre']]
    category_df = df[categories].sum() 
    category_counts = list(category_df.values)
    category_names = list(category_df.index.str.replace("_", " ").str.title())
    print(category_counts)
    print(category_names)
    
    # Data for graph 3
    category_df = pd.DataFrame(category_df).rename(columns = {0: "counts"})
    top5 = category_df.sort_values(by = 'counts', ascending = False).head(6)[1:6]
    top5_names = list(top5.index.str.replace("_", " ").str.title())
    top5_values = list(top5.counts.values)
    # create visuals
    # Graph 1
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                    width=.5,
                    textfont = {'family' : 'Arial'},
                        marker=dict(color='BlueViolet')
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"

                },
                'padding': 150
                
            },

        }
    ]
    
    # Graph 2
    graphs.append({
    'data': [
        Bar(
            x=category_names,
            y=category_counts,
            orientation = 'v',
            width=.5,
            textfont = {'family' : 'Arial'},
                marker=dict(color='Indigo')
        )
    ],

    'layout': {
        'title': 'Count of Messages in Different Categories',
        'yaxis': {
            'title': "Count"
        },
        'xaxis': {
            'title': "Category",
            'automargin': True
        },
                'padding': 150
    },

    })

    # Graph 3
    graphs.append({
            'data': [
                Bar(
                    y=top5_values,
                    x=top5_names,
                    orientation = 'v',
                    width=.5,
                    textfont = {'family' : 'Arial'},
                        marker=dict(color='navy')
                )
            ],

            'layout': {
                'title': 'Top Five Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                },
                'padding': 1000
            }


        })
        

        
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html  
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    # Note: Use http://localhost:3001
    app.run(host='0.0.0.0', port=3001, debug=True)
    

if __name__ == '__main__':
    main()