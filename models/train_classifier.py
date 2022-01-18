"""
Machine learning (ML) script that creates and trains a classifier 
and stores the classifier in a pickle file.

Note: This script takes between 5-15 minutes to run to completion, 
depending on the specification of the machine it is run on. 
"""
import time # to time how long training & saving take

import nltk 
import sys
import pandas as pd
import pickle # for saving the serialized ML model to a file.
from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,  GridSearchCV 

from sklearn.metrics import classification_report

def load_data(database_filepath):
    """
    Reads SQLite database database into a pandas DataFrame
    
    Parameters
    ----------
    database_filepath : sql file location
    
        This sql file must contain a table named 'DisasterResponse'

    Returns
    -------
    X : list of messages
    
    Y : list of lists of values for each category  
    
    category_names : list of category names
    """

    engine = create_engine('sqlite:///'+database_filepath)  
    df = pd.read_sql_table('DisasterResponse', engine)
    
    X = df.message
    category_names = [column for column in df.columns if column not in ['index', 'id', 'message', 'original', 'genre']]
    Y = df[category_names]
    
    return X, Y, category_names


def tokenize(text):
    """
    Tokenizes, lemmatizes and case normalizes each word in a piece of text.
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)
        
    return clean_tokens


def build_model():
    """
    Returns a pipeline that processes text and classifies the text into the correct categories.
    
    Uses GridSearchCV to find the best parameters.
    """
    
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(AdaBoostClassifier()))])

    parameters  = {'clf__estimator__n_estimators': [5,10,15]}
        
    # Use GridSearch CV to find the best parameters for the model.    
    cv = GridSearchCV(pipeline, param_grid=parameters)
    #pipeline.set_params(**parameters)
    
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Reports the accuracy of the model.
    Also reports the f1 score, precision and recall for each output category of the dataset.
    """
    
    Y_pred = model.predict(X_test)
   
    for i, a_column in enumerate(category_names):
        print(a_column)
        print(classification_report(Y_test[a_column], Y_pred[:,i]))
          
    accuracy = (Y_pred == Y_test).mean()
    print("Overall Accuracy")
    print(accuracy.mean())


def save_model(model, model_filepath):
    """
    Saves the classifier model into a pickle file.
    
    Parameters
    ----------
    model_filepath : pickle file location
    """
    
    with open(model_filepath, 'wb') as the_file:
        pickle.dump(model, the_file)


def main():
    if len(sys.argv) == 3:
        t0 = time.time()
        print('Timer started...')
        print('Please note: This script takes between 5-15 minutes to run to completion, '\
        'depending on the specification of the machine it is run on.')
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')
        
        t1 = time.time()
        print('Timer stopped...')
        print('Time taken in seconds:')
        print(t1 - t0)
    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
