import sys


def load_data(database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)
    table_name = database_filepath.split("\\")[-1]
    table_name = table_name[:-3]
    query = 'SELECT * FROM '+ table_name
    
    # Read query into a DataFrame
    df = pd.read_sql(sql=query, con=engine)
    
    X = df.message.values
    category_names = [column for column in df.columns if column not in ['id', 'message', 'original', 'genre']]
    Y = df[category_names].values
    
    return X, Y, category_names


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)
        
    return clean_tokens


def build_model():
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(AdaBoostClassifier()))])

    parameters  = {'clf__estimator__n_estimators': 3}
      
    #cv = GridSearchCV(pipeline, param_grid=parameters)
    pipeline.set_params(**parameters)
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as the_file:
        pickle.dump(model, the_file)


def main():
    if len(sys.argv) == 3:
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

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()