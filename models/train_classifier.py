import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import pickle

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier



def load_data(database_filepath):
    
    # use sql
    #engine = create_engine('sqlite:///disasterdb.db')
    #engine = create_engine(database_filepath)
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    tablenames = engine.table_names()
    print(tablenames)
    df = pd.read_sql_table(tablenames[0], engine)
    
    tablenames = engine.table_names()
    #print(tablenames)
    # use csv
    #df = pd.read_csv(database_filepath)
    #check sample
    print(df.loc[0:10, 'message'])
 
    X = df['message']
    y = df[df.columns[4:]]
    #y = df[df.columns[5:]]
    #
    print("X and y dimensions: ", X.shape, " ", y.shape)

    df.head()
    ylabels = df.columns[4:]
    #ylabels = df.columns[5:]
    print("Columns headers for y:\n", ylabels)
    return X, y, ylabels


def tokenize(text):
    stop_words = stopwords.words("english")
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    txt = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(word) for word in txt if word not in stop_words]
    
    return clean_tokens


def build_model(classifier):
    parameters = {
        'tfidf__use_idf': (True, False),
        'vect__max_features': (None, 5000), 
        'clf__estimator__n_estimators': [100, 200] 
    }
    pipeline = Pipeline([
        
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(classifier))
    ])
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2)
    
    return cv

def display_results(Y_pred, Y_true, labels):
    Y_pred_ = pd.DataFrame(Y_pred, columns=labels)
    report = {}
    for i, var in enumerate(labels): 
        print(var)
        print(classification_report(Y_true.iloc[:,i], Y_pred_.iloc[:,i]))
        report[var] = classification_report(Y_true.iloc[:,i], Y_pred_.iloc[:,i])
    return report
  

def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    report = display_results(Y_pred, Y_test, category_names)


def save_model(model, model_filepath):
    
    pkl_filename = model_filepath
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        classifier = AdaBoostClassifier()
        model = build_model(classifier)
        
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