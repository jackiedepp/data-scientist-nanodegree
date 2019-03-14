# Import libraries
import sys
import re
import pickle
import pandas as pd
from sqlalchemy import create_engine

import re
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])

from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    '''
    Load data from database as dataframe
    
    Parameters:
        database_filepath: the filepath of sql database
        
    Returns:
        X: Message data 
        y: Categories
        category_names: Category names
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('SELECT * FROM InsertTableName', engine)
    X = df['message']
    y = df.iloc[:, 4:]
    category_names = df.columns[4:]
    
    return X, y, category_names


def tokenize(text):
    '''
    Clean the text
    
    Parameters:
        text: text will be cleaned
        
    Returns
        clean_tokens: list of word from the text cleaned
    '''
    
    # Normalize text
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Remove stopwords
    tokens = [w for w in tokens if w not in stopwords.words('english')]
    
    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        # lemmatize and remove leading/trailing whith space
        clean_token = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_token)
    
    return clean_tokens 


def build_model():
    '''
    Build a ML pipeline model using itidf, random forest and gridsearch
    
    Parameters:
        none
        
    Returns:
        cv: a ML pipeline model
    '''
    
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                         ('tdidf', TfidfTransformer()),
                         ('clf', MultiOutputClassifier(RandomForestClassifier()))
                        ])
    
    parameters = {'clf__estimator__n_estimators': [50, 100],
                  'clf__estimator__min_samples_split': [2,3,4],
                  'clf__estimator__criterion': ['entropy', 'gini']
                 }
    
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate model performance 
    
    Parameters:
        model: Model to be evaluated
        X_test: Test data (features)
        y_test: True labels of X_test
        category_names: Labels of categories
    
    Returns:
        none
    '''
    
    y_pred = model.predict(X_test)
    
    # Caculate the f1-score for each category
    for i in len(category_names):
        col = category_names[i]
        report = classification_report(y_test[col].values, y_pred[:,i])
        score = accuracy_score(y_test[col].values, y_pred[:,i])
        print(f'Category: {col}\n {report}')
        print(f'Accuracy of {col}: {score:.2f}')


def save_model(model, model_filepath):
    '''
    Save model as a pickle file
    Parameters:
        model: Model to be saved
        model_filepath: path of the output pickle file
    
    Returns:
        none
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


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