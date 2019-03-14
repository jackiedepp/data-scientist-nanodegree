# Disaster Response Pipeline Project

### Table of Contents
1. [Introduction](#Introduction)
2. [Prerequisites](#Prerequisites)
3. [File structure](#files)
4. [Results](#Results)
5. [License](#License)
6. [Instructions](#Instructions)

## Introduction <a name="Introduction"></a>

The project applies data engineering skills to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.

## Prerequisites <a name="Prerequisites"></a>

Except the Anaconda distribution of Python, the following packages need to be installed for nltk:
* punkt
* wordnet
* stopwords

## File structure <a name="files"></a>

There are three main foleders:
1. app
    - run.py: Flask file to run the web application
    - templates: Contains html file for the web applicatin
2. data
    - disaster_categories.csv: dataset for the categories 
    - disaster_messages.csv: dataset for the messages
    - process_data.py: ETL pipeline scripts to read, clean, and save data into a database
    - DisasterResponse.db: output of the ETL pipeline
3. models
    - train_classifier.py: machine learning pipeline scripts to train and export a classifier
    - classifier.pkl: output of the machine learning pipeline

## Results <a name="Results"></a>
1. An ETL pipleline was built to read data from two csv files, clean data, and save data into a SQLite database.
2. A machine learning pipepline was developed to train a classifier to performs multi-output classification on the 36 categories in the dataset.
3. A Flask app was created to show data visualization and classify the message that user enters on the web page.

## License <a name="License"></a>

Credits must be given to Udacity for the starter codes and FigureEight for provding the data used by this project. 
## Instructions: <a name="Instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
