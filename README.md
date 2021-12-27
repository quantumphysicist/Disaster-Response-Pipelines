### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [Instructions](#instructions)
4. [File Descriptions](#files)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>
Anaconda distribution of Python 3.0. 

## Project Motivation<a name="motivation"></a>

I want to show that machine learning can be used to classify messages sent during a disaster into appropriate categories, to help ensure an efficient response by relevant agencies.

## Instructions: <a name="instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run the ETL (Extract, Transform Load) pipeline that cleans data and stores in database  
      `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
        
    - To run the ML (Machine Leazning) pipeline that trains and saves classifier  
      `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.  
   `python run.py`

3. Go to http://0.0.0.0:3001/ (On a windows machine, this is effectively the same as http://localhost:3001/ )

## File Descriptions <a name="files"></a>

### File structure

-- app  
|- template  
| |- master.html  (main page of the web app)      
| |- go.html  (classification result page of the web app)    
|- run.py  (Flask file that runs app)    

-- data  
|- disaster_categories.csv  (data to process)       
|- disaster_messages.csv  (data to process)      
|- process_data.py  (ETL script)    
|- DisasterResponse.db   (database that cleaned data is saved to)     

-- models  
|- train_classifier.py (ML script)   
|- classifier.pkl  (saved ML model)     

### Key files
`process_data.py`
ETL script. This ETL script cleans two datasets and stores the cleaned data in a SQLite database (`DisasterResponse.db`). 
The messages and categories datasets are merged; the category column is split into separate, named columns; the values are converted to binary; and duplicates are dropped. 

`train_classifier.py`
ML script. This ML script creates and trains a classifier and stores the classifier in a pickle file (`classifier.pkl`).
The script builds a pipeline to process text and classifies the text into the categories of the dataset (where more than one category is possible.)
For each category of the test set, the f1 score, precision and recall is outputted.
GridSearchCV is used to find the best parameters for the model.

`run.py`
Creates a web app that can be accessed at http://localhost:3001/.
The web app visualises the dataset on the homepage.
When a message is inputted into the app, the message is classified into one or more categories. 

jupyter notebooks

## Licensing, Authors, and Acknowledgements <a name="licensing"></a>
Thanks to Udacity for the idea for (and structure of) this project, and to Figure8 for making their data available.

