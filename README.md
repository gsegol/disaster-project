
### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python.  The code should run with no issues using Python versions 3.*.
The project uses two *.csv files included in this repository.  
The project creates a *.db file to store the cleaned data, and a *.pkl file to store the trained model.

## Project Motivation<a name="motivation"></a>

The purpose of this project is to apply machine-learning techniques to the classification of messages received during a natural disaster.  The ultimate objective is to provide a tool that facilitates the work of help organizations.

## File Descriptions <a name="files"></a>

The project is run in three steps:  
1. Data preparation
  * cd to the data folder
  * Run: python process_data.py disaster_messages.csv disaster_categories.csv data/database_filename.db 
database_filename.db is the output file storing the cleaned data; the name is specified in the command line.
2. Data classification
  * cd to the models folder
  * Run: python train_classifier.py ../data/database_filename.db ../models/model_filename.pkl
model_filename.pkl is the pickle file storing the trained model; the name is specified in the command line.
3. Results
  * cd to the app folder 
  * Run: 'python run.py database_filename.db model_filename.pkl' to create the web app.

## Results<a name="results"></a>

Based on the text analysis, the messages can be separated into 36 categories corresponding to the keywords used. These 36 categories can be grouped into six themes, namely: 

  * Basic needs (such as food and shelter)
  * Medical
  * Weather (earthquake, flood, etc.)
  * Infrastructure (buildings, utilities, etc.)
  * People (missing persons, security, etc.)
  * Other/unspecified

Interestingly, more than half of the messages are in the Other/unspecified category, which shows where more work is needed.  The original messages where in French or local dialect, which introduced another element of uncertainty.

![](https://github.com/gsegol/disaster-project/blob/master/img/disaster_results_sm.png " pg.1 screenshot")

## Licensing, Authors, Acknowledgements<a name="licensing"></a>
The initial data were provided by Figure Eight (now Appen https://appen.com/).  Otherwise, the code is free to use.
