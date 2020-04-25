## for this project we have used LSTM (Long Short-Term Memory) neural network which is type of RNN (recurrent neural network).
Below is the step by step procedure.

## we have saved the traind model into 'lstmmodel.sav' so no need to train again if you want to see and test the project.

### to test the system
    1) install dependencies
        -> pip install -r requirements.txt
    2) run 'python main.py' command
    3) enter keyword on which you want to get sentiment
    4) output will be stored in 'demo_clean_test.csv' file
    5) also 3 png files (barchart,piechart,wordcloud) will be created.
    

### to train the model
    1) Download dataset
        -> Please download the dataset from https://www.kaggle.com/kazanova/sentiment140
        -> after downloading make new folder named 'data' and extract it as 'dataset.csv'
    2) split dataset into train and test data
        -> python train-test-split.py
    3) preprocessing data
        -> python preprocessing.py
        -> in preprocessing we did follwing steps.
            -> removed tagged user (i.e. @username)
            -> Removed non-alphabetic characters + spaces + apostrophe
            -> Removed links
            -> Removed stopwords
            -> Lemmatize words
            -> Stem words
    4) now we are ready to train model
        -> python lstm.py
        -> after training model is created into 'lstmmodel.sav' file
        
