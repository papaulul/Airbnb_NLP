import re 
import json 
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os
import time
from nltk.tokenize import TreebankWordTokenizer
import spacy
import nltk 
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import multiprocessing
from multiprocessing import Pool
import scipy.sparse as sp
from nltk.util import ngrams
from joblib import Parallel, delayed


def readin(AWS):
    """
    AWS is a boolean whether or not I'm using an EC2 instance  
    returns a list of all the datasets 
    """
    if AWS: 
        # If AWS is true, then we will look over the full dataset
        aws_add = "" 
    else:
        # If it's local, then we look over just the sample set
        aws_add = "_sample"
        os.chdir('/Users/pkim/Dropbox/Projects/Airbnb_NLP')
    file_path = "data/"
    # Reading in all of the files
    LA_reviews = pd.read_csv(file_path+"LA_review"+aws_add+".csv")
    SF_reviews = pd.read_csv(file_path+"SF_review"+aws_add+".csv")
    LA_hosting = pd.read_csv(file_path+"LA_nlp_ds.csv")
    SF_hosting = pd.read_csv(file_path+"SF_nlp_ds.csv")
    return [LA_reviews,LA_hosting],[SF_reviews,SF_hosting]

"""
def heatmaps():
    sns.heatmap(LA_reviews.isna())
    sns.heatmap(SF_reviews.isna())
    sns.heatmap(LA_hosting.isna())
    sns.heatmap(SF_hosting.isna())
"""

def cleaning(tables, to_zip):
    """
    Cleaning tables for 
    """
    for table,table_name in zip(tables, to_zip): 
        tab_len = len(table)
        print(table_name,"\n")
        for columns in table.select_dtypes("object").columns:
            missing_per = sum(table[columns].isna())/tab_len
            print(columns,": ", round(missing_per*100,4),"%")
            if missing_per > 0:
                table[columns+"_missing"] = table[columns].isna()
                table[columns] = table[columns].fillna("")
            table[columns] = table[columns].apply(lambda x: re.sub('[^A-Za-z0-9]+',' ',str(x)))
        print("*"*64)
    return tables

def multi_tokenize(doc):
    x = Parallel(n_jobs=num_partitions)(delayed(tokenize)(line) for line in doc)
    return x

def tokenize(doc):
    tokenizer = TreebankWordTokenizer()
    token = tokenizer.tokenize(doc)
    #token = grams(token)
    return token 

def grams(token):
    two_gram = [" ".join(x) for x in list(ngrams(token,2))]
    #three_gram = [" ".join(x) for x in list(ngrams(token,3))]
    return token+two_gram #+ three_gram

def dummy_token(text):
    return text 

def fit_model(data, column, table_name):
    tfidf_vectorizer = TfidfVectorizer(stop_words=nltk.corpus.stopwords.words('english'),
    tokenizer= dummy_token, lowercase=False, min_df=.1, max_df=.8,max_features=300)
    tfidf = tfidf_vectorizer.fit(data)
    pickle.dump(tfidf, open("data/"+table_name+"_"+column+".pkl","wb"))
    return tfidf

def parallelize_dataframe(df, func):
    a = np.array_split(df, num_partitions)
    del df
    pool = Pool(num_cores)
    #df = pd.concat(pool.map(func, [a,b,c,d,e]))
    df = sp.vstack(pool.map(func, a), format='csr')
    pool.close()
    pool.join()
    return df

def test_func(data):
    #print("Process working on: ",data)
    tfidf_matrix = tfidf_vectorizer.transform(data)
    #return pd.DataFrame(tfidf_matrix.toarray())
    return tfidf_matrix


if __name__ == "__main__":
    # Timing the program 
    start_time = time.time()
    # All of the nltk files needed
    try:
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('stopwords')
    except:
        print("Can't download files")
    # initiate readin
    train,test = readin(True)
    # Changing date to date time
    for table in train+test: 
        try: 
            table['date'] = pd.to_datetime(table['date'])
        except:
            pass 
    # Name of all of the dataframes
    to_zip_train =  ["LA_reviews","LA_hosting"]
    to_zip_test = ["SF_reviews","SF_hosting"]
    # 
    train = cleaning(train, to_zip_train)
    test = cleaning(test,to_zip_test)


    num_cores = multiprocessing.cpu_count()
    num_partitions = num_cores-1 # I like to leave some cores for other
    print(num_partitions)
    for table,table_name in zip(train,to_zip_train):
        for column in table.select_dtypes("object").columns:
            if "name" not in column:
                print(column,",",table_name)
                token_time = time.time()
                
                table[column+"_token"] = multi_tokenize(table[column].values)
                print("Took: %.3f seconds for tokenization" % (time.time() - token_time))
                fit_time = time.time()
                
                tfidf_vectorizer = fit_model(table[column+"_token"], column, table_name)
                #tfidf_vectorizer = fit_model(table[column], column, table_name)
                
                print("Took: %.3f seconds for fitting model" % (time.time() - fit_time))

                transform_time = time.time()

                table[column+"_token"] = parallelize_dataframe(table[column+"_token"], test_func)
                
                print("Took: %.3f seconds for transform model\n" % (time.time() - transform_time))
                
        pickle.dump(table, open("data/"+table_name+".pkl", "wb"))
    
    for table,table_name in zip(test,to_zip_test):
        pickle.dump(table, open("data/"+table_name+".pkl", "wb"))

    print("Took: %.3f seconds total" % (time.time() - start_time))
