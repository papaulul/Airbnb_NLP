import json 
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os
import time
start_time = time.time()
import spacy
spacy.load("en")
# Makes sure we're the correct directory
os.chdir('/Users/pkim/Dropbox/Projects/Airbnb_NLP')
print(os.getcwd())

"""
text_cols = ['name','summary','space','description',
             'neighborhood_overview','notes','transit',
             'access','interaction','house_rules','host_about'
            ]
id_cols = ['id','host_id','host_name']
nlp_dataset = population[text_cols + id_cols]
nlp_dataset.to_csv('files/july19/LA_nlp_ds.csv',index=False)
"""
# Datasets in use
LA_reviews = pd.read_csv("data/reviews_LA_7_8.csv")
SF_reviews = pd.read_csv("data/reviews_SF_7_8.csv")
LA_hosting = pd.read_csv("data/LA_nlp_ds.csv")
SF_hosting = pd.read_csv("data/SF_nlp_ds.csv")
