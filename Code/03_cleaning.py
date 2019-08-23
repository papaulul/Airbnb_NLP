#%%
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
start_time = time.time()
import spacy
import nltk 
from collections import Counter
# Makes sure we're the correct directory

AWS = True
if AWS: 
    file_path = "data/"
else: 
    os.chdir('/Users/pkim/Dropbox/Projects/Airbnb_NLP')
    file_path = "data/"

LA_reviews = pd.read_csv(file_path+"reviews_LA_7_8.csv")
SF_reviews = pd.read_csv(file_path+"reviews_SF_7_8.csv")
LA_hosting = pd.read_csv(file_path+"LA_nlp_ds.csv")
SF_hosting = pd.read_csv(file_path+"SF_nlp_ds.csv")

#%%
"""
sns.heatmap(LA_reviews.isna())
sns.heatmap(SF_reviews.isna())
sns.heatmap(LA_hosting.isna())
sns.heatmap(SF_hosting.isna())
"""
#%% 
tables = [LA_reviews,SF_reviews,LA_hosting,SF_hosting]
to_zip =  ["LA_reviews","SF_reviews","LA_hosting","SF_hosting"]

for table,table_name in zip(tables, to_zip): 
    tab_len = len(table)
    print(table_name,"\n")
    for columns in table.columns:
        missing_per = sum(table[columns].isna())/tab_len
        print(columns,": ", round(missing_per*100,4),"%")
    print("*"*64)
#%%
for table in tables:
    for columns in table.columns:
        missing_per = sum(table[columns].isna())/tab_len
        if missing_per > 0:
            table[columns+"_missing"] = table[columns].isna()
            table[columns] = table[columns].fillna("")
#%%
columns = ['name','summary','space','description','neighborhood_overview',
          'notes','transit','access','interaction','house_rules',
           'host_about']

for col in columns: 
    LA_hosting[col] = LA_hosting[col].apply(lambda x: 
                                re.sub('[^A-Za-z0-9]+',' ',str(x))
                               )
    SF_hosting[col] = SF_hosting[col].apply(lambda x: 
                                re.sub('[^A-Za-z0-9]+',' ',str(x))
                               )
LA_hosting.head()

#%%
nlp = spacy.load("en")
prefix_re = nlp.Defaults.prefixes
suffix_re = nlp.Defaults.suffixes

tokenizer = spacy.tokenizer.Tokenizer(nlp.vocab,prefix_search=prefix_re,
                                     suffix_search = suffix_re)

#%%
for table in tables:
    for columns in table.select_dtypes("object").columns:
        table[columns+"_tokenized"] = table[columns].apply(lambda col: col.lower()).apply(nlp)\
            .apply(lambda toke: [token for token in toke if not token.is_stop])
        table[columns+"_pos"] = table[columns+"_tokenized"].apply(lambda token: [toke.pos_ for toke in token])
        table[columns+"_noun_count"] = table[columns+"_pos"]\
            .apply(lambda pos: sum([1 for part in pos if part =="NOUN"]))
        table[columns+"_proper_noun_count"] = table[columns+"_pos"]\
            .apply(lambda pos: sum([1 for part in pos if part =="PROPN"]))
        table[columns+"_adj_adv_count"] = table[columns+"_pos"]\
            .apply(lambda pos: sum([1 for part in pos if part in ["ADV","ADJ"]]))
        table[columns+"_verb_count"] = table[columns+"_pos"]\
            .apply(lambda pos: sum([1 for part in pos if part =="VERB"]))
        print(columns, " DONE")
#%%


for table, table_name in zip(tables, to_zip):   
    table.to_csv(table_name+".csv",index=False)
