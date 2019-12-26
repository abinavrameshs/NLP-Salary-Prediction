#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 08:45:09 2019

@author: abinavrameshsundararaman
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from nltk.corpus import stopwords
import nltk
from nltk.corpus import reuters
from nltk.corpus import wordnet
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from operator import itemgetter
from sklearn.metrics import classification_report
import csv
import os


salary=pd.read_csv("Train_rev1.csv")

salary=salary.dropna()

# Select random 2500 samples

salary_subset = salary.sample(n=2500)
salary_subset.columns

"""""""""""""""
Part A1-- top 5 parts of speech and how frequently they occur

"""""""""""""""

# remove all non-alphabets

salary_subset["tokens"] = salary_subset["FullDescription"].apply(nltk.word_tokenize)

# keep only aphabets
def include_only_alphas(list1):
    l=list()
    for i in list1 : 
        if(i.isalpha()):
            l.append(i)
    return l
        
salary_subset["tokens"]=salary_subset["tokens"].apply(lambda x :include_only_alphas(x) )
salary_subset["tag_pos"]=salary_subset["tokens"].apply(nltk.pos_tag)
salary_subset['freq_dist']=salary_subset['tag_pos'].apply(lambda x: nltk.FreqDist(tag for (word, tag) in x))


from collections import defaultdict
d = defaultdict(int)

for item in salary_subset['freq_dist']:
    for i in item : 
        d[i] += item[i]


df=pd.DataFrame(list(d.keys()),list(d.values()))
df=df.reset_index()
df.columns=['count','POS']

"""
Part A1-- top 5 parts of speech and how frequently they occur-- if i remove stopwords

"""


stopwords.words('english')

def content_without_stopwords(text):
    stopwords = nltk.corpus.stopwords.words('english')
    content = [w for w in text if w.lower() not in stopwords]
    return content


salary_subset["tokens_without_stop"] = salary_subset["tokens"].apply(lambda x: content_without_stopwords(x))

salary_subset["tag_pos_without_stop"]=salary_subset["tokens_without_stop"].apply(nltk.pos_tag)
salary_subset['freq_dist_without_stop']=salary_subset['tag_pos_without_stop'].apply(lambda x: nltk.FreqDist(tag for (word, tag) in x))


from collections import defaultdict
d_without_stop = defaultdict(int)

for item in salary_subset['freq_dist_without_stop']:
    for i in item : 
        d_without_stop[i] += item[i]


df_without_stop=pd.DataFrame(list(d_without_stop.keys()),list(d_without_stop.values()))
df_without_stop=df_without_stop.reset_index()
df_without_stop.columns=['count','POS']


"""
Part A2-- Does this data support Zipf's law-- without removing stop words

"""


# convert into lower case
def return_list_lower(list1):
    l=list()
    for i in list1 :
        l.append(i.lower())
    return l

salary_subset["tokens_lower"]=salary_subset["tokens"].apply(lambda x: return_list_lower(x))

# create a combined list of words
combined_tokens = list()

for i in salary_subset["tokens_lower"]:
    for j in i: 
        combined_tokens.append(j)
        

frequency = nltk.FreqDist(word for word in combined_tokens)
frequency.most_common(5)

frequency.plot(100,cumulative=False)


######################## Plotting zipf's law-

combined_tokens_set = set(combined_tokens)
counts = [(w, combined_tokens.count(w)) for w in combined_tokens_set]

# sort the tuples in decreasing order
token_df=pd.DataFrame(counts, columns=['token', 'counts'])
token_df = token_df.sort_values('counts',ascending=False)

# Plot for the top 100 words

token_df.iloc[0:100,:].plot.bar(x='token', y='counts')

token_df['rank'] = token_df.counts.rank(ascending=1)


"""
Part A2-- Does this data support Zipf's law-- -- if i remove stopwords

"""

salary_subset["tokens_without_stoplower"]=salary_subset["tokens_without_stop"].apply(lambda x: return_list_lower(x))

# create a combined list of words
combined_tokens_without_stop = list()

for i in salary_subset["tokens_without_stoplower"]:
    for j in i: 
        combined_tokens_without_stop.append(j)
        

frequency_without_stop = nltk.FreqDist(word for word in combined_tokens_without_stop)
frequency_without_stop.most_common(5)

frequency_without_stop.plot(100,cumulative=False)

######################## Plotting zipf's law-- 
combined_tokens_without_stop_set = set(combined_tokens_without_stop)
counts_without_stop_set = [(w, combined_tokens_without_stop.count(w)) for w in combined_tokens_without_stop_set]

# sort the tuples in decreasing order
token_df_without_stop_set=pd.DataFrame(counts_without_stop_set, columns=['token', 'counts'])
token_df_without_stop_set = token_df_without_stop_set.sort_values('counts',ascending=False)

# Plot for the top 100 words

x=np.arange(1, 100, 1)
y=1/x

token_df_without_stop_set.iloc[0:100,:].plot.bar(x='token', y='counts')

token_df_without_stop_set['rank'] = token_df_without_stop_set.counts.rank(ascending=False)

plt.plot(token_df_without_stop_set.iloc[0:100,:]['rank'],token_df_without_stop_set.iloc[0:100,:]['counts'])
plt.plot(token_df_without_stop_set.iloc[0:100,:]['rank'],token_df_without_stop_set.iloc[0:100,:]['counts'])
#plt.yscale('log')
#plt.xscale('log')
plt.plot(token_df_without_stop_set.iloc[0:100,:]['rank'],token_df_without_stop_set.iloc[0:100,:]['counts'])
plt.plot(x,y,c='r')
#plt.yscale('log')
#plt.xscale('log')




"""""""""""""""""""""""""""""""""""""""
Part A3-- If we remove stopwords and lemmatize the data, what are the 10 most common words? What are their frequencies?

"""""""""""""""""""""""""""""""""""""""

wnl = nltk.WordNetLemmatizer()
lemmatized=[wnl.lemmatize(t) for t in combined_tokens_without_stop]

frequency_lemmatized = nltk.FreqDist(word for word in lemmatized)
frequency_lemmatized.most_common(5)




"""""""""""""""""""""""""""""""""""""""
Part B1-- train a model to predict high/low salary from all the numeric columns

"""""""""""""""""""""""""""""""""""""""



salary_subset_B = salary_subset.loc[:,['LocationNormalized','ContractType', 'ContractTime','Category','SalaryNormalized']]


## Creating categorical columns
highest_COL=["London","Oxford","Brighton","Cambridge","Bristol","Reading","Berkshire","York","Portsmouth","Edinburgh","Leeds"]

category_keep = ["Accounting & Finance Jobs","Teaching Jobs","IT Jobs","Engineering Jobs","PR, Advertising & Marketing Jobs","Sales Jobs","Admin Jobs","Other/General Jobs","HR & Recruitment Jobs","Healthcare & Nursing Jobs","Legal Jobs","Consultancy Jobs"]

salary_subset_B = salary_subset_B.assign(Location_high_COL = [1 if a in highest_COL else 0 for a in salary_subset_B['LocationNormalized']])

salary_subset_B = salary_subset_B.assign(category_categorical = [a if a in category_keep else "Other" for a in salary_subset_B['Category']])

salary_75=salary_subset_B.quantile(0.75, numeric_only=True)['SalaryNormalized']
salary_subset_B = salary_subset_B.assign(Salary_categorical = ["high" if a >= salary_75 else "low" for a in salary_subset_B['SalaryNormalized']])

### Creating dummy variables and create a clean dataset

category_dummies=pd.get_dummies(salary_subset_B.category_categorical,prefix='category')

ContractTime_dummies = pd.get_dummies(salary_subset_B.ContractTime,prefix='ContractTime')

ContractType_dummies = pd.get_dummies(salary_subset_B.ContractType,prefix='ContractType')

salary_clean =pd.concat([salary_subset_B.loc[:,['Location_high_COL','Salary_categorical']], category_dummies,ContractTime_dummies,ContractType_dummies], axis=1)



##### Creating a Bernoulli Naive Bayes Classifier
X_classification = salary_clean.loc[:,~salary_clean.columns.isin (['Salary_categorical'])]
y_classification = salary_clean.loc[:,'Salary_categorical']

X_train, X_test, y_train, y_test = train_test_split(X_classification, y_classification, test_size = 0.8, random_state = 1)

bernoulli_nb=BernoulliNB()

model_bernoulli=bernoulli_nb.fit(X_train ,y_train)
y_pred = model_bernoulli.predict(X_test)

accuracy_score(y_test, y_pred)
confusion_matrix(y_test, y_pred)


##### Creating a MultinomialNB Naive Bayes Classifier
X_classification = salary_clean.loc[:,~salary_clean.columns.isin (['Salary_categorical'])]
y_classification = salary_clean.loc[:,'Salary_categorical']

X_train, X_test, y_train, y_test = train_test_split(X_classification, y_classification, test_size = 0.8, random_state = 1)

Multinomial_nb=MultinomialNB()

model_multi=Multinomial_nb.fit(X_train ,y_train)
y_pred = model_multi.predict(X_test)

accuracy_score(y_test, y_pred)
confusion_matrix(y_test, y_pred)

"""""""""""""""""""""""""""""""""""""""
Part B2-- train a model to predict high/low salary from only the text

"""""""""""""""""""""""""""""""""""""""
salary_subset.columns


salary_75=salary_subset.quantile(0.75, numeric_only=True)['SalaryNormalized']

salary_subset = salary_subset.assign(Salary_categorical = ["high" if a >= salary_75 else "low" for a in salary_subset['SalaryNormalized']])

salary_subset_B2 = salary_subset.loc[:,['FullDescription','Salary_categorical']]



salary_list = [(token_list, category) for category in salary_subset.Salary_categorical for token_list in salary_subset.tokens_without_stoplower]




##########################


salary_subset.columns
salary_subset["tokens_with_space"]=salary_subset["tokens_without_stop"].apply(lambda x : " ".join(x) )

vectorizer = TfidfVectorizer(min_df=1, 
 ngram_range=(1, 2), 
 stop_words='english', 
 strip_accents='unicode', 
 norm='l2')

X_train, X_test, y_train, y_test = train_test_split(salary_subset['tokens_with_space'], salary_subset['Salary_categorical'], test_size = 0.5, random_state=1)


X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

nb_classifier = MultinomialNB().fit(X_train, y_train)
y_nb_predicted = nb_classifier.predict(X_test)

accuracy_score(y_test, y_nb_predicted)
confusion_matrix(y_test, y_nb_predicted)

features=vectorizer.get_feature_names()
type(features)
len(features)
### FInd top 10 features of each category


def top_tfidf_feats(row, features, top_n=25):
    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df


def top_feats_in_doc(Xtr, features, row_id, top_n=25):
    row = np.squeeze(Xtr[row_id].toarray())
    return top_tfidf_feats(row, features, top_n)

top_feats_in_doc(X_train,features,2)



def top_mean_feats(Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=25):
    ''' Return the top n features that on average are most important amongst documents in rows
        indentified by indices in grp_ids. '''
    if grp_ids:
        D = Xtr[grp_ids].toarray()
    else:
        D = Xtr.toarray()

    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)


def top_feats_by_class(Xtr, y, features, min_tfidf=0.1, top_n=25):
    ''' Return a list of dfs, where each df holds top_n features and their mean tfidf value
        calculated across documents with the same class label. '''
    dfs = []
    labels = np.unique(y)
    for label in labels:
        ids = np.where(y==label)
        feats_df = top_mean_feats(Xtr, features, ids, min_tfidf=min_tfidf, top_n=top_n)
        feats_df.label = label
        dfs.append(feats_df)
    return dfs

top_feats_by_class(X_test,y_test,features,min_tfidf=0,top_n=10)
np.unique(y_test)






"""""""""""""""""""""""""""""""""""""""
Part B3-- train a model to predict high/low salary from both numeric and text

"""""""""""""""""""""""""""""""""""""""



vectorizer_B3 = TfidfVectorizer(min_df=1, 
 ngram_range=(1, 2), 
 stop_words='english', 
 strip_accents='unicode', 
 norm='l2')


salary_subset_vectorized = vectorizer_B3.fit_transform(salary_subset['tokens_with_space'])

salary_subset_vectorized_df = pd.DataFrame(salary_subset_vectorized.todense())

features_B3=vectorizer_B3.get_feature_names()

salary_subset_vectorized_df.columns = features_B3

salary_subset_vectorized_df=pd.concat([salary_subset_vectorized_df, salary_subset['Salary_categorical']], axis=1, join_axes=[salary_subset['Salary_categorical'].index])




##### Creating a Bernoulli Naive Bayes Classifier

salary_subset_vectorized_df=salary_subset_vectorized_df.fillna(0)

X_classification = salary_subset_vectorized_df.loc[:,~salary_subset_vectorized_df.columns.isin (['Salary_categorical'])]
y_classification = salary_subset_vectorized_df.loc[:,'Salary_categorical']

X_train, X_test, y_train, y_test = train_test_split(X_classification, y_classification, test_size = 0.8, random_state = 1)

X_train=X_train.fillna(0)
X_test=X_test.fillna(0)

bernoulli_nb=BernoulliNB()

model_bernoulli=bernoulli_nb.fit(X_train ,y_train)
y_pred = model_bernoulli.predict(X_test)

accuracy_score(y_test, y_pred)
confusion_matrix(y_test, y_pred)

##### Creating a MultinomialNB Naive Bayes Classifier
X_classification = salary_clean.loc[:,~salary_clean.columns.isin (['Salary_categorical'])]
y_classification = salary_clean.loc[:,'Salary_categorical']

X_train, X_test, y_train, y_test = train_test_split(X_classification, y_classification, test_size = 0.8, random_state = 1)

Multinomial_nb=MultinomialNB()

model_multi=Multinomial_nb.fit(X_train ,y_train)
y_pred = model_multi.predict(X_test)

accuracy_score(y_test, y_pred)

confusion_matrix(y_test, y_pred)

