#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
In test dataset, the label include -1. labels for the test data; value of -1 indicates it was not used for scoring.
If you want to generate a submmit file, set flag = True
If you want to see metrics set flag = False
"""


import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

flag = False

train_set = pd.read_csv('dataset/cleaned_train.csv')
test_set = pd.read_csv('dataset/cleaned_test.csv')
test_label = pd.read_csv('dataset/test_labels.csv')

if flag:    
    df1 = test_label
    df2 = test_set
    df = pd.merge(df1, df2, how='left', on='id')
    columnsTitles = ['id','comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult',
           'identity_hate']
    test_set = df.reindex(columns=columnsTitles)

    
else:
    df1 = test_label.loc[test_label.toxic != -1]
    df2 = test_set
    df = pd.merge(df1, df2, how='left', on='id')
    columnsTitles = ['id','comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult',
           'identity_hate']
    test_set = df.reindex(columns=columnsTitles)


label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

count = CountVectorizer(max_features = 8000)
train_bag_of_words = count.fit_transform(train_set.comment_text)
test_bag_of_words = count.transform(test_set.comment_text)

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score

test_label_pro = test_set.values[:,2:]
test_label_pro = np.asarray(test_label_pro, dtype=int)


train_label_pro = train_set.values[:,2:]
train_label_pro = np.asarray(train_label_pro, dtype=int)

p_list = []



if flag:  
    for i in range(len(label_cols)):
        clf = MultinomialNB()
        model = clf.fit(train_bag_of_words, train_label_pro[:,i])
        predicted_y = model.predict(test_bag_of_words)
        p_list.append(predicted_y)
    
    temp = p_list[0]
    for i in range(1,len(p_list)):
        temp = np.append(temp, p_list[i], axis=0)
    
    temp = np.reshape(temp, [len(label_cols), -1])
    
    temp = np.transpose(temp)

    sample_submission = pd.read_csv('dataset/sample_submission.csv')
    sample_submission[label_cols] = temp
    sample_submission.to_csv('mnb_submission.csv', index=False)
    
else:
    for i in range(len(label_cols)):
        print(f"------{label_cols[i]}------")
        clf = MultinomialNB()
        model = clf.fit(train_bag_of_words, train_label_pro[:,i])
        predicted_y = model.predict(test_bag_of_words)
        p_list.append(predicted_y)
        print(classification_report(test_label_pro[:,i], predicted_y))
        print('roc: ', roc_auc_score(test_label_pro[:,i], predicted_y))




