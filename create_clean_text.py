#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 12:12:30 2020

@author: ray
"""

"""
Generate cleaned dataset, saved in dataset file. 
If after cleaning process, comment_text_length < 1, we fill "unkonwn" in comment_text.

"""
import pandas as pd
import numpy as np
import data_helper

train_set = pd.read_csv('dataset/train.csv')
test_set = pd.read_csv('dataset/test.csv')

train_text = train_set.comment_text
test_text = test_set.comment_text

train_clean = []
for text in train_text:
    s = data_helper.clean_text(text)
    if len(s)>0:
        train_clean.append(s)
    else:
        train_clean.append('unknown')
    
test_clean = []
for text in test_text:
    s = data_helper.clean_text(text)
    if len(s)>0:
        test_clean.append(s)
    else:
        test_clean.append('unknown')
        
train_set['comment_text'] = train_clean
test_set['comment_text'] = test_clean

train_set.to_csv('dataset/cleaned_train.csv', index=False)
test_set.to_csv('dataset/cleaned_test.csv', index=False)