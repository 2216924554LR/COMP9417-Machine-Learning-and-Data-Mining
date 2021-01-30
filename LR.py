#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_union
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from test_helper import draw_AOC_ROC_curve

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train_set = pd.read_csv('dataset/cleaned_train.csv')
test_set=pd.read_csv('dataset/cleaned_test.csv')
test_label = pd.read_csv('dataset/test_labels.csv')

train_text = train_set['comment_text']
test_text = test_set['comment_text']
all_text = pd.concat([train_text, test_text])

df1 = test_label.loc[test_label.toxic != -1]
df2 = test_set
df = pd.merge(df1, df2, how='left', on='id')
columnsTitles = ['id','comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult','identity_hate']
test_set1 = df.reindex(columns=columnsTitles)

test_text_not_minus = test_set1['comment_text']

word_vectorizer = TfidfVectorizer(sublinear_tf=True,strip_accents='unicode',analyzer='word',token_pattern=r'\w{2,}',ngram_range=(1, 1),max_features=30000)

word_vectorizer.fit(train_text)
train_features = word_vectorizer.transform(train_text)
test_features = word_vectorizer.transform(test_text)
test_features1 = word_vectorizer.transform(test_text_not_minus)

train_label_pro = train_set.values[:,2:]
train_label_pro = np.asarray(train_label_pro, dtype=int)

test_label_pro = test_set1.values[:,2:]
test_label_pro = np.asarray(test_label_pro, dtype=int)

submission = pd.DataFrame.from_dict({'id': test_set['id']})

predict_list=[]
losses=[]
auc=[]
for i in range(len(class_names)):
    #train_target = train_set[class_name]
    classifier = LogisticRegression(C=1, solver='sag')
    model=classifier.fit(train_features,train_label_pro[:,i])
    predicted_y = model.predict(test_features1)
    predict_list.append(predicted_y)
    cv_score = np.mean(cross_val_score(classifier, train_features, train_label_pro[:,i], cv=5, scoring='accuracy'))
    print('CV Accuracy score for class {} is {}'.format(class_names[i], cv_score))
    cv_loss = np.mean(cross_val_score(classifier, train_features, train_label_pro[:,i], cv=5, scoring='neg_log_loss'))
    losses.append(cv_loss)
    print('CV Log_loss score for class {} is {}'.format(class_names[i], cv_loss))
    auc_score = np.mean(cross_val_score(classifier, train_features, train_label_pro[:,i], cv=5, scoring='roc_auc'))
    auc.append(auc_score)
    print("CV ROC_AUC score is {}\n".format(auc_score))
    print(classification_report(test_label_pro[:,i], predicted_y))
    submission[class_names[i]] = classifier.predict_proba(test_features)[:, 1]
submission.to_csv('submissions.csv', index=False)
print('Total average CV Log_loss score is {}'.format(np.mean(losses)))
print('Total average CV ROC_AUC score is {}'.format(np.mean(auc)))

temp = predict_list[0]
for i in range(1,len(predict_list)):
    temp = np.append(temp, predict_list[i], axis=0)

temp = np.reshape(temp, [len(class_names), -1])

temp = np.transpose(temp)

draw_AOC_ROC_curve(test_label_pro, temp, 'LRNew')