#!/Python27/python
# -*- coding: UTF-8 -*-

from os import path
import os
import pandas as pd
from wordcloud import STOPWORDS
from sklearn import preprocessing
from gensim.models import LdaModel
from Topics import Topics
from gensim import corpora
from sklearn import cross_validation
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
import csv
import numpy as np
import time

start_time = time.time()

# Get path
d = path.dirname(__file__)
pd.options.mode.chained_assignment = None

if not path.exists(d+"/Export"):
    os.makedirs(d+"/Export")

print "Reading train_set.csv and test_set.csv . . .\n"
# df = pd.read_csv("C:/xampp/htdocs/Python/TED2/train_set.csv", header=0, quoting=3, sep="\t")
df = pd.read_csv(d+"/train_set.csv", header=0, quoting=3, sep="\t")
dt = pd.read_csv(d+"/test_set.csv", header=0, quoting=3, sep="\t")

# Appending Title to Content (train_set.csv)
print "train_set.csv ==> appending title to content . . ."
for indexTrain, rowTrain in df.iterrows():
    title = " " + rowTrain['Title']
    df['Content'][indexTrain] += 20 * title

# REMOVE COMMON ENGLISH WORDS
delimiter = " "
vectorizer = CountVectorizer(stop_words='english', max_features=200)
stoplist = vectorizer.get_stop_words()
mystoplist = ["the", "a", "an", "will", "of", "-", ".", "!", ",", "and"]
stopwords = []

for stop in stoplist:
    stopwords.append(stop)
for stop in STOPWORDS:
    stopwords.append(stop)
for stop in mystoplist:
    stopwords.append(stop)

print "Removing common words from train_set.csv . . .\n"
for indexTrain, rowTrain in df.iterrows():
    whitelist = []
    df['Content'][indexTrain] = rowTrain['Content'].lower().split()
    for word in df['Content'][indexTrain]:
        if word not in stopwords:
            if word == "the":
                print word
            whitelist.append(word)
    df['Content'][indexTrain] = delimiter.join(whitelist)

# Appending Title to Content (test_set.csv)
print "test_set.csv ==> appending title to content . . ."
for indexTest, rowTest in dt.iterrows():
    title = " " + rowTest['Title']
    dt['Content'][indexTest] += 20 * title

print "Removing common words from test_set.csv . . .\n"
for indexTest, rowTest in dt.iterrows():
    whitelist = []
    dt['Content'][indexTest] = rowTest['Content'].lower().split()
    for word in dt['Content'][indexTest]:
        if word not in stoplist:
            whitelist.append(word)
    dt['Content'][indexTest] = delimiter.join(whitelist)
########################################################################
# Prepare csvs
print "Creating a 2-dimensional array for EvaluationMetric_10fold_lda_only.csv . . ."
w, h = 6, 5
EvalMatrix = [[0 for x_eval in range(w)] for y_eval in range(h)]

EvalMatrix[0][0] = "Statistic Measure"
EvalMatrix[0][1] = "Naive Bayes"
EvalMatrix[0][2] = "KNN"
EvalMatrix[0][3] = "SVM"
EvalMatrix[0][4] = "Random Forest"
EvalMatrix[0][5] = "My Method - SVM"
EvalMatrix[1][0] = "Accuracy K = 10"
EvalMatrix[2][0] = "Accuracy K = 50"
EvalMatrix[3][0] = "Accuracy K = 100"
EvalMatrix[4][0] = "Accuracy K = 1000"

print "Creating a 2-dimensional array for EvaluationMetric_10fold_ex1_features.csv . . ."
EvalMatrix2 = [[0 for x_eval in range(w)] for y_eval in range(h)]

EvalMatrix2[0][0] = "Statistic Measure"
EvalMatrix2[0][1] = "Naive Bayes"
EvalMatrix2[0][2] = "KNN"
EvalMatrix2[0][3] = "SVM"
EvalMatrix2[0][4] = "Random Forest"
EvalMatrix2[0][5] = "My Method - SVM"
EvalMatrix2[1][0] = "Accuracy K = 10"
EvalMatrix2[2][0] = "Accuracy K = 50"
EvalMatrix2[3][0] = "Accuracy K = 100"
EvalMatrix2[4][0] = "Accuracy K = 1000"

print "Open the 3 files . . .\n"
evaluation = open(d+"/Export/EvaluationMetric_10fold_lda_only.csv", 'wb')
evaluation_ex1 = open(d+"/Export/EvaluationMetric_10fold_ex1_features.csv", 'wb')
categories = open(d+"/Export/testSet_categories.csv", 'wb')
#############################################################################

# PREPROCESSING
print "Preprocessing train_set . . .\n"
# Preprocess the target variable (ex.1)
le = preprocessing.LabelEncoder()
le.fit(df["Category"])
Y_train = le.transform(df["Category"])
X_train = df['Content']

#############################################################################

# Convert docs to a list where elements are a tokens list
print "Converting docs to list . . .\n"
docsList = [document.lower().split() for document in X_train]

# Create Gen-Sim dictionary (Similar to SKLearn vectorizer)
print "Generating Gen-Sim dictionary . . .\n"
dictionary = corpora.Dictionary(docsList)

# Create the Gen-Sim corpus using the vectorizer
print "Generating Gen-Sim corpus . . .\n"
corpus = [dictionary.doc2bow(text.split()) for text in X_train]

#############################################################################
# Train LDA with K = 10
print "Training LDA with K = 10 . . .\n"
lda10 = LdaModel(corpus, id2word=dictionary, num_topics=10)

# GET AND CLEAR TOPICS
print "Cleaning topics for K = 10. . .\n"
topics = []
for topic in lda10.show_topics():
    topics.append(topic[1])

clear_topics10 = Topics(topics)
clear_topics10.clean_topics()

# For every doc get its topic distribution
corpus_lda10 = lda10[corpus]
X10 = []
print "Creating vectors for K = 10. . .\n"
# Create numpy arrays from the GenSim output
for l, t in zip(corpus_lda10, corpus):
    ldaFeatures = np.zeros(10)
    for l_k in l:
        ldaFeatures[l_k[0]] = l_k[1]
    X10.append(ldaFeatures)

#############################################################################
# Train LDA with K = 50
print "Training LDA with K = 50 . . .\n"
lda50 = LdaModel(corpus, id2word=dictionary, num_topics=50)

# GET AND CLEAR TOPICS
print "Cleaning topics for K = 50. . .\n"
topics = []
for topic in lda50.show_topics():
    topics.append(topic[1])

clear_topics50 = Topics(topics)
clear_topics50.clean_topics()

# For every doc get its topic distribution
corpus_lda50 = lda50[corpus]
X50 = []
print "Creating vectors for K = 50. . .\n"
# Create numpy arrays from the GenSim output
for l, t in zip(corpus_lda50, corpus):
    ldaFeatures = np.zeros(50)
    for l_k in l:
        ldaFeatures[l_k[0]] = l_k[1]
    X50.append(ldaFeatures)
#############################################################################
# Train LDA with K = 100
# print "Training LDA with K = 100 . . .\n"
# lda100 = LdaModel(corpus, id2word=dictionary, num_topics=100)
#
# # GET AND CLEAR TOPICS
# print "Cleaning topics for K = 100. . .\n"
# topics = []
# for topic in lda100.show_topics():
#     topics.append(topic[1])
#
# clear_topics100 = Topics(topics)
# clear_topics100.clean_topics()
#
# # For every doc get its topic distribution
# corpus_lda100 = lda100[corpus]
# X100 = []
# print "Creating vectors for K = 100 . . .\n"
# # Create numpy arrays from the GenSim output
# for l, t in zip(corpus_lda100, corpus):
#     ldaFeatures = np.zeros(100)
#     for l_k in l:
#         ldaFeatures[l_k[0]] = l_k[1]
#     X100.append(ldaFeatures)

# #############################################################################
# # Train LDA with K = 1000
# print "Training LDA with K = 1000 . . .\n"
# lda1000 = LdaModel(corpus, id2word=dictionary, num_topics=1000)
#
# # GET AND CLEAR TOPICS
# print "Cleaning topics for K = 1000. . .\n"
# topics = []
# for topic in lda1000.show_topics():
#     topics.append(topic[1])
#
# clear_topics1000 = Topics(topics)
# clear_topics1000.clean_topics()
#
# # For every doc get its topic distribution
# corpus_lda1000 = lda1000[corpus]
# X1000 = []
# print "Creating vectors for K = 1000 . . .\n"
# # Create numpy arrays from the GenSim output
# for l, t in zip(corpus_lda1000, corpus):
#     ldaFeatures = np.zeros(1000)
#     for l_k in l:
#         ldaFeatures[l_k[0]] = l_k[1]
#     X1000.append(ldaFeatures)
# #############################################################################
print "Declaring Multinomial Classifier . . ."
classifier = MultinomialNB()

print "Finding Accuracy for K = 10 . . ."
scores = cross_validation.cross_val_score(classifier, X10, Y_train, cv=10)
EvalMatrix[1][1] = "%0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
print "Finding Accuracy for K = 50 . . .\n"
scores = cross_validation.cross_val_score(classifier, X50, Y_train, cv=10)
EvalMatrix[2][1] = "%0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
# print "Finding Accuracy for K = 100 . . .\n"
# scores = cross_validation.cross_val_score(classifier, X100, Y_train, cv=10)
# EvalMatrix[3][1] = "%0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
# print "Finding Accuracy for K = 1000 . . .\n"
# scores = cross_validation.cross_val_score(classifier, X1000, Y_train, cv=10)
# EvalMatrix[4][1] = "%0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
#############################################################################
print "Declaring K-Nearest Neighbor Classifier . . ."
classifier = KNeighborsClassifier()

print "Finding Accuracy for K = 10 . . ."
scores = cross_validation.cross_val_score(classifier, X10, Y_train, cv=10)
EvalMatrix[1][2] = "%0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
print "Finding Accuracy for K = 50 . . .\n"
scores = cross_validation.cross_val_score(classifier, X50, Y_train, cv=10)
EvalMatrix[2][2] = "%0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
# print "Finding Accuracy for K = 100 . . .\n"
# scores = cross_validation.cross_val_score(classifier, X100, Y_train, cv=10)
# EvalMatrix[3][2] = "%0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
# print "Finding Accuracy for K = 1000 . . .\n"
# scores = cross_validation.cross_val_score(classifier, X1000, Y_train, cv=10)
# EvalMatrix[4][2] = "%0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
###############################################################################
print "Declaring classifier (SVM) . . . "
classifier = svm.SVC(kernel='linear', probability=True)

print "Finding Accuracy for K = 10 . . ."
scores = cross_validation.cross_val_score(classifier, X10, Y_train, cv=10)
EvalMatrix[1][3] = "%0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
print "Finding Accuracy for K = 50 . . .\n"
scores = cross_validation.cross_val_score(classifier, X50, Y_train, cv=10)
EvalMatrix[2][3] = "%0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
# print "Finding Accuracy for K = 100 . . .\n"
# scores = cross_validation.cross_val_score(classifier, X100, Y_train, cv=10)
# EvalMatrix[3][3] = "%0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
# print "Finding Accuracy for K = 1000 . . .\n"
# scores = cross_validation.cross_val_score(classifier, X1000, Y_train, cv=10)
# EvalMatrix[4][3] = "%0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
###############################################################################
print "Declaring Random Forest Classifier . . ."
classifier = RandomForestClassifier()

print "Finding Accuracy for K = 10 . . ."
scores = cross_validation.cross_val_score(classifier, X10, Y_train, cv=10)
EvalMatrix[1][4] = "%0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
print "Finding Accuracy for K = 50 . . .\n"
scores = cross_validation.cross_val_score(classifier, X50, Y_train, cv=10)
EvalMatrix[2][4] = "%0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
# print "Finding Accuracy for K = 100 . . .\n"
# scores = cross_validation.cross_val_score(classifier, X100, Y_train, cv=10)
# EvalMatrix[3][4] = "%0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
# print "Finding Accuracy for K = 1000 . . .\n"
# scores = cross_validation.cross_val_score(classifier, X1000, Y_train, cv=10)
# EvalMatrix[4][4] = "%0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
# ###############################################################################
print "----------------CUSTOMIZED-----------------"
print "Declaring Classifier SVM"
classifier = svm.SVC(kernel='linear', probability=True, shrinking=True)

print "Finding Accuracy for K = 10 . . ."
scores = cross_validation.cross_val_score(classifier, X10, Y_train, cv=10)
EvalMatrix[1][5] = "%0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
print "Finding Accuracy for K = 50 . . .\n"
scores = cross_validation.cross_val_score(classifier, X50, Y_train, cv=10)
EvalMatrix[2][5] = "%0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
# print "Finding Accuracy for K = 100 . . .\n"
# scores = cross_validation.cross_val_score(classifier, X100, Y_train, cv=10)
# EvalMatrix[3][5] = "%0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
# print "Finding Accuracy for K = 1000 . . .\n"
# scores = cross_validation.cross_val_score(classifier, X1000, Y_train, cv=10)
# EvalMatrix[4][5] = "%0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
###############################################################################
le = preprocessing.LabelEncoder()
le.fit(df["Category"])
Y_train10 = le.transform(df["Category"])
X_train10 = df['Content']

print "---------------------------------------"
print "MERGING LDA FEATURES WITH EX1 FEATURES"
print "---------------------------------------\n"
print "Adding LDA features of K = 10 . . ."
for index, corp in enumerate(corpus_lda10):
    for inner_index, prob in enumerate(corp):
        if prob[1] >= 0.75:
            lda_Feature = " " + clear_topics10.get_string_for(prob[0])
            X_train10[index] += lda_Feature * 50

# Vectorizer
print "Vectorizer . . ."
vectorizer10 = CountVectorizer(stop_words='english', max_features=200)
X_train10 = vectorizer10.fit_transform(X_train10, Y_train10)

# Transformer
print "Transformer . . .\n"
transformer10 = TfidfTransformer()
X_train10 = transformer10.fit_transform(X_train10, Y_train10)

###############################################################################
print "Declaring Multinomial Classifier . . ."
classifier = MultinomialNB()

print "Finding Accuracy for K = 10 . . .\n"
scores = cross_validation.cross_val_score(classifier, X_train10, Y_train10, cv=10)
EvalMatrix2[1][1] = "%0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
###############################################################################
# SVD
print "SVD . . .\n"
svd10 = TruncatedSVD(n_components=100, random_state=42)
X_train10 = svd10.fit_transform(X_train10, Y_train10)
###############################################################################
print "Declaring K-Nearest Neighbor Classifier . . ."
classifier = KNeighborsClassifier()

print "Finding Accuracy for K = 10 . . .\n"
scores = cross_validation.cross_val_score(classifier, X_train10, Y_train10, cv=10)
EvalMatrix2[1][2] = "%0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
###############################################################################
print "Declaring classifier (SVM) . . . "
classifier = svm.SVC(kernel='linear', probability=True)

print "Finding Accuracy for K = 10 . . .\n"
scores = cross_validation.cross_val_score(classifier, X_train10, Y_train10, cv=10)
EvalMatrix2[1][3] = "%0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
###############################################################################
print "Declaring Random Forest Classifier . . ."
classifier = RandomForestClassifier()

print "Finding Accuracy for K = 10 . . .\n"
scores = cross_validation.cross_val_score(classifier, X_train10, Y_train10, cv=10)
EvalMatrix2[1][4] = "%0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
###############################################################################
print "----------------CUSTOMIZED-----------------"
df = pd.read_csv(d+"/train_set.csv", header=0, quoting=3, sep="\t")

# Appending Title to Content (train_set.csv)
print "train_set.csv ==> appending title to content . . ."
for indexTrain, rowTrain in df.iterrows():
    title = " " + rowTrain['Title']
    df['Content'][indexTrain] += 20 * title

print "Removing common words from train_set.csv . . ."
for indexTrain, rowTrain in df.iterrows():
    whitelist = []
    df['Content'][indexTrain] = rowTrain['Content'].lower().split()
    for word in df['Content'][indexTrain]:
        if word not in stopwords:
            if word == "the":
                print word
            whitelist.append(word)
    df['Content'][indexTrain] = delimiter.join(whitelist)

le = preprocessing.LabelEncoder()
le.fit(df["Category"])
Y_train_cust10 = le.transform(df["Category"])
X_train_cust10 = df['Content']

print "Adding LDA features of K = 10 . . ."
for index, corp in enumerate(corpus_lda10):
    for inner_index, prob in enumerate(corp):
        if prob[1] >= 0.75:
            lda_Feature = " " + clear_topics10.get_string_for(prob[0])
            X_train_cust10[index] += lda_Feature * 50

print "Vectorizer . . ."
vector_cust10 = CountVectorizer(stop_words='english', max_features=500)
X_train_cust10 = vector_cust10.fit_transform(X_train_cust10, Y_train_cust10)

print "Transformer . . ."
transf_cust10 = TfidfTransformer()
X_train_cust10 = transf_cust10.fit_transform(X_train_cust10, Y_train_cust10)

print "SVD . . ."
svd_cust10 = TruncatedSVD(n_components=300, random_state=42)
X_train_cust10 = svd_cust10.fit_transform(X_train_cust10, Y_train_cust10)

print "Declaring Classifier SVM"
classifier = svm.SVC(kernel='linear', probability=True, shrinking=True)

print "Finding Accuracy for K = 10 . . .\n"
scores = cross_validation.cross_val_score(classifier, X_train_cust10, Y_train_cust10, cv=10)
EvalMatrix2[1][5] = "%0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
###############################################################################


###############################################################################
df = pd.read_csv(d+"/train_set.csv", header=0, quoting=3, sep="\t")

# Appending Title to Content (train_set.csv)
print "train_set.csv ==> appending title to content . . ."
for indexTrain, rowTrain in df.iterrows():
    title = " " + rowTrain['Title']
    df['Content'][indexTrain] += 20 * title

print "Removing common words from train_set.csv . . .\n"
for indexTrain, rowTrain in df.iterrows():
    whitelist = []
    df['Content'][indexTrain] = rowTrain['Content'].lower().split()
    for word in df['Content'][indexTrain]:
        if word not in stopwords:
            if word == "the":
                print word
            whitelist.append(word)
    df['Content'][indexTrain] = delimiter.join(whitelist)

le = preprocessing.LabelEncoder()
le.fit(df["Category"])
Y_train50 = le.transform(df["Category"])
X_train50 = df['Content']

print "---------------------------------------"
print "MERGING LDA FEATURES WITH EX1 FEATURES"
print "---------------------------------------\n"
print "Adding LDA features of K = 50 . . ."
for index, corp in enumerate(corpus_lda50):
    for inner_index, prob in enumerate(corp):
        if prob[1] >= 0.75:
            lda_Feature = " " + clear_topics50.get_string_for(prob[0])
            X_train50[index] += lda_Feature * 50

# Vectorizer
print "Vectorizer . . ."
vectorizer50 = CountVectorizer(stop_words='english', max_features=200)
X_train50 = vectorizer50.fit_transform(X_train50, Y_train50)

# Transformer
print "Transformer . . .\n"
transformer50 = TfidfTransformer()
X_train50 = transformer50.fit_transform(X_train50, Y_train50)
###############################################################################
print "Declaring Multinomial Classifier . . ."
classifier = MultinomialNB()

print "Finding Accuracy for K = 50 . . .\n"
scores = cross_validation.cross_val_score(classifier, X_train50, Y_train50, cv=10)
EvalMatrix2[2][1] = "%0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
###############################################################################
# SVD
print "SVD . . .\n"
svd50 = TruncatedSVD(n_components=100, random_state=42)
X_train50 = svd50.fit_transform(X_train50, Y_train50)
###############################################################################
print "Declaring K-Nearest Neighbor Classifier . . ."
classifier = KNeighborsClassifier()

print "Finding Accuracy for K = 50 . . .\n"
scores = cross_validation.cross_val_score(classifier, X_train50, Y_train50, cv=10)
EvalMatrix2[2][2] = "%0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
###############################################################################
print "Declaring classifier (SVM) . . . "
classifier = svm.SVC(kernel='linear', probability=True)

print "Finding Accuracy for K = 50 . . .\n"
scores = cross_validation.cross_val_score(classifier, X_train50, Y_train50, cv=10)
EvalMatrix2[2][3] = "%0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
###############################################################################
print "Declaring Random Forest Classifier . . ."
classifier = RandomForestClassifier()

print "Finding Accuracy for K = 50 . . .\n"
scores = cross_validation.cross_val_score(classifier, X_train50, Y_train50, cv=10)
EvalMatrix2[2][4] = "%0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
###############################################################################
print "----------------CUSTOMIZED-----------------"
df = pd.read_csv(d+"/train_set.csv", header=0, quoting=3, sep="\t")

# Appending Title to Content (train_set.csv)
print "train_set.csv ==> appending title to content . . ."
for indexTrain, rowTrain in df.iterrows():
    title = " " + rowTrain['Title']
    df['Content'][indexTrain] += 20 * title

print "Removing common words from train_set.csv . . ."
for indexTrain, rowTrain in df.iterrows():
    whitelist = []
    df['Content'][indexTrain] = rowTrain['Content'].lower().split()
    for word in df['Content'][indexTrain]:
        if word not in stopwords:
            if word == "the":
                print word
            whitelist.append(word)
    df['Content'][indexTrain] = delimiter.join(whitelist)

le = preprocessing.LabelEncoder()
le.fit(df["Category"])
Y_train_cust50 = le.transform(df["Category"])
X_train_cust50 = df['Content']

print "Adding LDA features of K = 50 . . ."
for index, corp in enumerate(corpus_lda50):
    for inner_index, prob in enumerate(corp):
        if prob[1] >= 0.75:
            lda_Feature = " " + clear_topics50.get_string_for(prob[0])
            X_train_cust50[index] += lda_Feature * 50

print "Vectorizer . . ."
vector_cust50 = CountVectorizer(stop_words='english', max_features=500)
X_train_cust50 = vector_cust50.fit_transform(X_train_cust50, Y_train_cust50)

print "Transformer . . ."
transf_cust50 = TfidfTransformer()
X_train_cust50 = transf_cust50.fit_transform(X_train_cust50, Y_train_cust50)

print "SVD . . ."
svd_cust50 = TruncatedSVD(n_components=300, random_state=42)
X_train_cust50 = svd_cust50.fit_transform(X_train_cust50, Y_train_cust50)

print "Declaring Classifier SVM"
classifier = svm.SVC(kernel='linear', probability=True, shrinking=True)

print "Finding Accuracy for K = 50 . . .\n"
scores = cross_validation.cross_val_score(classifier, X_train_cust50, Y_train_cust50, cv=10)
EvalMatrix2[2][5] = "%0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
###############################################################################
print "Writing to EvaluationMetric_10fold_lda_only.csv . . .\n"
wr = csv.writer(evaluation, delimiter=',', quoting=csv.QUOTE_ALL)
for values in EvalMatrix:
    wr.writerow(values)
###############################################################################
print "Writing to EvaluationMetric_10fold_ex1_features.csv . . .\n"
wr = csv.writer(evaluation_ex1, delimiter=',', quoting=csv.QUOTE_ALL)
for values in EvalMatrix2:
    wr.writerow(values)
###############################################################################
df = pd.read_csv(d+"/train_set.csv", header=0, quoting=3, sep="\t")

# Appending Title to Content (train_set.csv)
print "train_set.csv ==> appending title to content . . ."
for indexTrain, rowTrain in df.iterrows():
    title = " " + rowTrain['Title']
    df['Content'][indexTrain] += 20 * title

print "Removing common words from train_set.csv . . .\n"
for indexTrain, rowTrain in df.iterrows():
    whitelist = []
    df['Content'][indexTrain] = rowTrain['Content'].lower().split()
    for word in df['Content'][indexTrain]:
        if word not in stopwords:
            if word == "the":
                print word
            whitelist.append(word)
    df['Content'][indexTrain] = delimiter.join(whitelist)

print "Adding LDA features of K = 50 to train_set. . ."
for index, corp in enumerate(corpus_lda50):
    for inner_index, prob in enumerate(corp):
        if prob[1] >= 0.75:
            lda_Feature = " " + clear_topics50.get_string_for(prob[0])
            df['Content'][index] += lda_Feature * 50

le_test = preprocessing.LabelEncoder()
le_test.fit(df["Category"])
Y_train = le_test.transform(df["Category"])
X_train = df['Content']
X_test = dt['Content']
X_test_copy = X_test[:]
X_test_id = dt['Id']

print "Converting docs to list . . .\n"
docsList_test = [document.lower().split() for document in X_test_copy]

# Create Gen-Sim dictionary (Similar to SKLearn vectorizer)
print "Generating Gen-Sim dictionary . . .\n"
dictionary_test = corpora.Dictionary(docsList_test)

# Create the Gen-Sim corpus using the vectorizer
print "Generating Gen-Sim corpus . . .\n"
corpus_test = [dictionary_test.doc2bow(text.split()) for text in X_test_copy]
print "Training LDA with K = 50 for test_set. . .\n"
lda_test = LdaModel(corpus_test, id2word=dictionary_test, num_topics=50)

# GET AND CLEAR TOPICS
print "Cleaning topics for K = 50 for test . . .\n"
topics = []
for topic in lda_test.show_topics():
    topics.append(topic[1])

clear_topics_test = Topics(topics)
corpus_lda_test = lda_test[corpus_test]

print "Adding LDA features of K = 50 to test_set. . .\n"
for index, corp in enumerate(corpus_lda_test):
    for inner_index, prob in enumerate(corp):
        if prob[1] >= 0.75:
            lda_Feature = " " + clear_topics_test.get_string_for(prob[0])
            dt['Content'][index] += lda_Feature * 50

vector = CountVectorizer(stop_words='english', max_features=500)

transf = TfidfTransformer()

svd_cust = TruncatedSVD(n_components=300, random_state=42)

classifier = svm.SVC(kernel='linear')

print "Pipeline . . ."
pipeline = Pipeline([
    ('vect', vector),
    ('tfidf', transf),
    ('svd', svd_cust),
    ('clf', classifier)
])

print "Pipeline fit . . ."
pipeline.fit(X_train, Y_train)
print "Predicting categories . . .\n"
predicted = pipeline.predict(X_test)
predicted = le_test.inverse_transform(predicted)

print "Writing to testSet_categories.csv . . .\n"
wr = csv.writer(categories, delimiter=',', quoting=csv.QUOTE_ALL)
wr.writerow(["ID", "Predicted Category"])
for i in range(0, len(predicted)):
    values = [X_test_id[i], predicted[i]]
    wr.writerow(values)

###############################################################################
print("--- %s seconds ---" % (time.time() - start_time))
