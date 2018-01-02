
# coding: utf-8

import numpy as np
import base64
import pandas as pd
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import ensemble, metrics, model_selection, naive_bayes

from sklearn.decomposition import NMF, LatentDirichletAllocation
from matplotlib import pyplot as plt
get_ipython().magic(u'matplotlib inline')
from gensim import corpora, models, similarities
import random
# print "All reached here twice now...."



# In[3]:


train = pd.read_csv("/home/shringi/Downloads/offline_challenge/offline_challenge/xtrain_obfuscated.txt")
print("Number of rows in train dataset : ",train.shape[0])
# print(train.shape)
# print (train)
# train.head()
# train.values.tolist()
lines = [line.rstrip('\n') for line in open("/home/shringi/Downloads/offline_challenge/offline_challenge/xtrain_obfuscated.txt")]
# with open("/home/dupree/Downloads/offline_challenge/offline_challenge/xtrain_obfuscated.txt") as f:
#     lines = f.readlines()
# random.shuffle(lines)
train_data_x = lines[:26010]
test_data_x = lines[26010:]

lines = [line.rstrip('\n') for line in open("/home/shringi/Downloads/offline_challenge/offline_challenge/ytrain.txt")]

# with open("/home/dupree/Downloads/offline_challenge/offline_challenge/ytrain.txt") as f:
#     lines = f.readlines()
# random.shuffle(lines)
train_data_y = lines[:26010]
test_data_y = lines[26010:]

# print len(train_data_x)
# print len(test_data_x)
# print len(train_data_y)
# print len(test_data_y)

print train_data_x[:3]
print train_data_y[:3]


# In[7]:


### Fit transform the tfidf vectorizer ###
tfidf_vec = TfidfVectorizer(ngram_range=(1,5), analyzer='char')
full_tfidf = tfidf_vec.fit_transform(train_data_x + test_data_x)
train_tfidf = tfidf_vec.transform(train_data_x)
test_tfidf = tfidf_vec.transform(test_data_x)
cv_scores = []
pred_full_test = 0
pred_train = np.zeros([26010, 3])




# In[8]:


# print type(full_tfidf)
# print full_tfidf.get_shape()
# print train_tfidf.get_shape()
# print test_tfidf.get_shape()
kf = model_selection.KFold(n_splits=5, shuffle=False, random_state=None)

# for train_index, test_index in kf.split(train_data_x):
# #     print("TRAIN:", train_index, "TEST:", test_index)
#     print train_index, test_index
# #     X_train, X_test = train_data_x[train_index], train_data_x[test_index]
# #     y_train, y_test = train_data_y[train_index], train_data_y[test_index]
#     print len(train_index), len(test_index)

print train_tfidf.shape[0]

train_data_y = np.asarray(train_data_y)
for train_index, val_index in kf.split(train_data_x):
    train_X, val_X = train_tfidf[train_index], train_tfidf[val_index]
    train_y, val_y = train_data_y[train_index], train_data_y[val_index]
    pred_val_y, pred_test_y, model = runMNB(train_X, train_y, val_X, val_y, test_tfidf)
    pred_full_test = pred_full_test + pred_test_y
#     pred_train[val_index,:] = pred_val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))
    print type(pred_test_y)
#     print "dev_X, val_X", dev_X, val_X 
#     print "dev_y, val_y", dev_y, val_y
print("Mean cv score : ", np.mean(cv_scores))
pred_full_test = pred_full_test / 5.
print pred_full_test


# In[66]:


train_data_y = lines[:26010]
test_data_y = lines[26010:]

print len(train_data_x)
print len(test_data_x)
print len(train_data_y)
print len(test_data_y)


# Always start with these features. They work (almost) everytime!
tfv = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='char',
            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1)

# Fitting TF-IDF to both training and test sets (semi-supervised learning)
tfv.fit(list(train_data_x) + list(test_data_x))
xtrain_tfv =  tfv.transform(train_data_x) 
xvalid_tfv = tfv.transform(test_data_x)
# print type(test_data_y)


# In[8]:


def tokenize_chars(s):
        return list(s)
print tokenize_chars(train_data_x[0])


# In[10]:


# Fitting a simple Logistic Regression on TFIDF
from sklearn.linear_model import LogisticRegression

def multiclass_logloss(actual, predicted, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    :param actual: Array containing the actual target classes
    :param predicted: Matrix with class predictions, one probability per class
    """
    # Convert 'actual' to a binary array if it's not already:
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2

    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota


clf = LogisticRegression(C=1.0)
clf.fit(xtrain_tfv, train_data_y)
predictions = clf.predict_proba(xvalid_tfv)

print type(predictions)
print predictions.shape[0]


# In[6]:


import math
print type(predictions)
print predictions.shape[0]
print predictions.shape[1]
print predictions[:4]

# metrics.log_loss(val_y, pred_val_y)
# print ("logloss: %0.3f " % metrics.accuracy_score(np.asarray(test_data_y), np.asarray(predictions.round())))

print ("logloss: %0.3f " % metrics.log_loss(test_data_y, predictions))
# print ("logloss: %0.3f " % metrics.mean_absolute_error(np.asarray(test_data_y), predictions))
print str(math.exp(-(metrics.log_loss(test_data_y, predictions)))) + "% is the accuracy percentage"
# print str(math.exp(-(0.05))) + "% is the accuracy percentage"


# In[11]:


ctv = CountVectorizer(analyzer='char',
            ngram_range=(1, 3))

# Fitting Count Vectorizer to both training and test sets (semi-supervised learning)
ctv.fit(train_data_x + test_data_x)
xtrain_ctv =  ctv.transform(train_data_x) 
xvalid_ctv = ctv.transform(test_data_x)

# Fitting a simple Logistic Regression on Counts
clf = LogisticRegression(C=1.0)
clf.fit(xtrain_ctv, train_data_y)
predictions = clf.predict_proba(xvalid_ctv)

print ("logloss: %0.3f " % metrics.log_loss(np.asarray(test_data_y), predictions))
print str(math.exp(-(metrics.log_loss(test_data_y, predictions)))) + "% is the accuracy percentage"
print test_data_y[:15]
print predictions[:15]


# In[12]:


from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.svm import SVC
import math
# Fitting a simple Naive Bayes on TFIDF
clf = MultinomialNB()
clf.fit(xtrain_tfv, train_data_y)
predictions = clf.predict_proba(xvalid_tfv)

print ("logloss: %0.3f " % metrics.log_loss(np.asarray(test_data_y), predictions))
print str(math.exp(-(metrics.log_loss(test_data_y, predictions)))) + "% is the accuracy percentage"
# Fitting a simple Naive Bayes on Counts
clf = MultinomialNB()
clf.fit(xtrain_ctv, train_data_y)
predictions = clf.predict_proba(xvalid_ctv)

print ("logloss: %0.3f " % metrics.log_loss(np.asarray(test_data_y), predictions))
print str(math.exp(-(metrics.log_loss(test_data_y, predictions)))) + "% is the accuracy percentage"


# Apply SVD, I chose 120 components. 120-200 components are good enough for SVM model.
svd = decomposition.TruncatedSVD(n_components=120)
svd.fit(xtrain_tfv)
xtrain_svd = svd.transform(xtrain_tfv)
xvalid_svd = svd.transform(xvalid_tfv)

# Scale the data obtained from SVD. Renaming variable to reuse without scaling.
scl = preprocessing.StandardScaler()
scl.fit(xtrain_svd)
xtrain_svd_scl = scl.transform(xtrain_svd)
xvalid_svd_scl = scl.transform(xvalid_svd)

print "Starting with fitting SVM"
# Fitting a simple SVM
clf = SVC(C=1.0, probability=True) # since we need probabilities
clf.fit(xtrain_svd_scl, train_data_y)
predictions = clf.predict_proba(xvalid_svd_scl)

print ("logloss: %0.3f " % metrics.log_loss(np.asarray(test_data_y), predictions))
print str(math.exp(-(metrics.log_loss(test_data_y, predictions)))) + "% is the accuracy percentage"


# In[13]:


# Fitting a simple xgboost on tf-idf
import xgboost as xgb
clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 
                        subsample=0.8, nthread=10, learning_rate=0.1)
clf.fit(xtrain_tfv.tocsc(), train_data_y)
predictions = clf.predict_proba(xvalid_tfv.tocsc())

print "Came till multiclass logloss funtion...."
print ("logloss: %0.3f " % metrics.log_loss(np.asarray(test_data_y), predictions))
print str(math.exp(-(metrics.log_loss(test_data_y, predictions)))) + "% is the accuracy percentage"


# In[14]:


# Fitting a simple xgboost on tf-idf svd features
clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 
                        subsample=0.8, nthread=10, learning_rate=0.1)
clf.fit(xtrain_svd, train_data_y)
predictions = clf.predict_proba(xvalid_svd)

print ("logloss: %0.3f " % metrics.log_loss(np.asarray(test_data_y), predictions))
print str(math.exp(-(metrics.log_loss(test_data_y, predictions)))) + "% is the accuracy percentage"


# In[15]:


# Fitting a simple xgboost on tf-idf CountVectorizer features
clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 
                        subsample=0.8, nthread=10, learning_rate=0.1)
clf.fit(xtrain_ctv.tocsc(), train_data_y)
predictions = clf.predict_proba(xvalid_ctv.tocsc())

print ("logloss: %0.3f " % metrics.log_loss(np.asarray(test_data_y), predictions))
print str(math.exp(-(metrics.log_loss(test_data_y, predictions)))) + "% is the accuracy percentage"


# In[69]:


#Grid Searching on available models now
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline

mll_scorer = metrics.make_scorer(metrics.log_loss, greater_is_better=False, needs_proba=True)

# Initialize SVD
svd = TruncatedSVD()
    
# Initialize the standard scaler 
scl = preprocessing.StandardScaler()

# We will use logistic regression here..
lr_model = LogisticRegression()

# Create the pipeline 
clf = pipeline.Pipeline([('svd', svd),
                         ('scl', scl),
                         ('lr', lr_model)])
param_grid = {'svd__n_components' : [120, 180],
              'lr__C': [0.1, 1.0, 10], 
              'lr__penalty': ['l1', 'l2']}


# In[66]:


# Initialize Grid Search Model
from sklearn.model_selection import GridSearchCV
model = GridSearchCV(estimator=clf, param_grid=param_grid, scoring=mll_scorer,
                                 verbose=10, n_jobs=-1, iid=True, refit=True, cv=2)

# Fit Grid Search Model
model.fit(xtrain_tfv, np.asarray(train_data_y))  # we can use the full data here but im only using xtrain
print("Best score: %0.3f" % model.best_score_)
print("Best parameters set:")
best_parameters = model.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))


# In[67]:


# This technique can be used to finetune xgboost or even multinomial naive bayes as below. We will use the tfidf data here:
from sklearn.naive_bayes import MultinomialNB
nb_model = MultinomialNB()

# Create the pipeline 
clf = pipeline.Pipeline([('nb', nb_model)])

# parameter grid
param_grid = {'nb__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

# Initialize Grid Search Model
model = GridSearchCV(estimator=clf, param_grid=param_grid, scoring=mll_scorer,
                                 verbose=10, n_jobs=-1, iid=True, refit=True, cv=2)

# Fit Grid Search Model
model.fit(xtrain_tfv, np.asarray(train_data_y))  # we can use the full data here but im only using xtrain. 
print("Best score: %0.3f" % model.best_score_)
print("Best parameters set:")
best_parameters = model.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))


# In[5]:


from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
# nltk.download()
# stop_words = stopwords.words('english')
from tensorflow.python.client import device_lib

def get_available_devices():  
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]

print(get_available_devices())

# Word Vectors
# load the GloVe vectors in a dictionary:
from tqdm import tqdm
embeddings_index = {}
f = open('/home/shringi/Downloads/glove.840B.300d.txt')
for line in tqdm(f):
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))



# In[9]:


# this function creates a normalized vector for the whole sentence
from nltk import word_tokenize
def sent2vec(s):
    words = str(s).lower().decode('utf-8')
    words = tokenize_chars(words)
#     words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(embeddings_index[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(450)
    return v / np.sqrt((v ** 2).sum())

# create sentence vectors using the above function for training and validation set
xtrain_glove = [sent2vec(x) for x in tqdm(train_data_x)]
xvalid_glove = [sent2vec(x) for x in tqdm(test_data_x)]

xtrain_glove = np.array(xtrain_glove)
xvalid_glove = np.array(xvalid_glove)

print  xtrain_glove[:4]
print  xvalid_glove[:4]


# In[23]:


# Fitting a xgboost on glove features
clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 
                        subsample=0.8, nthread=10, learning_rate=0.1, silent=False)
clf.fit(xtrain_glove, train_data_y)
predictions = clf.predict_proba(xvalid_glove)
print ("logloss: %0.3f " % metrics.log_loss(np.asarray(test_data_y), predictions))
print str(math.exp(-(metrics.log_loss(test_data_y, predictions)))) + "% is the accuracy percentage"


# In[20]:


# Fitting a simple xgboost on glove features
import xgboost as xgb
clf = xgb.XGBClassifier(nthread=10, silent=False)
clf.fit(xtrain_glove, train_data_y)
predictions = clf.predict_proba(xvalid_glove)
print ("logloss: %0.3f " % metrics.log_loss(np.asarray(test_data_y), predictions))
print str(math.exp(-(metrics.log_loss(test_data_y, predictions)))) + "% is the accuracy percentage"



# In[10]:


# Deep Learning application -
# scale the data before any neural net:
scl = preprocessing.StandardScaler()
xtrain_glove_scl = scl.fit_transform(xtrain_glove)
xvalid_glove_scl = scl.transform(xvalid_glove)


# we need to binarize the labels for the neural net
ytrain_enc = np_utils.to_categorical((train_data_y))
yvalid_enc = np_utils.to_categorical((test_data_y))

# create a simple 3 layer sequential neural net
model = Sequential()

model.add(Dense(300, input_dim=300, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(300, activation='relu'))
model.add(Dropout(0.3))
model.add(BatchNormalization())

model.add(Dense(12))
model.add(Activation('softmax'))

# compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam')

model.fit(xtrain_glove_scl, y=ytrain_enc, batch_size=64, 
          epochs=350, verbose=1, 
          validation_data=(xvalid_glove_scl, yvalid_enc))


# In[11]:


# print type(ytrain_enc)
# print ytrain_enc.shape
# print yvalid_enc.shape
print xtrain_glove_scl.shape
# print xtrain_glove_scl[0]
# print xtrain_glove[0]
print len(xtrain_glove[45])



# In[12]:


#USING LSTM's now -----
# using keras tokenizer here
token = text.Tokenizer(num_words=None, char_level=True)
max_len = 450

token.fit_on_texts(list(train_data_x) + list(test_data_x))
xtrain_seq = token.texts_to_sequences(train_data_x)
xvalid_seq = token.texts_to_sequences(test_data_x)

# zero pad the sequences
xtrain_pad = sequence.pad_sequences(xtrain_seq, maxlen=max_len)
xvalid_pad = sequence.pad_sequences(xvalid_seq, maxlen=max_len)

word_index = token.word_index


# In[13]:


print len(xtrain_seq)
print len(xvalid_seq)

print len(xvalid_pad[0])

print xtrain_pad[0]
print word_index


# In[64]:


# create an embedding matrix for the words we have in the dataset
embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in tqdm(word_index.items()):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# we need to binarize the labels for the neural net
ytrain_enc = np_utils.to_categorical((train_data_y))
yvalid_enc = np_utils.to_categorical((test_data_y))        

# A simple LSTM with glove embeddings and two dense layers
model = Sequential()
model.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=max_len,
                     trainable=False))
model.add(SpatialDropout1D(0.3))
model.add(LSTM(100, dropout=0.3, recurrent_dropout=0.3))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(12))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model with early stopping callback
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
model.fit(xtrain_pad, y=ytrain_enc, batch_size=512, epochs=500, 
          verbose=1, validation_data=(xvalid_pad, yvalid_enc))


# In[67]:


# create an embedding matrix for the words we have in the dataset
embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in tqdm(word_index.items()):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# we need to binarize the labels for the neural net
ytrain_enc = np_utils.to_categorical((train_data_y))
yvalid_enc = np_utils.to_categorical((test_data_y))

embedding_matrix.shape
# embedding_matrix[0]



# In[16]:


# A simple bidirectional LSTM with glove embeddings and two dense layers
model = Sequential()
model.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=max_len,
                     trainable=False))
model.add(SpatialDropout1D(0.3))
model.add(Bidirectional(LSTM(300, dropout=0.3, recurrent_dropout=0.3)))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(12))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Fit the model with early stopping callback
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
model.fit(xtrain_pad, y=ytrain_enc, batch_size=64, epochs=200, 
          verbose=1, validation_data=(xvalid_pad, yvalid_enc), callbacks=[earlystop])


# In[18]:


# GRU with glove embeddings and two dense layers
model = Sequential()
model.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=max_len,
                     trainable=False))
model.add(SpatialDropout1D(0.3))
model.add(GRU(300, dropout=0.3, recurrent_dropout=0.3, return_sequences=True))
model.add(GRU(300, dropout=0.3, recurrent_dropout=0.3))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(12))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Fit the model with early stopping callback
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
model.fit(xtrain_pad, y=ytrain_enc, batch_size=64, epochs=100, 
          verbose=1, validation_data=(xvalid_pad, yvalid_enc), callbacks=[earlystop])


# In[29]:


# Training a 1D convnet with existing GloVe features/vectors
model = Sequential()
model.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=max_len,
                     trainable=False))
model.add(Dropout(0.2))

# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
model.add(Conv1D(128,
                 5,
                 activation='relu'))
# we use max pooling:
model.add(MaxPooling1D(5))

model.add(Conv1D(128,
                 5,
                 activation='relu'))
# we use max pooling:
model.add(MaxPooling1D(5))

model.add(Conv1D(128,
                 5,
                 activation='relu'))
# we use max pooling:
model.add(MaxPooling1D(3))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(12, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])


# Fit the model with early stopping callback
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
model.fit(xtrain_pad, y=ytrain_enc, batch_size=64, epochs=100, 
          verbose=1, validation_data=(xvalid_pad, yvalid_enc))



# In[54]:


#Preparing test data for final predictions

lines_pred = [line.rstrip('\n') for line in open("/home/shringi/Downloads/offline_challenge/offline_challenge/xtest_obfuscated.txt")]
# with open("/home/dupree/Downloads/offline_challenge/offline_challenge/xtrain_obfuscated.txt") as f:
#     lines = f.readlines()
# random.shuffle(lines)
pred_data_x = lines_pred
len(pred_data_x)
xpred_seq = token.texts_to_sequences(pred_data_x)

# zero pad the sequences
xpred_pad = sequence.pad_sequences(xpred_seq, maxlen=max_len)

print len(xpred_pad)
# print xpred_pad[0]
# result = model.predict(xpred_pad)
# print result[:5]
predictions = model.predict_classes(xpred_pad)
print predictions[:5]
print (predictions).shape

for class_label in predictions:
    print class_label


# In[59]:


# Using a Convolution1D, with GloVe features
# # scale the data before any neural net:
# scl = preprocessing.StandardScaler()
# xtrain_glove_scl = scl.fit_transform(xtrain_glove)
# xvalid_glove_scl = scl.transform(xvalid_glove)

# model.add(Dense(300, input_dim=300, activation='relu'))
# model.add(Dropout(0.2))
# model.add(BatchNormalization())

# model.add(Dense(300, activation='relu'))
# model.add(Dropout(0.3))
# model.add(BatchNormalization())

# model.add(Dense(12))
# model.add(Activation('softmax'))

# Training a 1D convnet with existing GloVe features/vectors
model = Sequential()

model.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=max_len,
                     trainable=False))
model.add(Dropout(0.2))
# model.add(BatchNormalization())

# word group filters of size filter_length:
model.add(Conv1D(128,
                 5,
                 activation='relu'))
# we use max pooling:
model.add(MaxPooling1D(5))

model.add(Conv1D(128,
                 5,
                 activation='relu'))
# we use max pooling:
model.add(MaxPooling1D(5))

model.add(Conv1D(128,
                 5,
                 activation='relu'))
# we use max pooling:
model.add(MaxPooling1D(3))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(12, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

# happy learning!
# model.fit(x_train, y_train, validation_data=(x_val, y_val),
#           epochs=2, batch_size=128)

# Fit the model with early stopping callback
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
model.fit(xtrain_pad, y=ytrain_enc, batch_size=64, epochs=100, 
          verbose=1, validation_data=(xvalid_pad, yvalid_enc))



