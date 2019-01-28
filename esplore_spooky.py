
"""
PREMLIMINARY
"""

# The imports
import pandas as pd

import numpy as np

import xgboost as xgb

from tqdm import tqdm

from sklearn.svm import SVC
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping

from nltk import word_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words('english')


# importing the data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample = pd.read_csv('sample_submission.csv')


# Implementing the Log-Loss function (even though I'm pretty sure it's in xgboost and sklearn)
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


# use LabelEncoder from scklearn to convert test labels to integers
lbl_enc = preprocessing.LabelEncoder()
y = lbl_enc.fit_transform(train.author.values)


# split the data into training and validation sets
xtrain, xvalid, ytrain, yvalid = train_test_split(train.text.values, y, stratify=y, test_size=0.1)


"""
BASIC MODELS
"""
# First, a TF-IDF
""" http://blog.christianperone.com/2011/09/machine-learning-text-feature-extraction-tf-idf-part-i/ """

# Always start with these features. They work (almost) everytime!
tfv = TfidfVectorizer(min_df=3,  max_features=None,
                      strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                      ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,
                      stop_words='english')

# Fitting TF-IDF to both training and test sets (semi-supervised learning)
tfv.fit(list(xtrain) + list(xvalid))
xtrain_tfv = tfv.transform(xtrain)
xvalid_tfv = tfv.transform(xvalid)


# Fitting a simple Logistic Regression on TFIDF
clf = LogisticRegression(C=1.0)
clf.fit(xtrain_tfv, ytrain)
predictions = clf.predict_proba(xvalid_tfv)

print("Logistic Regression on TFIDF logloss: %0.3f " % multiclass_logloss(yvalid, predictions))


# scikit-learn's Count Vectorizer
ctv = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}',
                      ngram_range=(1, 3), stop_words='english')

# Fitting Count Vectorizer to both training and test sets (semi-supervised learning)
ctv.fit(list(xtrain) + list(xvalid))
xtrain_ctv = ctv.transform(xtrain)
xvalid_ctv = ctv.transform(xvalid)

# Fitting a simple Logistic Regression on Count Vectorizer
clf = LogisticRegression(C=1.0)
clf.fit(xtrain_ctv, ytrain)
predictions = clf.predict_proba(xvalid_ctv)

print("Logistic Regression on Count Vectorizer logloss: %0.3f " % multiclass_logloss(yvalid, predictions))
""" An improvement because maybe stopwords show up less frequently in snippets or else they are just
    more predictive of Author than they would be for for example search relevance (this is what tfidf is for) """


# Naive Bayes
""" Now we take the sparse matrices produced by our first two models and apply naive Bayes instead of logistic regression """
# Fitting a simple Naive Bayes on TFIDF
clf = MultinomialNB()
clf.fit(xtrain_tfv, ytrain)
predictions = clf.predict_proba(xvalid_tfv)

print("Naive Bayes on TFIDF logloss: %0.3f " % multiclass_logloss(yvalid, predictions))


# Fitting a simple Naive Bayes on Counts
clf = MultinomialNB()
clf.fit(xtrain_ctv, ytrain)
predictions = clf.predict_proba(xvalid_ctv)

print("Naive Bayes on Counts logloss: %0.3f " % multiclass_logloss(yvalid, predictions))


# SVM

# Reducing the number of features using Singular Value Decomposition because SVM takes a long time
svd = decomposition.TruncatedSVD(n_components=120)
svd.fit(xtrain_tfv)
xtrain_svd = svd.transform(xtrain_tfv)
xvalid_svd = svd.transform(xvalid_tfv)

# Scale the data obtained from SVD. Renaming variable to reuse without scaling.
scl = preprocessing.StandardScaler()
scl.fit(xtrain_svd)
xtrain_svd_scl = scl.transform(xtrain_svd)
xvalid_svd_scl = scl.transform(xvalid_svd)

# Fitting a simple SVM
clf = SVC(C=1.0, probability=True)  # since we need probabilities
clf.fit(xtrain_svd_scl, ytrain)
predictions = clf.predict_proba(xvalid_svd_scl)

print("SVM logloss: %0.3f " % multiclass_logloss(yvalid, predictions))

# And for fun, SVM on CountVectoriser
svd = decomposition.TruncatedSVD(n_components=120)
svd.fit(xtrain_ctv)
xtrain_svd = svd.transform(xtrain_ctv)
xvalid_svd = svd.transform(xvalid_ctv)

scl = preprocessing.StandardScaler()
scl.fit(xtrain_svd)
xtrain_svd_scl = scl.transform(xtrain_svd)
xvalid_svd_scl = scl.transform(xvalid_svd)

clf.fit(xtrain_svd_scl, ytrain)
predictions = clf.predict_proba(xvalid_svd_scl)

print("SVM on CountVectoriser logloss: %0.3f " % multiclass_logloss(yvalid, predictions))


# XGBoost
""" http://xgboost.readthedocs.io/en/latest/model.html """
# Fitting a simple xgboost on tf-idf
clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8,
                        subsample=0.8, nthread=10, learning_rate=0.1)
clf.fit(xtrain_tfv.tocsc(), ytrain)
predictions = clf.predict_proba(xvalid_tfv.tocsc())

print("XGBoost logloss: %0.3f " % multiclass_logloss(yvalid, predictions))


# Fitting a simple xgboost on tf-idf svd features
clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8,
                        subsample=0.8, nthread=10, learning_rate=0.1)
clf.fit(xtrain_svd, ytrain)
predictions = clf.predict_proba(xvalid_svd)

print("xgboost on tf-idf svd logloss: %0.3f " % multiclass_logloss(yvalid, predictions))


# Hyperparameter Optimization of XGBoost

# Grid Search

# Making a scorer

mll_scorer = metrics.make_scorer(multiclass_logloss, greater_is_better=False,
                                 needs_proba=True)

# Making a pipeline

""" Initialize SVD """
svd = TruncatedSVD()

""" Initialize the standard scaler """
scl = preprocessing.StandardScaler()

""" Logistic Regression """
lr_model = LogisticRegression()

""" Create the pipeline """
clf = pipeline.Pipeline([('svd', svd),
                         ('scl', scl),
                         ('lr', lr_model)])

# A grid of parameters:

param_grid = {'svd__n_components': [120, 180],
              'lr__C': [0.1, 1.0, 10],
              'lr__penalty': ['l1', 'l2']}

# Initialize the grid search model

model = GridSearchCV(estimator=clf, param_grid=param_grid, scoring=mll_scorer,
                     verbose=10, n_jobs=-1, iid=True, refit=True, cv=2)

# Fit grid search model
model.fit(xtrain_tfv, ytrain)

print("Best parameters set:")
best_parameters = model.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

print("Best score: %0.3f" % model.best_score_)

"""
As is often the case in multi-class text classification problems, Naive Bayes has outperformed all the other
more complicated models, including SVM and hyperoptimized grid search model. It is interesting that it performed better
on the count vectorizer data than the TF-IDF data, this could be because
1) There is a randomness to the train/test split and it just happened to split in a way that made the TF-IDF have a 
    higher log-loss score.
2) The stop-words are actually important in distinguishing the prose of the three authors.
    
A rigorous method of testing this would be to implement some kind of cross-validation into the model training. However,
since I've run this script roughly twenty times and count-vectorized models have beaten TF-IDF models each time,
by roughly the same margin, for all the models (not just naive Bayes but the others too) strongly suggests the explanation
is (2)
"""


