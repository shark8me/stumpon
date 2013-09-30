'''
Created on 30-Sep-2013

@author: kiran
'''

import numpy as np
from sklearn import metrics,preprocessing,cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer,CountVectorizer
from sklearn.pipeline import Pipeline 
import sklearn.linear_model as lm
import pandas as p


def build_tfidf_logreg():

    tfv = TfidfVectorizer(min_df=3,  max_features=None, strip_accents='unicode',
                        analyzer='word',token_pattern=r'\w{1,}',
                        ngram_range=(1, 2), use_idf=1,smooth_idf=1,
                        sublinear_tf=1)
    lreg = lm.LogisticRegression(penalty='l1', dual=False, tol=0.0001,
                                 C=1, fit_intercept=True, intercept_scaling=1.0,
                                 class_weight=None, random_state=None)
    pipeline = Pipeline([('tfv', tfv), ('logreg', lreg)])
    return pipeline

def build_m2():

    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', lm.SGDClassifier()),
    ])

    parameters = {
        'vect__max_df': (0.5, 0.75, 1.0),
        #'vect__max_features': (None, 5000, 10000, 50000),
        'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
        #'tfidf__use_idf': (True, False),
        #'tfidf__norm': ('l1', 'l2'),
        'clf__alpha': (0.00001, 0.000001),
        'clf__penalty': ('l2', 'elasticnet'),
        #'clf__n_iter': (10, 50, 80),
    }

    return pipeline



