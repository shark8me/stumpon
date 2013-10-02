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
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB,BernoulliNB


def gettfv():
    return TfidfVectorizer(strip_accents='unicode',
                               use_idf=1,smooth_idf=1,
                               min_df=1,ngram_range=(1,1),
                             analyzer='word',token_pattern=r'\w{1,}',
                                max_features=None)
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
    '''20 Fold CV Score:  0.871311563683'''

    pipeline = Pipeline([
        ('vect', CountVectorizer(max_df=0.5,max_features=None,
                                 ngram_range=(1,1))),
        ('tfidf', TfidfTransformer(use_idf=True,norm='l1')),       
        ('clf', lm.SGDClassifier(alpha=1e-05,penalty='l2',n_iter=50)),
    ])

    parameters = {
        'vect__max_df': (0.5, 0.75, 1.0), #0.5
        'vect__max_features': (None, 5000, 10000), #0.5
        'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams 0.5
        'tfidf__use_idf': (True, False),#True
        'tfidf__norm': ('l1', 'l2'), #'l1'
        'clf__alpha': (0.00001, 0.000001), #best 1e-05
        'clf__penalty': ('l2', 'elasticnet'), #'l2'
        'clf__n_iter': (10, 50, 80), #best 50
    }
    #pipe=build_tfidf_logreg();
    return pipeline,parameters

def build_m3():
    '''
    Result:
    done in 1230.728s
()
Best score: 0.809
Best parameters set:
    logreg__C: 1
    logreg__penalty: 'l2'
    tfv__min_df: 1
    tfv__ngram_range: (1, 1)
    
    20 Fold CV Score:  0.876657636033
    '''
    pipeline = Pipeline([
        ('tfv', TfidfVectorizer(strip_accents='unicode',
                               use_idf=1,smooth_idf=1,
                               min_df=1,ngram_range=(1,1),
                             analyzer='word',token_pattern=r'\w{1,}',
                                max_features=None)),        
        ('logreg',lm.LogisticRegression(penalty='l2', dual=False, tol=0.001,
                                 C=1, fit_intercept=True, intercept_scaling=1.0,
                                 class_weight=None, random_state=None))
    ])

    parameters = {
        'tfv__min_df':(1,2,3),
        'tfv__ngram_range': ((1, 1), (1, 2)),
        'logreg__penalty':('l1','l2'), #l2 is the best
        'logreg__C':(0.1,0.5,1)
    }
    return pipeline,parameters

def build_m4():
    '''
    all these results are without scaling in the pipeline.
    done in 1196.240s
()
Best score: 0.796
Best parameters set:
    svm__C: 100
    svm__gamma: 0.001
    svm__kernel: 'rbf'
    
    linear kernel:
    
    done in 562.902s
()
Best score: 0.809
Best parameters set:
    svm__C: 1
    svm__kernel: 'linear'
    
    results with scaling:
    
    done in 262.323s
()
Best score: 0.745
Best parameters set:
    svm__C: 1
    svm__kernel: 'linear'
    probably because 
    '''
    parameters = {#'svm__kernel': ['rbf'], 
                  # 'svm__gamma': [1e-3, 1e-4],
                   #'svm__C': [1, 10, 100, 1000]}
                    'svm__kernel': ['linear'], 
#                    'svm__C': [1, 10, 100, 1000]
                    'svm__C': [1]    }
                    
    pipeline = Pipeline([
        ('tfv', TfidfVectorizer(strip_accents='unicode',
                               use_idf=1,smooth_idf=1,
                               min_df=1,ngram_range=(1,1),
                             analyzer='word',token_pattern=r'\w{1,}',
                                max_features=None)),
        ('scaler',StandardScaler(with_mean=False)),        
        ('svm',SVC(C=1,kernel='linear'))
    ])
    return pipeline,parameters

def build_nb():
    '''
    done in 57.059s
()
Best score: 0.758
Best parameters set:
    nb__alpha: 0.1
    
    with scoring='roc_auc'
    Best score: 0.844
    
    20 Fold CV Score:  0.848171497265
    '''
    parameters = {'nb__alpha':[0.1]}
                    
    pipeline = Pipeline([
        ('tfv', gettfv()),       
        ('nb',BernoulliNB(alpha=0.1))
    ])
    return pipeline,parameters
    
