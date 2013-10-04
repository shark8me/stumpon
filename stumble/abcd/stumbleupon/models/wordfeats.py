'''
Created on 02-Oct-2013

@author: kiran
'''
'''
Created on 02-Oct-2013

@author: kiran
'''

from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
import pandas as p
import json
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest,SelectPercentile,chi2

class wordfeats():
    
    def __init__(self):
        tfv=TfidfVectorizer(strip_accents='unicode',
                                       use_idf=1,smooth_idf=1,
                                       min_df=1,ngram_range=(1,1),
                                       analyzer='word',token_pattern=r'\w{1,}',
                                       max_features=None)
        self.datadir='/home/kiran/kaggle/stumbleupon/data/'
        ptab=p.read_table(self.datadir+'train.tsv')
        self.traindata = list(np.array(ptab)[:,2])
        #self.traindata = np.array(ptab['boilerplate'])
        testdata = list(np.array(p.read_table(self.datadir+'test.tsv'))[:,2])
        #y = np.array(p.read_table(self.datadir+'train.tsv'))[:,-1]
        self.y=np.array(ptab['label'])
        X_all = self.traindata + testdata
        lentrain = len(self.traindata)
        self.X = X_all[:lentrain]
        print "fitting pipeline wordfeats "
        self.tfv=tfv.fit(X_all)
        

    
    def transform(self,X):        
        ret=self.tfv.transform(self.traindata)
        self.sb=SelectPercentile(score_func=chi2,percentile=90).fit(ret,self.y)
        ret1=self.sb.transform(ret)
        return ret1
   

    def fit(self,X, y=None):
        return self

    '''     
    def fit_transform(self,X, y=None):
        return super(TfidfVectorizer,self).fit_transform(X)
      
      '''
          






