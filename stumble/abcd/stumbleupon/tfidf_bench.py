'''
Created on 30-Sep-2013

@author: kiran
'''
# -*- coding: utf-8 -*-
import numpy as np
from sklearn import metrics,preprocessing,cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.linear_model as lm
import pandas as p


datadir='/home/kiran/kaggle/stumbleupon/data/'

def main():

  print "loading data.."
  traindata = list(np.array(p.read_table(datadir+'train.tsv'))[:,2])
  testdata = list(np.array(p.read_table(datadir+'test.tsv'))[:,2])
  y = np.array(p.read_table(datadir+'train.tsv'))[:,-1]

  tfv = TfidfVectorizer(min_df=3,  max_features=None, strip_accents='unicode',
                        analyzer='word',token_pattern=r'\w{1,}',
                        ngram_range=(1, 2), use_idf=1,smooth_idf=1,
                        sublinear_tf=1)

  rd = lm.LogisticRegression(penalty='l1', dual=False, tol=0.0001,
                             C=1, fit_intercept=True, intercept_scaling=1.0,
                             class_weight=None, random_state=None)

  X_all = traindata + testdata
  #X_all = traindata
  lentrain = len(traindata)

  print "fitting pipeline"
  tfv.fit(X_all)
  print "transforming data"
  X_all = tfv.transform(X_all)

  X = X_all[:lentrain]
  X_test = X_all[lentrain:]

  print "20 Fold CV Score: ", np.mean(cross_validation.cross_val_score(rd, X, y, cv=20, scoring='roc_auc'))

  print "training on full data"
  rd.fit(X,y)
  pred = rd.predict_proba(X_test)[:,1]
  testfile = p.read_csv(datadir+'test.tsv', sep="\t", na_values=['?'], index_col=1)
  pred_df = p.DataFrame(pred, index=testfile.index, columns=['label'])
  pred_df.to_csv('benchmark.csv')
  print "submission file created.."


if __name__ == '__main__':
    main()