'''
Created on 30-Sep-2013

@author: kiran
'''


import numpy as np
from sklearn import metrics,preprocessing,cross_validation
import pandas as p
import stumbleupon.models.tfidf_logreg as m1

datadir='/home/kiran/kaggle/stumbleupon/data/'

def loaddata():
    print "loading data.."
    ptab=p.read_table(datadir+'train.tsv')
    traindata = list(np.array(ptab)[:,2])
    testdata = list(np.array(p.read_table(datadir+'test.tsv'))[:,2])
    #y = np.array(p.read_table(datadir+'train.tsv'))[:,-1]
    y=np.array(ptab['label'])

    X_all = traindata + testdata
    
    lentrain = len(traindata)
       
    X = X_all[:lentrain]
    X_test = X_all[lentrain:]
    
    return X,y,X_all,X_test

def main():

    X,y,X_all,X_test=loaddata()
    pipeline,_=m1.build_nb() 
    print "20 Fold CV Score: ", np.mean(cross_validation.cross_val_score(pipeline, X, y, cv=20, scoring='roc_auc'))


#if __name__ == '__main__':
#    main()