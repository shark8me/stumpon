'''
Created on 04-Oct-2013

@author: kiran

Meta-classifier.
combines the output of multiple classifiers to give one output.
'''
from sklearn.feature_extraction.text import TfidfVectorizer
from stumbleupon.models import tfidf_logreg,runmodel
import sklearn.linear_model as lm
import pandas as p
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn import metrics,preprocessing,cross_validation

'''
with metaclassifier:
20 Fold CV Score:  0.995858316259: on kaggle leaderboard ~74%, 
probably overfitting the data.

with top 1000 word features: 20 Fold CV Score:  0.997274291149 
on kaggle leaderboard: ~74%
roc score for each (knn, nb, logreg,sgd)
[0.79818693783046513, 0.80984469547359916, 0.8677072588656668, 0.98022321942255941]
''' 
def get_classifiers(xtrain,xtest,y,get_prob=False):
    knn=KNeighborsClassifier(n_neighbors=25,n_jobs=-1)
    nb=BernoulliNB(alpha=0.1)
    svc=SVC(C=1,kernel='linear')
    logreg=lm.LogisticRegression(penalty='l2', dual=False, tol=0.001,
                                 C=1, fit_intercept=True, intercept_scaling=1.0,
                                 class_weight=None, random_state=None)
    sgd=lm.SGDClassifier(alpha=1e-05,penalty='l2',n_iter=50,loss='log',
                         n_jobs=-1
                         )
    cls=[knn,nb,svc,logreg,sgd]
    #cls=[sgd]
    trainpred=[]
    testpred=[]
    for i in cls:
        print " fitting classifier "
        #scores = cross_validation.cross_val_score(i, xtrain, y,
        #                                          scoring='roc_auc')
        i.fit(xtrain,y)
        #print " predicting with classifier ",scores
        if(get_prob):
            trainpred.append(i.predict_proba(xtrain)[:,1])
            testpred.append(i.predict_proba(xtest)[:,1])
        else:
            pr=i.predict(xtrain)
            trainpred.append(pr)            
            testpred.append(i.predict(xtest))
        print " roc score ",metrics.classification_report(y,pr)
    return trainpred,testpred

def get_lowdim_vec(X_all):
    tfv=TfidfVectorizer(strip_accents='unicode',
                               use_idf=1,smooth_idf=1,
                               min_df=1,ngram_range=(1,1),
                             analyzer='word',token_pattern=r'\w{1,}',
                                max_features=1000).fit(X)
    return tfv

def getavgscore(arr):
    ret=np.average(arr,axis=1)
    r2=ret>0.5
    r3=r2.astype(int)
    return r3

''' kaggle score of 79.811 , with roc_auc_score being 88.22
this score is the average of probabilities of all 4 classifiers,
since SVM does not output a probability.
'''
def get_predfile():
    tfv=tfidf_logreg.gettfv()
    X,y,X_all,X_test=runmodel.loaddata()
    tfv.fit(X_all)
    trainwordfeats=tfv.transform(X)
    testwordfeats=tfv.transform(X_test)
    tp,ttp=get_classifiers(trainwordfeats,testwordfeats,y)
    trainarr=np.vstack((tp[0],tp[1],tp[2],tp[3])).transpose()
    testarr=np.vstack((ttp[0],ttp[1],ttp[2],ttp[3])).transpose()
    r3=getavgscore(trainarr)
    print " roc score ",metrics.roc_auc_score(y,r3)
    sfn=lambda r:(r>0.5).astype(int)
    print " roc score i ",[metrics.roc_auc_score(y,sfn(i)) for i in trainarr.T]
    return getavgscore(testarr)

def writefile(pred):
    testfile = p.read_csv(runmodel.datadir+'test.tsv', sep="\t", na_values=['?'], index_col=1)
    pred_df = p.DataFrame(pred, index=testfile.index, columns=['label'])
    pred_df.to_csv('benchmark.csv')

''' local auc score: 0.886325480881  
on kaggle: 0.80126    '''
def get_majority_vote():
    tfv=tfidf_logreg.gettfv()
    X,y,X_all,X_test=runmodel.loaddata()
    tfv.fit(X_all)
    trainwordfeats=tfv.transform(X)
    testwordfeats=tfv.transform(X_test)
    tp,ttp=get_classifiers(trainwordfeats,testwordfeats,y)
    trainarr=np.vstack((tp[0],tp[1],tp[2],tp[3],tp[4])).transpose()
    testarr=np.vstack((ttp[0],ttp[1],ttp[2],ttp[3],ttp[4])).transpose()
    print " roc score ",metrics.roc_auc_score(y,(np.sum(trainarr,axis=1)>3).astype(int))
    return (np.sum(testarr,axis=1)>3).astype(int)

x=get_majority_vote()
writefile(x)