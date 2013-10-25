'''
Created on 06-Oct-2013

@author: kiran
'''

import numpy as np
from sklearn import metrics,preprocessing,cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer,CountVectorizer
from sklearn.pipeline import Pipeline,FeatureUnion 
import sklearn.linear_model as lm
import pandas as p
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB,BernoulliNB
from sklearn.neighbors import KNeighborsClassifier,DistanceMetric
#import sklearn.neighbors.DistanceMetric
from sklearn.ensemble import GradientBoostingClassifier
from stumbleupon import logreg
from nltk import stem
import re
from nltk.stem.wordnet import WordNetLemmatizer

datadir='/home/kiran/kaggle/stumbleupon/data/'
''' use of porter stemmer against lancaster stemmer gives one percentage point more: 
0.878633822231 as against 0 Fold CV Score:  0.878574145464'''
stemmer = stem.PorterStemmer()
#stemmer = stem.LancasterStemmer()
lmztr=WordNetLemmatizer()


def getdefclf():
    rd = lm.LogisticRegression(penalty='l2', dual=True, tol=1e-9,
                           C=1, fit_intercept=True, intercept_scaling=1.0,
                           class_weight=None, random_state=None)
    return rd

''' 
20 Fold CV Score:  0.876809206097'''
def bench():
    tfv = TfidfVectorizer(min_df=3,  max_features=None, strip_accents='unicode',
      analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1)
    return tfv

''' 20 Fold CV Score:  0.875301310254 '''
def onegram():
    tfv = TfidfVectorizer(min_df=3,  max_features=None, strip_accents='unicode',
      analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1, 1), use_idf=1,smooth_idf=1,sublinear_tf=1)
    return tfv

''' 20 Fold CV Score:  0.877015846439 '''
def v1():
    stop_words=['www','html','title','url','body','if']
    tfv = TfidfVectorizer(min_df=3,  max_features=None, 
                          strip_accents='unicode',
      analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1, 2), 
      use_idf=1,smooth_idf=1,sublinear_tf=1,stop_words=stop_words)
    return tfv

def splitdirtystr(inp):
    x=re.compile(r'\d{1,10}')
    xfn=lambda a:inp[a[0]:a[1]]
    return [i.span() for i in x.finditer(inp)]

def strspl2(instr):
    tuplst=splitdirtystr(instr)
#    print "tuplst ",tuplst, "str ",instr
    retlst=[]
    xfn=lambda i:instr[i[0]:i[1]]
    if len(tuplst)==0:
        return instr
    
    if len(tuplst) >1:
        for i in range(1,len(tuplst)):
            j,k=tuplst[i]        
            ls=[l for l in range(tuplst[i-1][1],j)]
#            print " ls ", ls, " ",(tuplst[i-1][1],j)
            if len(ls) > 0:
                retlst.append((ls[0],ls[-1]+1))
            retlst.append((j,k))
        if len(tuplst) >0:
            retlst.insert(0,tuplst[0])     
    else:        
        a=[i for i in range(len(instr)) if i < tuplst[0][0]]
        b=[i for i in range(len(instr)) if i>tuplst[0][1]]
#        print " a ",a," ",b
        if len(a) > 0:
            try:
                retlst.append((a[0],a[1]+1))
            except (IndexError):
                retlst.append((a[0],a[0]+1))
        retlst.append(tuplst[0])
        if len(b) > 0:
            retlst.append((b[0]-1,len(instr)))    
    ret=" ".join([xfn(i) for i in retlst])
#    print " ",ret
    return ret

def preproc(x):
    #print " in preprc",x
    z=x.encode('ascii',errors='ignore')
    z=" ".join([strspl2(i) for i in z.split(" ")])
    z=" ".join([logreg.redcn(i,lem=False) for i in z.split(" ")])
    z=" ".join([stemmer.stem(i) for i in z.split(" ")])
    x1=re.sub(r' [0-9][0-9][0-9][0-9] [0-9][0-9] [0-9][0-9]',
              ' datestr ',z)
#    x1=re.sub('\\\\u[0-9a-f][0-9a-f][0-9a-f][0-9a-f]',"",x)
    x1=re.sub(r'[2][0-9][0-9][0-9]'," twothousand ",x1)
    x1=re.sub(r'[1][9][0-9][0-9]'," nineties ",x1)
    x1=re.sub(r'\d{1,6}[ ]*(lbs?|oz|g|kg|ml|c|d|f|st|th|rd|nd|cm|m|px|mm|am|pm|mb|kb|gb)',' digit msr',x1)
    #x2=re.sub(r'\d',"",x1)
    #careful! changed a line "Personal Style" in test.tsv
    x2=re.sub(r'\d{1,9} '," digit ",x1)
    return x2

''' 20 Fold CV Score:  0.877235936028 
using Porter Stemming: 20 Fold CV Score:  0.878233298325
num feats  230249
Lancaster stemming: 20 Fold CV Score:  0.878004696807
num feats  214099
Lemmatizer: 20 Fold CV Score:  0.877459271411
num feats  236296
using redcn: 20 Fold CV Score:  0.87765106977
num feats  219576
using only words where len>3:
20 Fold CV Score:  0.87675860268
num feats  164859
with redcn, and stemmer, with lemmatization off: 20 Fold CV Score:  0.878574145464
num feats  213159
'''
def v3():
    stop_words=['www','html','title','url','body','if']
    tfv = TfidfVectorizer(min_df=3,  max_features=None, 
                          strip_accents='unicode',
                          preprocessor=preproc,
      analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1, 2), 
      use_idf=1,smooth_idf=1,sublinear_tf=1,stop_words=stop_words)
    return tfv

def runpipeline(tfv):
    traindata = list(np.array(p.read_table(datadir+'train.tsv'))[:,2])
    testdata = list(np.array(p.read_table(datadir+'test.tsv'))[:,2])
    y = np.array(p.read_table(datadir+'train.tsv'))[:,-1]
    X_all = traindata + testdata
    lentrain = len(traindata)
    print "fitting pipeline"
    tfv.fit(X_all)
    print "transforming data"
    X_all = tfv.transform(X_all)    
    X = X_all[:lentrain]
    X_test = X_all[lentrain:]    
    clf=getdefclf()
    print "20 Fold CV Score: ", np.mean(cross_validation.cross_val_score(clf, X, y, cv=20, scoring='roc_auc'))
    print "training on full data"
    clf.fit(X,y)
    pred = clf.predict_proba(X_test)[:,1]
    testfile = p.read_csv(datadir+'test.tsv', sep="\t", na_values=['?'], index_col=1)
    pred_df = p.DataFrame(pred, index=testfile.index, columns=['label'])
    pred_df.to_csv('benchmark.csv')
    print "submission file created.."
    return tfv

tfv=runpipeline(v3())
print "num feats ",len(tfv.get_feature_names())
#tfv.get_feature_names()[19200:19300]