'''
Created on 02-Oct-2013

@author: kiran
'''
import pandas as p
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from stumbleupon.models import NonwordFeats,tfidf_logreg,wordfeats
from stumbleupon.models.NonwordFeats import NonwordFeats
from stumbleupon.models.wordfeats import wordfeats
import sklearn.linear_model as lm
from sklearn import metrics,preprocessing,cross_validation
from scipy.sparse import hstack,csr_matrix
 
datadir='/home/kiran/kaggle/stumbleupon/data/'

def loaddata():
    print "loading data.."
    ptab=p.read_table(datadir+'train.tsv')
    ptab.replace({'alchemy_category': '?',
                  "alchemy_category_score":'?',
                  "avglinksize":'?',
                  "commonlinkratio_1":'?',
                  "commonlinkratio_2":'?',
                  "commonlinkratio_3":'?',
                  "commonlinkratio_4":'?',
                  "compression_ratio":'?',
                  "embed_ratio":'?',
                  "framebased":'?',
                  "frameTagRatio":'?',
                  "hasDomainLink":'?',
                  "html_ratio":'?',
                  "image_ratio":'?',
                  "is_news":'?',
                  "lengthyLinkDomain":'?',
                  "linkwordscore":'?',
                  "news_front_page":'?',
                  "non_markup_alphanum_characters":'?',
                  "numberOfLinks":'?', 
                  "numwords_in_url":'?',   
                  "parametrizedLinkRatio":'?', 
                  "spelling_errors_ratio":'?'
                  }, -1,inplace=True)
    y=np.array(ptab['label'])
    return ptab,y

'''
this method horizontally stacks word and non-word features.
Does not give a better score than word features alone.

 scores for feat 1(non-words) and 2(words)
20 Fold CV Score: non words  0.673401583654
20 Fold CV Score: words  0.877581643454
20 Fold CV Score: combined  0.87670419112

The combined score is worse that the best individual score!
'''
def combinedfeatures():
    X,y=loaddata()
    
    nwf=NonwordFeats().fit(X, y)
    tfv=wordfeats().fit(X, y)
    
    feat1=nwf.transform(X)
    feat2=tfv.transform(X)
    c1=csr_matrix(np.array(feat1,dtype=np.float32))
    c3=hstack([feat2,c1])
        
    #combined_features = FeatureUnion([("nwf", nwf), ("tfv", tfv)])
    #X_features = combined_features.fit(X, y).transform(X)
    logreg=lm.LogisticRegression(penalty='l2', dual=False, tol=0.001,
                                 C=1, fit_intercept=True, intercept_scaling=1.0,
                                 class_weight=None, random_state=None)
    #logreg.fit(X, y)
    print "20 Fold CV Score: non words ", np.mean(cross_validation.cross_val_score(logreg, feat1, y, cv=20, scoring='roc_auc'))
    print "20 Fold CV Score: words ", np.mean(cross_validation.cross_val_score(logreg, feat2, y, cv=20, scoring='roc_auc'))
    print "20 Fold CV Score: combined ", np.mean(cross_validation.cross_val_score(logreg, c3, y, cv=20, scoring='roc_auc'))
