'''
Created on 02-Oct-2013

@author: kiran
'''

from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder,Imputer,MinMaxScaler
import pandas as p
import json
import re
import numpy as np
from scipy import sparse

class NonwordFeats(BaseEstimator):
    
    def __init__(self):
        self.datadir='/home/kiran/kaggle/stumbleupon/data/'
        self.ptab=p.read_table(self.datadir+'train.tsv')
        self.countries=self.loadcountries()
        self.cities=self.loadcities()
        #self.data=self._fit()
        
    def loadcountries(self):
        cnt=[]
        with open(self.datadir+'countries.txt') as fp:
            for line in fp:
                cnt.append(line)
        return set(cnt)

    def loadcities(self):
        return set(p.read_csv(self.datadir+'cities.txt')['City'].values)

    def hascountries(self,col,countries,cities):
        totcit=[]
        totcnt=[]
        for i in col:
            cit=0
            cnt=0
            j=json.loads(i)
            x0=j['body']
            if x0 is not None:
                z=x0.encode('ascii','ignore')
            for x in re.split('\W{1,}', z):
                if x in cities:
                    cit=cit+1
                if x in countries:
                    cnt=cnt+1
            totcit.append(cit)
            totcnt.append(cnt)    
        return np.array(totcnt),np.array(totcit)
        
    def _fit(self,tab):
        #tab=self.ptab.copy(deep=False)
        #tab.replace("?",0,inplace=True)
        print " tab shape ",type(tab)
        xall=np.array(tab)
        cit,cnt=self.hascountries(xall[:,2],self.countries,self.cities)
        #tab['cit']=cit
        #tab['cnt']=cnt
        lex=LabelEncoder()
        cat=lex.fit_transform(xall[:,3])
        tab['cat']=cat
        del tab['label']
        del tab['url']
        del tab['urlid']
        del tab['boilerplate']
        del tab['alchemy_category']
        ret=np.array(tab)
        return ret 
    
    def transform(self,X):
        ret=self._fit(X)
        im=Imputer(missing_values=-1).fit(ret)
        ret1=im.transform(ret)
        ss=MinMaxScaler(copy=False).fit(ret1)
        ret2=ss.transform(ret1)
        print " nonwordfeats transform",sparse.issparse(ret) 
        return ret2
    
    
    def fit(self,X, y=None):
        return self
    ''' 
    def fit_transform(self,X, y=None):
        return self._fit()
    ''' 






