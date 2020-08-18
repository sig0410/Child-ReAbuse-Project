#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import re
import nltk

import pandas as pd
import numpy as np

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
english_stemmer=nltk.stem.SnowballStemmer('english')

from sklearn.feature_selection.univariate_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import random
import itertools

import sys
import os
import argparse
from sklearn.pipeline import Pipeline
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
import six
from abc import ABCMeta
from scipy import sparse
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y, check_array
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.preprocessing import normalize, binarize, LabelBinarizer
from sklearn.svm import LinearSVC

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Lambda
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.preprocessing.text import Tokenizer
from collections import defaultdict
from keras.layers.convolutional import Convolution1D
from keras import backend as K

import matplotlib.pyplot as plt

from matplotlib import rc
rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

# In[6]:


data_file = '/Users/Moon/Desktop/Project/Child(preprocessing).csv'
n = 122676
s = 122676
skip = sorted(random.sample(range(1,n),n-s))


df = pd.read_csv( data_file, delimiter = ",", skiprows = skip, encoding = 'cp949')
#41만개의 데이터 중 3000개의 데이터만 뽑아서 분석을 진행


# In[7]:


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
#이거 꼭 기억하기 개꿀따리임

# In[10]:


df['재신고 이전 사건'].isnull().sum()


# In[11]:


df['재신고 이전 사건']


# In[12]:


a = []
for i in df['재신고 이전 사건'].notnull():
    if i == True:
        a.append(1)
    else:
        a.append(0)


# In[13]:


a = pd.Series(a)
df = pd.concat([df,a], axis = 1)
df.head()


# In[14]:


df.isnull().sum()


# In[15]:


df = df.drop('재신고 이전 사건', axis = 1)


# In[16]:


df.rename(columns = {0 : '재신고 이전 사건 '}, inplace = True)

# In[17]:


for i in df.columns:
    print(i, ':', df[i].unique())

# 성별, 생년월일, 최종 학력, 거주상태, 친권자 유형, 가족 유형, 가구 소득 구분코드, 기초생활수급 유형, 재신고 유형, 신고접수 구분, 피해아동 상태 구분, 아동 동거 여부, 재신고 여부, 통계 거점, 학대 혐의 여부, 재신고 이전 사건


# In[18]:


round(df.isnull().sum() / len(df), 3) * 100

# In[19]:


df.columns

# In[20]:


df1 = df[
    ['성별', '생년월일', '거주상태', '가족 유형', '가구 소득 구분코드', '기초생활수급 유형', '재신고 유형', '신고접수 구분', '피해아동 상태 구분', '아동 동거 여부', '재신고 여부',
     '통계 거점', '학대 혐의 여부', '재신고 이전 사건 ']]
# 결측치가 50%를 넘어가는 column을 삭제


# In[21]:

# In[22]:


# for i in df1:
#     print(i, ' : ', df1[i].unique())


# In[23]:



# In[24]:



# In[25]:


Sex = {'M' : 1, "F" : 1, "Z" : 0} #<- 이거는 다시해야함
sex = lambda x : Sex.get(x,x)

Home = {'자택' : 4, '보증금(전세)+월세' : 3, '전세' : 3, '월세' : 3, '영구임대아파트  또는 영구임대주택' : 1, '기타' : 2, '보호시설' : 1, '무상' : 2}
home = lambda x : Home.get(x,x)
#다문화가족 내 피해아동의 내재화문제 결정요인보고서를 근거로 함

Family = {'재혼가정' : 4, '부자가족(별거)' : 1, '친부모가정' : 5, '부자가족(이혼)' : 4, '친인척보호' : 1, '부자가족(가출)' : 1, '시설보호' : 1, '모자가족(사별)' : 1, '모자가족(별거)' : 1, '모자가족(이혼)' : 4, '부자가족(사별)' : 1, '미혼부·모가정' : 1, '동거(사실혼포함)' : 2, '위탁가정' : 0, '입양가정' : 6, '기타' : 1, '소년소녀가정' : 2,  '모자가족(가출)' : 1}
family = lambda x: Family.get(x, x)
#다문화가족 내 피해아동의 내재화문제 결정요인보고서를 근거로 함

Income = {'50만원미만' : 2, '50만원이상~100만원미만' : 1, '100만원이상-150만원미만' : 4, '150만원이상-200만원미만' : 2, '200만원이상-250만원미만' : 2, '250만원이상-300만원미만' : 1, '300만원이상' : 5}
income = lambda x: Income.get(x, x)
#아동학대_주요통계를 근거로 함 (최종)

Basic = {'비수급권대상' : 0, '수급권대상' : 1}
basic = lambda x: Basic.get(x, x)

Recall_type={'동일센터 사례종결후 재신고':1,'사례진행중 재신고': 2,'타센터 사례종결후 재신고':1,'일반상담 후 재신고':0}
recall_type = lambda x : Recall_type.get(x, x)

Call = {'아동보호전문기관접수' : 0, '경찰접수' : 1}
call = lambda x: Call.get(x, x)

Child = {'해당사항없음' : 0, '아동사망' : 2, '중상해(의식불명 포함)' : 1}
child = lambda x: Child.get(x, x)

Livewith = {'동거' : 1, '비동거' : 0}
livewith = lambda x: Livewith.get(x, x)

Recall={'Y' : 1, 'N' : 0}
recall = lambda x: Recall.get(x, x)

Location = {'서울': 2, '인천' : 1, '전남' : 2, '경남':1, '전북' : 1, '경기' : 2, '강원' : 1, '충남' : 2,  '부산' : 2, '광주' : 1, '충북' : 0, '울산' : 0, '경북' : 1, '대구' : 2, '제주' : 0, '대전' : 0,  '중앙' : 0,'세종' : 0}
location = lambda x: Location.get(x, x)

Judge =  {'Y' : 1, 'N' : 0}
judge = lambda x: Judge.get(x, x)


# In[26]:

X = df1.drop(['재신고 이전 사건 '],axis = 1)
y = df1['재신고 이전 사건 ']

# In[27]:


for i in X.columns: #총 14개의 컬럼으로 구성
    if i == '성별':
        X[i] = X[i].map(sex)
    elif i == '거주상태':
        X[i] = X[i].map(home)
    elif i == '가족 유형':
        X[i] = X[i].map(family)
    elif i == '가구 소득 구분코드':
        X[i] = X[i].map(income)
    elif i == '기초생활수급 유형':
        X[i] = X[i].map(basic)
    elif i == '재신고 유형':
        X[i] = X[i].map(recall_type)
    elif i == '신고접수 구분':
        X[i] = X[i].map(call)
    elif i == '피해아동 상태 구분':
        X[i] = X[i].map(child)
    elif i == '아동 동거 여부':
        X[i] = X[i].map(livewith)
    elif i == '재신고 여부':
        X[i] = X[i].map(recall)
    elif i == '통계 거점':
        X[i] = X[i].map(location)
    elif i == '학대 혐의 여부':
        X[i] = X[i].map(judge)


# In[28]:

# In[30]:

X= X.fillna(0)
print(X.head())
#
#
# # In[31]:

X['생년월일'] = X['생년월일'].astype('str')

# # In[32]:


b = []
for i in X['생년월일']:
    a = i[:4]
    a = float(a)
    b.append(2019 - a)


# In[33]:


c = []
for i in b:
    if i >= 1 and i <= 5:
        c.append(1)
    elif i > 5 and i <= 12:
        c.append(2)
    elif i > 12 and i <= 19:
        c.append(3)
    elif i >19 and i <= 21:
        c.append(1)
    else:
        c.append(0)


# In[34]:

#
# print(c)


# In[35]:
X = X.drop(['생년월일'], axis = 1 )
c = pd.Series(c)
print(c)
X = pd.concat([X,c], axis = 1)
print(X.head())
c = pd.Series(c)
print(c)

# X = X.reset_index()
# c = c.reset_index()

X.rename(columns = {            0 : '생년월일'}, inplace = True)

# X = df1.drop(['index'],axis = 1)
print(X.columns)
print(X.tail())

df = pd.concat([X,y], axis = 1)
df.to_csv('/Users/Moon/Desktop/Project/child2.csv', header = True, index = True, encoding = 'cp949')