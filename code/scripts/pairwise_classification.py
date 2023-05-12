import json
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
from scipy import stats

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import xgboost
import warnings
import contextlib
import pdb
import shap
import wave
warnings.filterwarnings("ignore")

data_path = '/home/data/shiba/EJShibaVoice'

possible_enen_pair1 = []
possible_jaja_pair1 = []
possible_enja_pair1 = []
with open(data_path + '/info/immortal_pair_enen.txt','r') as fp:
    data = fp.readlines()
for s_data in data:
    possible_enen_pair1.append(s_data[:-1].split(' '))

with open(data_path + '/info/immortal_pair_jaja.txt','r') as fp:
    data = fp.readlines()
for s_data in data:
    possible_jaja_pair1.append(s_data[:-1].split(' '))

with open(data_path + '/info/immortal_pair_enja.txt','r') as fp:
    data = fp.readlines()
for s_data in data:
    possible_enja_pair1.append(s_data[:-1].split(' '))
    
    
# prepare the dataset
# 0 - enen 2300
# 1 - jaja 2300
# 2 - enja 2300
# 3 - jaen 2300
# data_path = '/home/jieyi/Ani/data/EJShibaVoice/audio_clip_words_features/gemaps/'
data_path = '/home/data/shiba/EJShibaVoice/audio_clip_words_features/filterbank/'
X = []
Y = []
for i in tqdm(range(2300)):
    pair = possible_enen_pair1[i]
    name1 = pair[0][:-4]
    name2 = pair[1][:-4]
#     x1 = pd.read_csv(data_path + name1 + '.csv').iloc[:,1:]
#     x1 = x1.sum()/x1.shape[0]

#     x2 = pd.read_csv(data_path + name2 + '.csv').iloc[:,1:]
#     x2 = x2.sum()/x2.shape[0]

#     x = np.concatenate([x1, x2])
    x = np.concatenate([pd.read_csv(data_path + name1 + '.csv').iloc[0][3:].values,pd.read_csv(data_path + name2 + '.csv').iloc[0][3:].values],axis=0)
    X.append(x)
    Y.append(0)
for i in tqdm(range(2300)):
    pair = possible_jaja_pair1[i]
    name1 = pair[0][:-4]
    name2 = pair[1][:-4]
#     x1 = pd.read_csv(data_path + name1 + '.csv').iloc[:,1:]
#     x1 = x1.sum()/x1.shape[0]

#     x2 = pd.read_csv(data_path + name2 + '.csv').iloc[:,1:]
#     x2 = x2.sum()/x2.shape[0]

#     x = np.concatenate([x1, x2])
    x = np.concatenate([pd.read_csv(data_path + name1 + '.csv').iloc[0][3:].values,pd.read_csv(data_path + name2 + '.csv').iloc[0][3:].values],axis=0)
    X.append(x)
    Y.append(1)
for i in tqdm(range(2300)):
    pair = possible_enja_pair1[i]
    name1 = pair[0][:-4]
    name2 = pair[1][:-4]
#     x1 = pd.read_csv(data_path + name1 + '.csv').iloc[:,1:]
#     x1 = x1.sum()/x1.shape[0]

#     x2 = pd.read_csv(data_path + name2 + '.csv').iloc[:,1:]
#     x2 = x2.sum()/x2.shape[0]

#     x = np.concatenate([x1, x2])
    x = np.concatenate([pd.read_csv(data_path + name1 + '.csv').iloc[0][3:].values,pd.read_csv(data_path + name2 + '.csv').iloc[0][3:].values],axis=0)
    X.append(x)
    Y.append(2)
for i in tqdm(range(2300)):
    pair = possible_enja_pair1[i + 2300]
    name1 = pair[1][:-4]
    name2 = pair[0][:-4]
#     x1 = pd.read_csv(data_path + name1 + '.csv').iloc[:,1:]
#     x1 = x1.sum()/x1.shape[0]

#     x2 = pd.read_csv(data_path + name2 + '.csv').iloc[:,1:]
#     x2 = x2.sum()/x2.shape[0]

#     x = np.concatenate([x1, x2])
    x = np.concatenate([pd.read_csv(data_path + name1 + '.csv').iloc[0][3:].values,pd.read_csv(data_path + name2 + '.csv').iloc[0][3:].values],axis=0)
    X.append(x)
    Y.append(3)
   
model = xgboost.XGBClassifier(max_depth=10,learning_rate=0.05,n_estimators=150)
print(cross_val_score(model, X, Y, cv=5).sum()/5)

model = KNeighborsClassifier()
print(cross_val_score(model, X,Y,cv=5).sum()/5)

model = LogisticRegression()
print(cross_val_score(model, X,Y,cv=5).sum()/5)

model = RandomForestClassifier()
print(cross_val_score(model, X,Y, cv=5).sum()/5)
  
