# -*- coding: utf-8 -*-
"""
Splitting dataset into train and test

Created on Tue Jan 16 01:02:49 2018

@author: kkalla
"""
from __future__ import print_function

import time

import pandas as pd

shop_data = pd.read_csv('../data/02_shopping_tran.txt',dtype={'ID':str,'PD_S_C':str,'RCT_NO':str,'DE_DT':str})
shop_data['DE_DT'] = pd.to_datetime(shop_data.DE_DT,format='%Y-%m-%d')

#non_shop_data = pd.read_csv('data/03_non_shopping_tran.txt')
#non_shop_data['CRYM'] = pd.to_datetime(non_shop_data.CRYM,format='%Y%m')

#sort by date and hr
shop_data = shop_data.sort_values(by=['ID','DE_DT','DE_HR'])
#non_shop_data = non_shop_data.sort_values(by=['ID','CRYM'])

# delete users who purchased items less than 10times
grouped = shop_data.groupby(by='ID')
count = grouped.count()
unused_index = count[count.RCT_NO < 10].index
unused = shop_data[shop_data.ID.isin(unused_index.values)].copy()
used = shop_data[~shop_data.ID.isin(unused_index.values)]

print('Saving unused users')
unused.to_csv('../data/rnn_unused_users.csv',index=False)

#reindexing
used = used.reindex

# split train and test
grouped = used.groupby(by='ID')
groups = grouped.groups
keys = used.ID.unique()

test = pd.DataFrame()
i = 0
start = time.time()
for key in keys:
    aa = groups.get(key)[-5:]
    aa = used.iloc[aa,:]
    test = test.append(aa)
    if i%1000==0:
        end = time.time()
        print('step %d'%(i))
        print('in time %f'%(end-start))
        print(key)
    i += 1
print('test_seq done!!')
test.to_csv('../data/test_seq.csv',index=False)


train = pd.DataFrame()
i = 0
start = time.time()
for key in keys:
    aa = groups.get(key)[:-5]
    aa = used.iloc[aa,:]
    train = train.append(aa)
    if i%1000==0:
        end = time.time()
        print('step %d'%(i))
        print('in time %f'%(end-start))
        print(key)
    i += 1
print('train_seq done!!')
print(train.info())
train.to_csv('../data/train_seq.csv',index=False)
