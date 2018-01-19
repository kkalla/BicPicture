# -*- coding: utf-8 -*-
"""
Splitting dataset into train and test

Created on Tue Jan 16 01:02:49 2018

@author: kkalla
"""
from __future__ import print_function

import pandas as pd

shop_data = pd.read_csv('data/02_shopping_tran.txt',dtype={'ID':str,'PD_S_C':str,
                                             'RCT_NO':str,'DE_DT':str})
non_shop_data = pd.read_csv('data/03_non_shopping_tran.txt')
shop_data['DE_DT'] = pd.to_datetime(shop_data.DE_DT,format='%Y-%m-%d')
#non_shop_data['CRYM'] = pd.to_datetime(non_shop_data.CRYM,format='%Y%m')
#sort by date and hr
shop_data = shop_data.sort_values(by=['ID','DE_DT','DE_HR'])
#non_shop_data = non_shop_data.sort_values(by=['ID','CRYM'])
shop_data.head()

test = pd.DataFrame()
#for i in shop_data.ID.unique():
#    test.append(shop_data[shop_data.ID == i][-5:])

#test.to_csv('data/test_seq.csv',index=False)
#print('test_seq done!!')
# Store last 5 items to testset
#test = pd.DataFrame()
#for i in shop_data.ID.unique():
#    test.append(shop_data[shop_data.ID == i][-5:])
#
#test.to_csv('data/test_seq.csv',index=False)
#print('test_seq done!!')

#train = pd.DataFrame()
#for i in shop_data.ID.unique():
#    train.append(shop_data[shop_data.ID == i][:-5])
#
#train.to_csv('data/train_seq.csv',index=False)
#print('train_seq done!!')

grouped = shop_data.groupby(by='ID')
groups = grouped.groups
keys = shop_data.ID.unique()
train = pd.DataFrame()
for key in keys:
    aa = groups.get(key)[:-5]
    aa = shop_data.iloc[aa,:]
    train = train.append(aa)
    print(key)

print('train_seq done!!')
train.info()
train.to_csv('data/train_seq.csv',index=False)
