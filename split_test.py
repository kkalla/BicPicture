# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 01:02:49 2018

@author: kkalla
"""
import pandas as pd

shop_data = pd.read_csv('data/shopping_sorted.csv',dtype={'ID':str,'PD_S_C':str,
                                             'RCT_NO':str,'DE_DT':str})
shop_data['DE_DT'] = pd.to_datetime(shop_data.DE_DT,format='%Y-%m-%d')
test = pd.DataFrame()
for i in shop_data.ID.unique():
    test.append(shop_data[shop_data.ID == i][-5:])

test.to_csv('data/test_seq.csv',index=False)

train = pd.DataFrame()
for i in shop_data.ID.unique():
    train.append(shop_data[shop_data.ID == i][:-5])

train.to_csv('data/train_seq.csv',index=False)
    


