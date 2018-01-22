# -*- coding: utf-8 -*-
"""
Preprocessing

Created on Mon Jan 22 13:23:58 2018

@author: user
"""

import pandas as pd

train = pd.read_csv('../data/train_seq.csv',dtype={'ID':str,'PD_S_C':str})
#sorting
train = train.sort_values(by=['ID','DE_DT','DE_HR'])
train.head()

#subsetting
train_sub = train[['ID','PD_S_C','DE_DT','DE_HR']].copy()
train_sub = train_sub.drop_duplicates()

#Save to csv
train_sub.to_csv('../data/train_processed.csv',index=False)