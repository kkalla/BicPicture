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
train_sub['DE_DT'] = pd.to_datetime(train_sub.DE_DT)
train_sub = train_sub.drop_duplicates()
train_sub.reset_index(inplace=True)
train_sub = train_sub.iloc[:,1:]

#Split source and target by quarter
grouped = train_sub.groupby('ID')
ids = train_sub.ID.unique()
src_dataset = pd.DataFrame()
tgt_dataset = pd.DataFrame()
i = 0
for id in ids:
    if i % 1000==0:
        print('# of steps: %d, id: %s'%(i+1,id))
    indexes = grouped.groups.get(id).values
    subset = train_sub.iloc[indexes,:]
    q1_src = subset[subset.DE_DT < pd.to_datetime('2015-03-01')]
    
    i += 1
    
def split_src_tgt(subset,quarters):
    q1_src
#Save to csv
print('saving train_processed')
train_sub.to_csv('../data/train_processed.csv',index=False)
print('saving src_dataset.csv')
src_dataset.to_csv('../data/src_dataset.csv',index=False)
print('saving trg_dataset')
tgt_dataset.to_csv('../data/tgt_dataset.csv',index=False)

