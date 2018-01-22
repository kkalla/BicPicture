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
train_sub.reset_index(inplace=True)

#Split train and target
# train .7 test .3
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
    length_subset = subset.shape[0]
    src_size = int(length_subset*.7)
    tgt_size = length_subset - src_size
    src_dataset = src_dataset.append(subset.iloc[:src_size,:])
    tgt_dataset = tgt_dataset.append(subset.iloc[-tgt_size:,:])
    i += 1
#Save to csv
print('saving train_processed')
train_sub.to_csv('../data/train_processed.csv',index=False)
print('saving src_dataset.csv')
src_dataset.to_csv('../data/src_dataset.csv',index=False)
print('saving trg_dataset')
tgt_dataset.to_csv('../data/tgt_dataset.csv',index=False)

