# -*- coding: utf-8 -*-
"""
Preprocessing

Created on Mon Jan 22 13:23:58 2018

@author: user
"""

import pandas as pd
import numpy as np

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
def get_quarters(subset,quarters=['2015-03-01','2015-06-01','2015-09-01']):
    quarters = pd.to_datetime(quarters)
    q1 = subset[subset.DE_DT < quarters[0]]
    q2 = subset[((subset.DE_DT < quarters[1]) & (subset.DE_DT>=quarters[0]))]
    q3 = subset[((subset.DE_DT <quarters[2]) & (subset.DE_DT >= quarters[1]))]
    q4 = subset[(subset.DE_DT >= quarters[2])]
    return q1,q2,q3,q4

def split_src_tgt(subset):
    max_month = subset.DE_DT.max().month
    max_date = pd.to_datetime('2015-'+str(max_month)+'-01')
    src = subset[subset.DE_DT < max_date]
    tgt = subset[subset.DE_DT >= max_date]
    return src,tgt

src_dataset = pd.DataFrame()
tgt_dataset = pd.DataFrame()

q1,q2,q3,q4 = get_quarters(train_sub)
i = 1
for q in [q1,q2,q3,q4]:
    src,tgt = split_src_tgt(q)
    src_group = np.repeat(i,len(src))
    tgt_group = np.repeat(i,len(tgt))
    src.loc[:,'GROUP'] = src_group
    tgt.loc[:,'GROUP'] = tgt_group
    src_dataset = src_dataset.append(src)
    tgt_dataset = tgt_dataset.append(tgt)
    i += 1

# sort src, tgt_dataset
src_dataset.sort_values(by=['ID','DE_DT','DE_HR'],inplace=True)
tgt_dataset.sort_values(by=['ID','DE_DT','DE_HR'],inplace=True)
#Save to csv
print('saving train_processed')
train_sub.to_csv('../data/train_processed.csv',index=False)
print('saving src_dataset.csv')
src_dataset.to_csv('../data/src_dataset.csv',index=False)
print('saving trg_dataset')
tgt_dataset.to_csv('../data/tgt_dataset.csv',index=False)

