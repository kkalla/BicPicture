# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 21:34:46 2018

@author: user
"""
import pandas as pd

        
data = pd.read_csv('data/02_shopping_tran.txt',
                   dtype={'ID':str,'RCT_NO':str,'PD_S_C':str,'DE_DT':str})
data['DE_DT'] = pd.to_datetime(data['DE_DT'],format='%Y%m%d')
#sort by date and hr
data = data.sort_values(by=['ID','DE_DT','DE_HR'])

data.to_csv('data/shopping_sorted.csv',index=False)