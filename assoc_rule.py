# -*- coding: utf-8 -*-
"""
Association rule mining

Created on Fri Dec 29 16:56:53 2017

@author: user
"""

import pandas as pd

shop_data = pd.read_csv('data/02_shopping_tran.txt')
shop_data.columns = shop_data.columns.str.lower()

shop_data.info()
shop_data['rct_no'] = shop_data['rct_no'].astype('str')
shop_data['pd_s_c'] = shop_data['pd_s_c'].astype('str')

basket = shop_data.groupby(['rct_no','pd_s_c'])['buy_ct']\
.sum().unstack().reset_index().fillna(0).set_index('rct_no')