# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 20:00:19 2018

@author: kkalla
"""
import os


import pandas as pd
import numpy as np

from six.moves import cPickle

class Data_loader():
    def __init__(self,data_dir,batch_size,seq_length,encoding='utf-8'):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.encoding = encoding
        
        origin_file = os.path.join(data_dir,'02_shopping_tran.txt')
        input_file = os.path.join(data_dir,"shopping_sorted.csv")
        vocab_file = os.path.join(data_dir,'vocab.pkl')
        tensor_file = os.path.join(data_dir,'data.npy')
        
        if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            print("Reading origin file and preprocessing")
            self.preprocess(origin_file,input_file,vocab_file,tensor_file)
        else:
            print("Loading preprocessed files")
            self.load_preprocessed(vocab_file,tensor_file)
        
        self.create_batches()
        self.reset_batch_pointer()
        
    def preprocess(self,origin_file,input_file,vocab_file,tensor_file):
        if not os.path.exists(input_file):
            #sorting data
            origin_data = pd.read_csv(origin_file,
                                      dtype={'ID':str,'PD_S_C':str,
                                             'RCT_NO':str,'DE_DT':str})
            origin_data['DE_DT'] = pd.to_datetime(origin_data['DE_DT'],format='%Y%m%d')
            origin_data = origin_data.sort_values(by=['ID','DE_DT','DE_HR'])
            # delete Ids of people who purchased items less than 15 times
            usedcount = origin_data.groupby(by='ID').count()
            used = usedcount[usedcount.RCT_NO<=15].index
            unused = origin_data[origin_data.ID.isin(used.values)].copy()
            origin_data = origin_data[~origin_data.ID.isin(used.values)]
            # save sorted data
            origin_data.to_csv(os.path.join(self.data_dir,'shopping_sorted.csv'),index=False)
            unused.to_csv(os.path.join(self.data_dir,'rnn_unused.csv'),index=False)
        # loading sorted data
        input_data = pd.read_csv(input_file,usecols=['ID','PD_S_C'],
                                 dtype={'ID':str,'PD_S_C':str})
        counts = input_data.PD_S_C.value_counts()
        self.chars = counts.index
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars,range(len(self.chars))))
        
        with open(vocab_file,'wb') as f:
            cPickle.dump(self.chars,f)
        self.tensor = np.array(list(map(self.vocab.get,input_data.PD_S_C.values)))
        np.save(tensor_file,self.tensor)
    
    def load_preprocessed(self,vocab_file,tensor_file):
        with open(vocab_file,'rb') as f:
            self.chars = cPickle.load(f)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars,range(len(self.chars))))
        self.tensor = np.load(tensor_file)
    
    def create_batches(self):
        self.num_batches = int(self.tensor.size / (self.batch_size * self.seq_length))
        self.tensor = self.tensor[:self.num_batches*self.batch_size*self.seq_length]
        xdata = self.tensor
        ydata = np.copy(self.tensor)
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]
        self.x_batches = np.split(xdata.reshape(self.batch_size,-1),self.num_batches,1)
        self.y_batches = np.split(ydata.reshape(self.batch_size,-1),self.num_batches,1)
    
    def next_batch(self):
        x, y = self.x_batches[self.pointer],self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0
