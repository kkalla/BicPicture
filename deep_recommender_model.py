# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 02:45:46 2018

@author: kkalla
"""

import tensorflow as tf

from model import RNN_Model


class Recommender_Model():
    def __init__(self,
                 args,
                 mode,
                 iterator,
                 vocab_table,
                 depth=1,
                 dropout = 0.4):
        self.args = args
        self.mode = mode
        self.iterator = iterator
        self.vocab_table = vocab_table
        self.depth = depth
        self.dropout = dropout
        self.vocab_size = vocab_table.size
        
        dense = self.get_dense(self.input_tensor,self.depth,self.dropout)
        self.logits= tf.layers.dense(inputs=dense,units=self.vocab_size)
        
        self.probs = tf.nn.softmax(self.logits)
        
        
    def get_dense(self,input_tensor,depth,dropout,mode):
        if depth ==1:
            dense = tf.layers.dense(inputs=input_tensor,units=256,activation=tf.nn.relu)
        elif depth==2:
            dense = tf.layers.dense(inputs=input_tensor,units=512,activation=tf.nn.relu)
            dropout = tf.layers.dropout(inputs=dense,rate=dropout,
                                        training=mode==tf.estimator.ModeKeys.TRAIN)
            dense = tf.layers.dense(inputs=dropout,units = 256,activation=tf.nn.relu)
            dropout2 = tf.layers.dropout(inputs=dense,rate=dropout,
                                         training=mode==tf.estimator.ModeKeys.TRAIN)
            dense = dropout2
        elif depth==3:
            dense1 = tf.layers.dense(inputs=input_tensor,units = 1024,activation=tf.nn.relu)
            dropout1 = tf.layers.dropout(inputs=dense1,rate=dropout,
                                        training=mode==tf.estimator.ModeKeys.TRAIN)
            dense2 = tf.layers.dense(inputs=dropout1,units = 512, activation=tf.nn.relu)
            dropout2 = tf.layers.dropout(inputs=dense2,rate=dropout,
                                         training=mode==tf.estimator.ModeKeys.TRAIN)
            dense3 = tf.layers.dense(inputs=dropout2,units=256,activation=tf.nn.relu)
            dropout3 = tf.layers.dropout(inputs=dense3,rate=dropout,
                                         training=mode==tf.estimator.ModeKeys.TRAIN)
            dense = dropout3
        
        return dense
            
            