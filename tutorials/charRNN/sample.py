# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 23:35:28 2018

@author: kkalla
"""
from __future__ import print_function

import os

from six.moves import cPickle
import numpy as np
import tensorflow as tf

from model import Model

class arguments:
    save_dir = 'save'
    n = 1000
    prime = 'X:1\n'
    sample = 1

def sample(args):
    with open(os.path.join(args.save_dir,'config.pkl'),'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(args.save_dir,'chars_vocab.pkl'),'rb') as f:
        chars, vocab = cPickle.load(f)
    model = Model(saved_args,True)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess = sess,save_path = ckpt.model_checkpoint_path)
            #Excute the model, generating a n char sequence
            print(model.sample(sess,chars,vocab,args.n,args.prime,args.sample))
            
def main():
    args = arguments()
    sample(args)
    
if __name__=='__main__':
    main()
