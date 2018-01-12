# -*- coding: utf-8 -*-
"""
Training RNN model

Created on Fri Jan 12 12:47:29 2018

@author: kkalla
"""
from __future__ import absolute_import

import argparse
import os
import time

import tensorflow as tf
import numpy as np

from six.moves import cPickle
from model import RNN_Model
from loading_data import Data_loader

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir',type=str,default='data',
                    help = 'directory of input datasets')
parser.add_argument('--save_dir',type=str,default='save',
                    help = 'directory to save model')
parser.add_argument('--rnn_size',type=int,default=64)
parser.add_argument('--num_layers',type=int,default=3,
                    help = 'number of rnn layers')
parser.add_argument('--num_epochs',type=int,default=5,
                    help = 'number of epochs')
parser.add_argument('--batch_size',type=int,default=50,
                    help = 'size of batch')
parser.add_argument('--seq_length',type=int,default=5,
                    help = 'length of sequences')
parser.add_argument('--save_every',type=int,default=1000,
                    help = 'number of steps to save checkpoints')
parser.add_argument('--grad_clip',type=float,default=5.)
parser.add_argument('-lr','--learning_rate',type=float,default=0.002,
                    help = 'learning rate')
parser.add_argument('-dr','--decay_rate', type = float, default=0.97,
                    help = 'decay rate')
parser.add_argument('--vocab_size',type=int,default=1000)

def train(args):
    data_loader = Data_loader(args.data_dir,args.batch_size,args.seq_length)
    args.vocab_size = data_loader.vocab_size
    # Save configurations 
    with open(os.path.join(args.save_dir,'config.pkl'),'wb') as f:
        cPickle.dump(args,f)
    # Save (chars, vocab) pairs
    with open(os.path.join(args.save_dir,'chars_vocab.pkl'),'wb') as f:
        cPickle.dump((data_loader.chars,data_loader.vocab), f)
    
    model = RNN_Model(args)
    
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        tf.summary.FileWriter(logdir=os.path.join(args.save_dir,'graph'),
                              graph=sess.graph).add_graph(sess.graph)
        
        for e in range(args.num_epochs):
            sess.run(tf.assign(model.lr,args.learning_rate*args.decay_rate**e))
            data_loader.reset_batch_pointer()
            state = sess.run(model.initial_state)
            
            for b in range(data_loader.num_batches):
                start = time.time()
                x,y = data_loader.next_batch()
                feed = {model.input_data:x, model.targets: y}
                for i, (c,h) in enumerate(model.initial_state):
                    feed[c] = state[i].c
                    feed[h] = state[i].h
                
                train_loss,state,_ = sess.run([model.cost,model.final_state,model.train_op],
                                              feed_dict = feed)
                end = time.time()
                print('{}/{} (epoch{}), train_loss = {:.3f},time/batch={:.3f}'.format(
                        e*data_loader.num_batches+b,args.num_epochs*data_loader.num_batches,
                        e,train_loss,end-start))
                
                # save for the last result
                if (e==args.num_epochs - 1 and b==data_loader.num_batches - 1):
                    checkpoint_path = os.path.join(args.save_dir,'model.ckpt')
                    saver.save(sess,checkpoint_path,
                               global_step = e*data_loader.num_batches + b)
                    print('model save to {}'.format(checkpoint_path))


def main():
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
    
