"""
Generating samples

@author: kkalla
"""

from __future__ import print_function
from __future__ import absolute_import

import os
import argparse

from six.moves import cPickle
import tensorflow as tf

from model import RNN_Model

parser = argparse.ArgumentParser()

parser.add_argument('--save_dir',type=str,default='save')
parser.add_argument('-n','--num_samples',type=int,default=10)
parser.add_argument('--prime',type=str,default=['0906','0807','0697','0265','0247'])
parser.add_argument('--sample',type=int,default=1)

def sample(args):
    with open(os.path.join(args.save_dir,'config.pkl'),'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(args.save_dir,'chars_vocab.pkl'),'rb') as f:
        chars, vocab = cPickle.load(f)

    model = RNN_Model(saved_args,mode='INFER')
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess = sess, save_path = ckpt.model_checkpoint_path)
            print(model.sample(sess,chars,vocab,args.num_samples,args.prime,args.sample))

def main():
    args = parser.parse_args()
    sample(args)

if __name__ =='__main__':
    main()
