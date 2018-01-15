# -*- coding: utf-8 -*-
"""
RNN model for recommender

Created on Wed Jan 10 16:46:43 2018

@author: kkalla
"""

import tensorflow as tf
import numpy as np
import tensorflow.contrib.rnn as rnn

from tensorflow.contrib import legacy_seq2seq
from tensorflow.contrib import seq2seq

class RNN_Model():
    def __init__(self,args, mode='TRAIN'):
        '''Create the model.
        
        Args:
            args: parsed arguments
            mode: TRAIN | EVAL | INFER
        '''
        # When sample, the batch and seq length = 1
        if mode == 'INFER':
            args.batch_size = 1
            args.seq_length = 1
        cell = rnn.BasicLSTMCell(args.rnn_size,state_is_tuple = True)
        self.cell = cell = rnn.MultiRNNCell([cell]*args.num_layers, state_is_tuple = True)
        
        # Build the inputs and outputs placeholders
        self.input_data = tf.placeholder(tf.int32,[args.batch_size,args.seq_length])
        self.targets = tf.placeholder(tf.int32,[args.batch_size,args.seq_length])
        self.initial_state = cell.zero_state(args.batch_size,dtype = tf.float32)
        
        with tf.name_scope('rnn_encoder'):
            # final w
            softmax_w = tf.get_variable('softmax_w',[args.rnn_size,args.vocab_size])
            # final bias
            softmax_b = tf.get_variable('softmax_b',[args.vocab_size])
            with tf.device('/cpu:0'):
                embedding = tf.get_variable('embedding',[args.vocab_size,args.rnn_size],
                                            dtype = tf.float32)
                inputs = tf.split(tf.nn.embedding_lookup(embedding,self.input_data),
                                  args.seq_length,1)
                inputs = [tf.squeeze(input_,[1]) for input_ in inputs]
        
        def loop(prev, _):
            prev = tf.matmul(prev,softmax_w) + softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(prev,1))
            return tf.nn.embedding_lookup(embedding,prev_symbol)
        
        ## Using legacy_seq2seq#####################################
        outputs, last_state = legacy_seq2seq.rnn_decoder(
                inputs,self.initial_state,cell,loop_function=loop if mode != 'INFER' else None,
                scope = 'rnn_encoder')
        output = tf.reshape(tf.concat(outputs,1),[-1,args.rnn_size])
        self.logits = tf.matmul(output,softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)
        loss = legacy_seq2seq.sequence_loss_by_example(
                logits = [self.logits],targets = [tf.reshape(self.targets,[-1])],
                weights = [tf.ones([args.batch_size*args.seq_length])])
        self.cost = tf.reduce_mean(loss)/args.batch_size/args.seq_length
        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable = False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost,tvars),args.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads,tvars))
        ########################################################
        ## Using current seq2seq
        # Helper
#        helper = seq2seq.TrainingHelper(inputs = inputs,seqence_length = args.seq_length,
#                                        name = 'Training_helper')
#        # Decoder
#        decoder = seq2seq.BasicDecoder(cell = cell,helper = helper,
#                                       initial_state = self.initial_state)
#        
#        # Dynamic decoding
#        outputs, final_state, _ = seq2seq.dynamic_decode(decoder = decoder)
#        sample_id = outputs.sample_id
#        output = tf.reshape(tf.concat(outputs,1),[-1,args.rnn_size])
#        self.logits = outputs.rnn_output
#        loss = seq2seq.sequence_loss(logits = [self.logits],
#                                     targets = [tf.reshape(self.targets,[-1])],
#                                     weights = [tf.ones([args.batch_size/args.seq_length])],
#                                       average_across_timesteps = False)
#        self.cost = tf.reduce_mean(loss)/args.batch_size/args.seq_length
#        self.lr = tf.Variable(0.0, trainable = False)
#        tvars = tf.trainable_variables()
#        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost,tvars),args.grad_clip)
#        optimizer = tf.train.AdamOptimizer(self.lr)
#        self.train_op = optimizer.apply_gradients(zip(grads,tvars))
#       #############################################################
    def sample(self, sess, chars, vocab, num=200,prime='0001',sampling_type=1):
        state = sess.run(self.cell.zero_state(1,tf.float32))
        
        for char in prime[:-1]:
            x = np.zeros((1,1))
            x[0,0] = vocab[char]
            feed = {self.input_data:x,self.initial_state:state}
            [state] = sess.run([self.final_state],feed)
        
        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return int(np.searchsorted(t,np.random.rand(1)*s))
        
        ret = prime
        char = prime[-1]
        for n in range(num):
            x = np.zeros((1,1))
            x[0,0] = vocab[char]
            feed = {self.input_data:x, self.initial_state: state}
            [probs,state] = sess.run([self.probs, self.final_state],feed_dict = feed)
            p = probs[0]
            sample = weighted_pick(p)
            pred = chars[sample]
            ret.append(pred)
            char = pred
        return ret