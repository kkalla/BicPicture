# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 23:44:55 2018

@author: user
"""
import tensorflow as tf
# from tensorflow.contrib import seq2seq
from tensorflow.contrib import legacy_seq2seq
import tensorflow.contrib.rnn as rnn
import numpy as np

class Model():
    def __init__(self, args, infer=False):
        self.args = args
        if infer: # When sample, the batch and seq length = 1
            args.batch_size = 1
            args.seq_length = 1
        cell_fn = rnn.BasicLSTMCell # Define the internal cell structure
        cell = cell_fn(args.rnn_size,state_is_tuple=True)
        self.cell = cell = rnn.MultiRNNCell([cell]*args.num_layers, state_is_tuple=True)
        #Build the inputs and outputs placeholders
        self.input_data = tf.placeholder(tf.int32, [args.batch_size,args.seq_length])
        self.targets = tf.placeholder(tf.int32, [args.batch_size,args.seq_length])
        self.initial_state = cell.zero_state(args.batch_size, dtype = tf.float32)
            
        with tf.name_scope('rnnlm'):
            # final w
            softmax_w = tf.get_variable("softmax_w", [args.rnn_size,args.vocab_size])
            # final bias
            softmax_b = tf.get_variable("softmax_b", [args.vocab_size])
            with tf.device("/cpu:0"):
                embedding = tf.get_variable("embedding", [args.vocab_size,args.rnn_size],
                                            dtype=tf.float32)
                inputs = tf.split(tf.nn.embedding_lookup(embedding,self.input_data),
                                  args.seq_length,1)
                inputs = [tf.squeeze(input_,[1]) for input_ in inputs]
            
        def loop(prev, _):
            prev = tf.matmul(prev,softmax_w) + softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(prev,1))
            return tf.nn.embedding_lookup(embedding,prev_symbol)
        
        outputs, last_state = legacy_seq2seq.rnn_decoder(
                inputs,self.initial_state,cell,loop_function=loop if infer else None,
                scope="rnnlm")
        output = tf.reshape(tf.concat(outputs,1),[-1,args.rnn_size])
        self.logits = tf.matmul(output,softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)
        loss = legacy_seq2seq.sequence_loss_by_example(
                logits = [self.logits],targets = [tf.reshape(self.targets,[-1])],
                weights = [tf.ones([args.batch_size*args.seq_length])])
        self.cost = tf.reduce_mean(loss)/args.batch_size/args.seq_length
        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost,tvars),args.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads,tvars))
        
    def sample(self, sess, chars, vocab, num=200,prime='START',sampling_type=1):
        state = sess.run(self.cell.zero_state(1,tf.float32))
        
        for char in prime[:-1]:
            x = np.zeros((1,1))
            x[0,0] = vocab[char]
            feed = {self.input_data:x,self.initial_state: state}
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
            feed = {self.input_data:x,self.initial_state: state}
            [probs,state] = sess.run([self.probs,self.final_state],feed_dict = feed)
            p = probs[0]
            sample = weighted_pick(p)
            pred = chars[sample]
            ret += pred
            char = pred
        return ret
            