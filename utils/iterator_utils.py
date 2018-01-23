# -*- coding: utf-8 -*-
"""
loading data

Created on Tue Jan 23 17:02:51 2018

@author: user
"""

import tensorflow as tf

def get_iterator(src_dataset,tgt_dataset,
                 vocab_table,
                 batch_size,
                 sos,eos,
                 random_seed,
                 num_buckets,
                 src_max_len=None,
                 tgt_max_len=None,
                 num_parallel_calls=4,
                 output_buffer_size=None,
                 skip_count=None,
                 num_shards=1,
                 shard_index = 0,
                 reshuffle_each_iteration=True):
    if not output_buffer_size:
        output_buffer_size = batch_size * 1000
    src_eos_id = tf.cast(vocab_table.lookup(tf.constant(eos)),tf.int32)
    tgt_sos_id = tf.cast(vocab_table.lookup(tf.constant(sos)),tf.int32)
    tgt_eos_id = tf.cast(vocab_table.lookup(tf.constant(eos)),tf.int32)
    
    src_tgt_dataset = tf.data.Dataset.zip((src_dataset,tgt_dataset))
    src_tgt_dataset = src_tgt_dataset.shard(num_shards,shard_index)
    
    if skip_count is not None:
        src_tgt_dataset = src_tgt_dataset.skip(skip_count)
    
    src_tgt_dataset = src_tgt_dataset.shuffle(
            output_buffer_size,random_seed, reshuffle_each_iteration)
    
    src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt:(
                    tf.strin_split([src]).values, tf.strin_split([tgt]).values),
                    num_parallel_calls = num_parallel_calls).prefetch(output_buffer_size)
    
    # Filter zero length input seq
    src_tgt_dataset = src_tgt_dataset.filter(
            lambda src, tgt: tf.logical_and(tf.size(src)>0,tf.size(tgt)>0))
    
    if src_max_len:
        src_tgt_dataset = src_tgt_dataset.map(
                lambda src, tgt: (src[:src_max_len],tgt),
                num_parallel_calls = num_parallel_calls).prefetch(output_buffer_size)
    if tgt_max_len:
        src_tgt_dataset = src_tgt_dataset.map(
                lambda src, tgt:(src,tgt[:tgt_max_len]),
                num_parallel_calls = num_parallel_calls).prefetch(output_buffer_size)
    
    # Convert item to ids
    src_tgt_dataset = src_tgt_dataset.map(
            lambda src,tgt:(tf.cast(vocab_table.lookup(src),tf.int32),
                            tf.cast(vocab_table.lookup(tgt),tf.int32)),
                num_parallel_calls = num_parallel_calls).prefetch(output_buffer_size)
    src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt:(src,
                             tf.concat(([tgt_sos_id],tgt),0),
                             tf.concat((tgt,[tgt_eos_id]),0)),
            num_parallel_calls = num_parallel_calls).prefetch(output_buffer_size)
    
            