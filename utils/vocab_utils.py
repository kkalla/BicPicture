# -*- coding: utf-8 -*-
"""
Utility to handle vocabularies

Created on Fri Jan 19 21:01:22 2018

@author: kkalla
"""
import codecs
import os

import tensorflow as tf

from tensorflow.contrib import lookup

from . import mics_utils as utils

SOS = 'sos'
EOS = 'eos'


def load_vocab(vocab_file):
    vocab = []
    with codecs.getreader('utf-8')(tf.gfile.GFile(vocab_file,'rb')) as f:
        vocab_size = 0
        for word in f:
            vocab_size += 1
            vocab.append(word.strip())
    return vocab, vocab_size

def check_vocab(vocab_file,data_dir,check_special_tokens = True,sos=None,eos=None):
    """Check vocab file"""
    if tf.gfile.Exists(vocab_file):
        utils.print_out('# Vocab file %s exists' % vocab_file)
        vocab, vocab_size = load_vocab(vocab_file)
        if check_special_tokens:
            if not sos: sos = SOS
            if not eos: eos = EOS
            assert len(vocab) >= 2
            if vocab[0] != sos or vocab[1] != eos:
                utils.print_out('The first 2 vocab words [%s,%s] are not [%s,%s]'%(
                        vocab[0],vocab[1],sos,eos))
                vocab = [sos,eos] + vocab
                vocab_size += 2
                new_vocab_file = os.path.join(data_dir,os.path.basename(vocab_file))
                with codecs.getwriter('utf-8')(tf.gfile.GFile(new_vocab_file,'wb')) as f:
                    for word in vocab:
                        f.write('%s\n'%word)
                vocab_file = new_vocab_file
    else:
        raise ValueError("vocab_file '%s' does not exists." % vocab_file)
    
    vocab_size = len(vocab)
    return vocab_size, vocab_file

def create_vocab_tables(vocab_file):
    """Creates vocab tables for vocab_file"""
    vocab_table = lookup.index_table_from_file(vocab_file,defalut_value=-1)
    return vocab_table

def load_embed_txt(embed_file):
    """Load embed_file into a python dictionary.
    
    Returns:
        a dictionary that maps word to vector, and the size of embedding dimensions.
    """
    emb_dict