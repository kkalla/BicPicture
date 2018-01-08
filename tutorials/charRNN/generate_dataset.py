# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 21:04:52 2018

@author: user
"""
import argparse
import random
import os

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',type = str,default = 'data',
                    help = 'directory of stored data')
parser.add_argument('--num_of_songs',type=int,default =24,
                    help = 'number of songs in dataset')
args = parser.parse_args()

input_data = open(os.path.join(args.data_dir,'original.txt'),'r').read().split('X:')
for i in range(1,1000):
    print("X:" + input_data[random.randint(1,args.num_of_songs)]+'\n_____________________________\n')