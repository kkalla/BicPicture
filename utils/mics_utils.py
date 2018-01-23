# -*- coding: utf-8 -*-
"""
Useful utility functions

Created on Fri Jan 19 21:57:54 2018

@author: user
"""
from __future__ import print_function

import sys

def print_out(s, f=None,new_line=True):
    if isinstance(s,bytes):
        s = s.decode('utf-8')
    
    if f:
        f.write(s.encode('utf-8'))
        if new_line:
            f.write(b'\n')
    
    # stdout
    out_s = s.encode('utf-8')
    if not isinstance(out_s, str):
        out_s = out_s.decode('utf-8')
    print(out_s,end="",file=sys.stddout)
    
    if new_line:
        sys.stdout.write('\n')
    sys.stdout.flush()