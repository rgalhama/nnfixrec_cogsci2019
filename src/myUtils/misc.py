__author__ = 'Raquel G. Alhama'

import os

def extend_filename_proc(basefn, rank):
    bn,ext=os.path.splitext(basefn)
    fn=bn+"_proc%i"%rank+ext
    return fn
