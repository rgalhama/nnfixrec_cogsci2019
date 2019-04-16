__author__ = 'Raquel G. Alhama'
from models.mappings.S2I import *

def get_char_mapping(words):
    c2i = String2IntegerMapper()
    for wordform in words:
        for c in wordform:
            c2i.add_string(c)
    return c2i

def get_word_mapping(words):
    w2i = String2IntegerMapper()
    for w in words:
        w2i.add_string(w)
    return w2i
