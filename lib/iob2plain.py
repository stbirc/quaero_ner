#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 17:27:45 2018

@author: stefan
"""

import numpy as np

np.random.seed(42)

iob_folder = '../resources/quaero_iob/'
plain_folder = '../resources/quaero_plain/'
if not os.path.exists(plain_folder):
    os.makedirs(plain_folder)

val_plain = open(plain_folder + 'valid.txt', 'w')
test_plain = open(plain_folder + 'test.txt', 'w')

plain_text = []

for tsv in [iob_folder + 'test_annotated-normalized_c.tsv', iob_folder + 'training_norm_trn_c.tsv']:
    with open(tsv) as iob_file:
        lines = '\n'.join([l.strip() for l in iob_file if not l.startswith('FILE')])
        sentences = [' '.join([iob[1] for iob in [tok.split('\t') for tok in sent.split('\n')] if len(iob) > 1]) for sent in lines.split('\n\n')]
        plain_text += sentences

np.random.shuffle(plain_text)
        
train_test_cutoff = int(.80 * len(plain_text)) 
train = plain_text[:train_test_cutoff]
test = plain_text[train_test_cutoff:]
 
test_val_cutoff = int(.5 * len(test))
val = test[:test_val_cutoff]
test = test[test_val_cutoff:]

for v in val:
    val_plain.write(v + '\n')
for t in test:
    test_plain.write(t + '\n')

train_100_cutoff = int(.01 * len(train))
#train_100 = []
cut = 0
for i in range(100):
    train_plain = open(plain_folder + 'train/train_split_' + str(i+1) + '.txt', 'w')
    for tr in train[cut:cut + train_100_cutoff]:
        train_plain.write(tr + '\n')
    #train_100.append(train[cut:cut + train_100_cutoff])
    cut += train_100_cutoff
    train_plain.close()

val_plain.close()
test_plain.close()