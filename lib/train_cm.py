#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import argparse
import os
import sys
import codecs
from flair.data import Dictionary
from flair.models import LanguageModel
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus

"""
Module for XXX

"""


sys.stderr = codecs.getwriter('UTF-8')(sys.stderr.buffer)
sys.stdout = codecs.getwriter('UTF-8')(sys.stdout.buffer)
sys.stdin = codecs.getreader('UTF-8')(sys.stdin.buffer)


def process(options):
    """
    Do the processing
    """

    # are you training a forward or backward LM?
    is_forward_lm =  not options.is_backward_lm

    # load the default character dictionary
    dictionary: Dictionary = Dictionary.load('chars')

    # get your corpus, process forward and at the character level
    corpus = TextCorpus(options.corpus_dir,
                        dictionary,
                        is_forward_lm,
                        character_level=True)

    # instantiate your language model, set hidden size and number of layers
    language_model = LanguageModel(dictionary,
                                   is_forward_lm,
                                   hidden_size=2048,
                                   nlayers=1,
                                   embedding_size=100, # recommendations?
                                   dropout=0) # dropout probs?

    # train your language model
    trainer = LanguageModelTrainer(language_model, corpus)

    trainer.train(options.model_dir, # embeddings_in_memory=False: effect on 'RuntimeError: CUDA out of memory'?
                  sequence_length=250,
                  learning_rate=20,
                  mini_batch_size=100,
                  anneal_factor=0.25,
                  patience=22, # 'patience' value of the learning rate scheduler: 1/2 training splits
                  clip=0.25, # clipping gradients?
                  max_epochs=75)

def main():
    """
    Invoke this module as a script
    """
    parser = argparse.ArgumentParser(
        usage = '%(prog)s [OPTIONS] [ARGS...]',
        description='Calculate something',
        epilog='Contact simon.clematide@uzh.ch'
        )
    parser.add_argument('--version', action='version', version='0.99')
    parser.add_argument('-l', '--logfile', dest='logfile',
                      help='write log to FILE', metavar='FILE')
    parser.add_argument('-q', '--quiet',
                      action='store_true', dest='quiet', default=False,
                      help='do not print status messages to stderr')
    parser.add_argument('-d', '--debug',
                      action='store_true', dest='debug', default=False,
                      help='print debug information')
    parser.add_argument('-c', '--corpus_dir',
                      action='store', dest='corpus_dir', default='corpus',
                      help='directory with corpus data %(default)')
    parser.add_argument('-m', '--model_dir',
                      action='store', dest='model_dir', default='model',
                      help='directory with model data %(default)')
    parser.add_argument('-B', '--is_backward_lm',
                      action='store_true', dest='is_backward_lm', default=False,
                      help='build backward model')
    parser.add_argument('args', nargs='*')
    options = parser.parse_args()
    if options.logfile:
        logging.basicConfig(filename=logfile)
    if options.debug:
        logging.basicConfig(level=logging.DEBUG)

    process(options)


if __name__ == '__main__':
    main()
