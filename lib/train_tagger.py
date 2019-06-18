#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 01 07:26:45 2019

@author: stefan
"""

import logging
import argparse
import os
import sys
import codecs
from flair.data import TaggedCorpus
from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.embeddings import FlairEmbeddings, TokenEmbeddings, WordEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.visual.training_curves import Plotter
from typing import List


def train_tagger(options):
	# Define columns
	columns = {1: 'text', 2: 'pos', 3: 'ner'}
	
	# What tag should be predicted?
	tag_type = 'ner'

	# Folder in which train, test and dev files reside
	data_folder = options.iob_dir + '/' + options.correction_mode
	
	# Folder in which to save tagging model and additional information
	tagger_folder = '/'.join([options.tagger_dir,
								options.ner_cycle,
								options.lm_domain,
								options.correction_mode]) + '-stringemb'

	# Retrieve corpus using column format, data folder and the names of the train, dev and test files
	corpus: TaggedCorpus = NLPTaskDataFetcher.load_column_corpus(data_folder, columns,
																	train_file='train.txt',
																	test_file='test.txt',
																	dev_file='dev.txt')
	
	# Make the tag dictionary from the corpus
	tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

	# Initialize embeddings
	char_embeddings = [
		FlairEmbeddings(options.lm_dir + options.lm_domain + '-fw/best-lm.pt', use_cache=False),
		FlairEmbeddings(options.lm_dir + options.lm_domain + '-bw/best-lm.pt', use_cache=False)]
	
	if not options.use_wiki_wordemb:
		if not options.use_press_wordemb:
			embedding_types: List[TokenEmbeddings] = char_embeddings
		else:
			embedding_types: List[TokenEmbeddings] = [WordEmbeddings('resources.d/embeddings/fasttext/pressfr-wikifr')] + char_embeddings
			tagger_folder = tagger_folder + '-wordemb-pr'
	else:
		embedding_types: List[TokenEmbeddings] = [WordEmbeddings('fr')] + char_embeddings
		tagger_folder = tagger_folder + '-wordemb'
	
	if options.use_crf:
		tagger_folder = tagger_folder + '-crf'
	
	# Print information
	print(tagger_folder)
	print(corpus)
	print(tag_dictionary.idx2item)
	
	embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

	# Initialize sequence tagger
	tagger: SequenceTagger = SequenceTagger(hidden_size=256,
											embeddings=embeddings,
											tag_dictionary=tag_dictionary,
											tag_type=tag_type,
											use_crf=options.use_crf)

	# Initialize trainer
	trainer: ModelTrainer = ModelTrainer(tagger, corpus)

	# Start training
	trainer.train(tagger_folder,
					learning_rate=0.1,
					mini_batch_size=32,
					max_epochs=50,
					patience=options.train_patience,
					#train_with_dev=True,
					anneal_against_train_loss=False,
					embeddings_in_memory=False)

	# Plot training curves (optional)
	plotter = Plotter()
	plotter.plot_training_curves(tagger_folder + '/loss.tsv')
	plotter.plot_weights(tagger_folder + '/weights.txt')

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
	parser.add_argument('-s', '--lm_dir',
						action='store', dest='lm_dir', default='resources.d/taggers/language-model/',
						help='directory where LMs are stored %(default)')
	parser.add_argument('-i', '--iob_dir',
						action='store', dest='iob_dir', default='data.d/quaero/quaero_iob',
						help='directory where iob training material is located %(default)')
	parser.add_argument('-t', '--tagger_dir',
						action='store', dest='tagger_dir', default='resources.d/taggers',
						help='directory where to store training output %(default)')
	parser.add_argument('-n', '--ner_cycle',
						action='store', dest='ner_cycle', default='ner',
						help='ner experiment cycle %(default)')
	parser.add_argument('-c', '--correction_mode',
						action='store', dest='correction_mode', default='raw',
						help='correction mode of the NEs in training data %(default)')
	parser.add_argument('-m', '--lm_domain',
						action='store', dest='lm_domain', default='pressfr',
						help='character level language model domain %(default)')
	parser.add_argument('-p', '--train_patience',
						action='store', dest='train_patience', type=int, default=3,
						help='training patience %(default)')
	parser.add_argument('-W', '--use_wiki_wordemb',
						action='store_true', dest='use_wiki_wordemb', default=False,
						help='use pre-trained wiki word embeddings')
	parser.add_argument('-P', '--use_press_wordemb',
						action='store_true', dest='use_press_wordemb', default=False,
						help='use indomain press word embeddings')
	parser.add_argument('-C', '--use_crf',
						action='store_true', dest='use_crf', default=False,
						help='use CRF layer')
	parser.add_argument('args', nargs='*')
	options = parser.parse_args()
	if options.logfile:
		logging.basicConfig(filename=logfile)
	if options.debug:
		logging.basicConfig(level=logging.DEBUG)

	train_tagger(options)


if __name__ == '__main__':
	main()
