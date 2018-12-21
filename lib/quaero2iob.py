#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 14:30:14 2018


@author: stbirc
"""

import Cutter
import os
import re
import spacy
import sys

from bs4 import BeautifulSoup

try:
    if len(sys.argv) == 2:
        path2files = sys.argv[1]
        correction_mode = 'n'
        print('Path to input files:', path2files)
        print('Option for correction mode not provided. Corrections will not be used.')
    else:
        path2files = sys.argv[1]
        correction_mode = sys.argv[2]
        print('Path to the input files:', path2files)
        if correction_mode == 'c':
            print('Correction mode: use corrected NEs')
        else:
            correction_mode == 'n'
            print('Correction mode: use erroneous NEs')
except(NameError, IndexError):
    print('Please indicate at least a valid path to the input files.')
    sys.exit(1)

iob_folder = '../resources/quaero_iob/' 
if not os.path.exists(iob_folder):
    os.makedirs(iob_folder)

iob_file = re.sub('/', '_', path2files.split('/data/')[1]) + correction_mode + '.tsv'
output_file = open(iob_folder + iob_file, 'w')
print('Generating output file', iob_file)

cutter = Cutter.Cutter(profile='fr')
nlp = spacy.load('fr')

re_newline = re.compile('(?<!\.|\!|\?)(?<!\.\s»)(?<!\.»)(?<![A-Z])\n(?![A-Z])')
re_punct = re.compile('(?<=\.|\!|\?)\s+(?![a-z]|<)')

pronouns = ['moi', 'toi', 'lui', 'il', 'elle', 'vous', 'nous', 'ce'] # expandable list

quaero_types = ['amount', 'func.coll', 'func.ind', 'loc.add.elec',
                'loc.add.phys', 'loc.adm.nat', 'loc.adm.reg', 'loc.adm.sup',
                'loc.adm.town', 'loc.fac', 'loc.oro', 'loc.other', 'loc.phys.astro',
                'loc.phys.geo', 'loc.phys.hydro', 'loc.unk', 'org.adm', 'org.ent',
                'pers.coll', 'pers.ind', 'prod.art', 'prod.award', 'prod.doctr',
                'prod.fin', 'prod.media', 'prod.object', 'prod.other', 'prod.rule',
                'prod.soft', 'prod.serv', 'time.date.abs', 'time.date.rel',
                'time.hour.abs', 'time.hour.rel', 'loc.unknown', 'pers.unknown',
                'org.unknown', 'prod.unknown', 'time.unknown', 'func.unknown',
                'pers.other', 'org.other', 'time.other']
quaero_components = ['address-number', 'award-cat', 'century', 'day', 'demonym',
                     'demonym.nickname', 'extractor', 'kind', 'millenium', 'month',
                     'name', 'name.first', 'name.last', 'name.middle', 'name.nickname',
                     'noisy-entities', 'object', 'other-address-component', 'po-box',
                     'qualifier', 'range-mark', 'reference-era', 'time-modifier',
                     'title', 'unit', 'val', 'week', 'year', 'zip-code']
quaero_tags = quaero_types + quaero_components

def filter_list(list_element):
    '''
    Helper function to filter out lists containing an empty string.
    
    :param list_element: list
    :return: True or False
    '''
    if list_element == ['']:
        return(False)
    else:
        return(True)

def lowup(input_lowup):
    '''
    Helper function to evaluate the proportion of lower and upper case
    characters in a string.
    
    :param input_lowup: string
    :return: True or False
    '''
    lower = sum(1 for l in input_lowup if l.islower())
    upper = sum(1 for u in input_lowup if u.isupper())
    if lower < upper:
        return(True)
    else:
        return(False)

def html_clean(input_html):
    '''
    Function to clean a string from unwanted (pointy) brackets/html tags.
    :param input_html: string
    :return: string
    '''
    clean_html = re.sub('(<)(/?([^<]*?)(\s|>|$))', lambda x: '*' + x.group(2) if x.group(3) not in quaero_tags else x.group(0), input_html)
    return(clean_html)
        
def pos_tag(input_pos):
    '''
    Function to provide the tokens of a sentence with POS tags.
    
    :param input_lines: list of tuples of strings with token [0], IOB [1]
    :return: list of tuples of strings with token [0], POS [1], IOB [2]
    '''
    doc = nlp.tagger(nlp.tokenizer.tokens_from_list([t[0] for t in input_pos]))
    pos_doc = [d.tag_.split('__')[0] for d in doc]
    iob_doc = [(iob[0], pos_doc[i], iob[1]) for i, iob in enumerate(input_pos)]
    return(iob_doc)  

def lines2frags(input_lines, correction):
    '''
    Function to initially concatenate sentence fragments (strings) which in the
    original documents figure on succesive lines due to space constraints of
    the newspaper format/layout.
    
    :param input_lines: entire document as string 
    :return: list of lists of strings with sentence fragments
    '''
    segments = re.sub('(?<=\w)(- )(\w+)', lambda x: '-' + x.group(2) if x.group(2) in pronouns else x.group(2),
                      re.sub(re_punct, '\n', re.sub(re_newline, ' ', input_lines)))
    html_segments = BeautifulSoup(segments, 'html.parser')
    if correction == 'c':
        for corr in html_segments.find_all(attrs={'correction':True}):
            corr_tag = html_segments.new_tag(corr.name)
            corr_tag.string = corr.attrs['correction']
    else:
        pass
    frags = []
    for c in html_segments.contents:
        if c.name is None:
            frags = frags + [[re.sub('\s+', ' ', s).strip()] for s in str(c).split('\n')]
        else:
            if c.name == 'noisy-entities':
                frags.append([re.sub('\s+', ' ', c.text).strip()])
            elif c.text.strip() != '':
                frags.append([re.sub('\s+', ' ', c.text).strip(), c.name, '-NE-'])
    frags = list(filter(filter_list, frags))
    return(frags)
  
def frags2sents(input_frags):
    '''
    Function to build rudimentary sentences out of the segments obtained from
    the ines2frags() function.
    
    :param input_frags: list of strings sentence fragments
    :return: list of lists of strings with sentences
    '''
    sents = []
    sent = [input_frags[0]]
    for i in range(1, len(input_frags)):
        if input_frags[i][0][0].islower() and not re.search('(\.|\!|\?|[A-Z])$', input_frags[i-1][0]):
            sent.append(input_frags[i])
        elif input_frags[i][-1] == '-NE-' and not re.search('(\.|\!|\?|[A-Z])$', input_frags[i-1][0]):
            sent.append(input_frags[i])
        elif lowup(input_frags[i][0]) is True and lowup(input_frags[i-1][0]) is True: # no ideal choice
            sent.append(input_frags[i])
        else:
            sents.append(sent)
            sent = [input_frags[i]]
    sents.append(sent)
    return(sents)

def sents2iob(input_sents):
    '''
    Function to provide tokens of sentences with IOB annotation.
    
    :param input_sents: sentences as list of lists of lists: [..., [..., ['Le'], ['Pacha', 'func.ind', '-NE-'], ...], ...]
    :return: list of lists of lists with token [0], POS [1], IOB [2]
    '''
    iob_sents = []   
    for sent in input_sents:
        iob_sent = []
        for i, seg in enumerate(sent):
            if seg[-1] == '-NE-':
                ne_tokens = [c[0] for c in cutter.cut(seg[0]) if c[0] != '']
                ne_iob = [[t, 'B-' + seg[1]] if x == 0 else [t, 'I-' + seg[1]] for x, t in enumerate(ne_tokens)]
                # Label initial articles and punctuation with 'O'
                if ne_iob[0][0] in ["l'", "L'", '(', '{', '"', '«']:
                    ne_iob[0][1] = 'O'
                    if ne_iob[1][0] in ["l'", "L'", '(', '{', '"', '«']:
                        ne_iob[1][1] = 'O'
                        ne_iob[2][1] = 'B' + ne_iob[2][1][1:]
                    else:
                        ne_iob[1][1] = 'B' + ne_iob[1][1][1:]
                # Label terminal punctuation with 'O'
                if ne_iob[-1][0] in [')', '}', '"', '»', ';', ',', '.', '?', '!']:
                    ne_iob[-1][1] = 'O'
                    if ne_iob[-2][0] in [')', '}', '"', '»', ';', ',', '.', '?', '!']:
                        ne_iob[-2][1] = 'O'
                iob_sent = iob_sent + ne_iob
            else:
                iob_sent = iob_sent + [[t[0], 'O'] for t in cutter.cut(seg[0]) if t[0] != '']
        iob_sents.append(pos_tag(iob_sent))
    return(iob_sents)

for f, input_file in enumerate(os.listdir(path2files)):
    if input_file.endswith('.norm.ne'):
        print('... processing file', input_file, '[' + str(f+1) +']')
        output_file.write('# FILE: ' + input_file + '\n\n')
        with open(path2files + input_file) as page:
            lines = html_clean(html_clean('\n'.join([p.strip() for p in page if p.strip()[-4:] != '.png' and len(p.strip()) > 0])))
            iob_file = sents2iob(frags2sents(lines2frags(lines, correction_mode)))
            for iob_s in iob_file:
                for i, iob_w in enumerate(iob_s):
                    output_file.write(str(i+1) + '\t' + iob_w[0] + '\t' + iob_w[1] + '\t' + iob_w[2] + '\n')
                output_file.write('\n')
    else:
        print('File', input_file, "doesn't match the requirements, it will be skipped.")
            
output_file.close()

print('Conversion to IOB format finished!')  