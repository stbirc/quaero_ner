#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 12:39:34 2018

@author: stefan
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, rc, ticker

#try:
#    if len(sys.argv) == 2:
#        path2file = sys.argv[1]
#        correction_mode = 'n'
#        print('Path to the input files:', path2file)
#        print('Option for correction mode not provided. Correction mode: use erroneous NEs')
#    else:
#        path2file = sys.argv[1]
#        correction_mode = sys.argv[2]
#        print('Path to the input files:', path2file)
#        if correction_mode == 'c':
#            print('Correction mode: use corrected NEs')
#        else:
#            correction_mode == 'n'
#            print('Correction mode: use erroneous NEs')
#except(NameError, IndexError):
#    print('Please indicate at least a valid path to the input files.')
#    sys.exit(1)

stats_folder = '../resources/quaero_statistics/' 
if not os.path.exists(stats_folder):
    os.makedirs(stats_folder)

iob_folder = '../resources/quaero_iob/'    
    
quaero_types = ['pers', 'func', 'loc', 'amount', 'time', 'org', 'prod']

def unit_count(input_unit):
    '''
    Function to determine the token and character lengths of a NE.
    
    :param input_unit: NE as list of strings (tokens)
    :return: token length and character length of the NE
    '''
    toks = input_unit
    chars = ' '.join(input_unit)
    l_toks = len(toks)
    l_chars = len(chars)
    return(l_toks, l_chars)
    
def sent2ioblist(input_sent):
    '''
    Function to collect inforamtions/counts of the NEs in a sentences.
    
    :param input_sent: sentence as list of tuples with (token), (PoS), (IOB tag)
    :return: list of 6 tuples with (quaero type), (quaero subtype), (tokens), (PoS), (N tokens), (N characters) for each entity in a sentence and list of quaero components (erroneous annotations) 
    '''
    ne_list = [(), (), (), (), (), ()]
    faulty = []
    i = 0
    while i < len(input_sent):
        if input_sent[i][2] == 'O':
            i += 1
        elif input_sent[i][2].startswith('B'):
            quaero_type = input_sent[i][2][2:].split('.')[0]
            if quaero_type in quaero_types:
                ne = [input_sent[i]]
                quaero_subtype = input_sent[i][2][2:]
                ne_list[0] = ne_list[0] + (quaero_type,)
                ne_list[1] = ne_list[1] + (quaero_subtype,)
                i += 1
                while i < len(input_sent) and input_sent[i][2][0] == 'I':
                    ne.append(input_sent[i])
                    i += 1
                ne_tokens = [t[0] for t in ne]
                ne_pos = [p[1] for p in ne]
                l = unit_count(ne_tokens)
                ne_list[2] = ne_list[2] + (' '.join(ne_tokens),)
                ne_list[3] = ne_list[3] + (' '.join(ne_pos),)
                ne_list[4] = ne_list[4] + (l[0],)
                ne_list[5] = ne_list[5] + (l[1],)
            else:
                faulty.append(quaero_type)
                i += 1
        else:
            i += 1
    return(ne_list, faulty)

def dftypes(dataframe):
    '''
    Function to create one data frame per NE type.
    
    :param dataframe: overall pandas data frame
    :return: data frames per NE type as dictionary {NE type : corresponding data frame}
    '''
    df_types = {}
    for ne_type in quaero_types:
        df_types['df_' + ne_type] = dataframe.loc[dataframe['type'] == ne_type]
    return(df_types)

def countfigures(dataframes, start, stop, step, unit):
    '''
    Function to create/save unit length figures (histograms).
    
    :param dataframes: dictionary with NE type (key) and corresponding data frame (value)
    :param start: int, start of np.arange interval
    :param stop: int, end of np.arange interval
    :param step: int, spacing between np.arange values
    :param unit: string, unit for which to create figure
    :return:
    '''
    u = 'n_' + unit + 's'
    
    color = cm.get_cmap('Set2')
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    tick = ticker.StrMethodFormatter('{x:,.0f}')
    
    bins = np.arange(start, stop, step)
    
    fig, ax = plt.subplots(figsize=(6, 3))
    _, bins, patches = plt.hist([np.clip(dataframes['df_pers'][u], bins[0], bins[-1]),
                                 np.clip(dataframes['df_func'][u], bins[0], bins[-1]),
                                 np.clip(dataframes['df_loc'][u], bins[0], bins[-1]),
                                 np.clip(dataframes['df_amount'][u], bins[0], bins[-1]),
                                 np.clip(dataframes['df_time'][u], bins[0], bins[-1]),
                                 np.clip(dataframes['df_org'][u], bins[0], bins[-1]),
                                 np.clip(dataframes['df_prod'][u], bins[0], bins[-1])],
                                bins=bins,
                                color=[color(0), color(1), color(2), color(3), color(4), color(5), color(6)],
                                label=[r'\textit{pers}', r'\textit{func}', r'\textit{loc}', r'\textit{amount}', r'\textit{time}', r'\textit{org}', r'\textit{prod}'],
                                histtype='bar')
    
    xlabels = np.array([str(bins[i-1]) + '-' + str(bins[i] - 1) for i in range(1, len(bins) - 1)])
    xlabels = np.append(xlabels, [str(bins[-2]) + '+'])
    
    N_labels = len(xlabels)
    
    plt.xlim([start, stop - step])
    plt.xticks(step * np.arange(N_labels) + step / 2)
    ax.set_xticklabels(xlabels)
    ax.yaxis.set_major_formatter(tick)
    ax.yaxis.grid(True, linewidth=0.25, color='0.75', linestyle='--')
    
    #plt.yticks([])
    plt.title('')
    plt.setp(patches, linewidth=0)
    plt.legend(loc='upper right')
    
    plt.xlabel('number of ' + unit + 's')
    plt.ylabel('number of entities')
    
    fig.tight_layout()
    
    plt.savefig(stats_folder + 'iob_' + unit + 's.png', dpi=300)

def counttable(dataframe, units, max_len):
    '''
    Function to 
    
    :param dataframe: overall pandas data frame
    :param units: string, units for which to create figure
    :param max_len: int, length at which to clip
    :return: data frame with counts
    '''
    df_counts = dataframe[['type', 'n_' + units]]
    df_counts['n_' + units] = df_counts['n_' + units].clip(0,max_len)
    df_counts = df_counts.groupby(['type', 'n_' + units]).size().unstack(fill_value=0)
    df_counts['Total'] = df_counts.sum(axis=1)
    df_counts['%'] = ((df_counts['Total']*100)/df_counts['Total'].sum()).round(2)
    df_counts = df_counts.sort_values(by='Total', ascending=False)
    df_counts.loc[len(df_counts)] = df_counts.sum(axis=0)
    df_counts.loc[len(df_counts)] = ((df_counts.iloc[7, 0:-2]*100)/df_counts.iloc[7, 0:-2].sum()).round(2)
    df_counts.iloc[:-1, :-1].astype(int)
    return(df_counts)

c = 0
ne_tuples = []
faulty_annotations = []

for tsv in [iob_folder + 'test_annotated-normalized_c.tsv', iob_folder + 'training_norm_trn_c.tsv']: #'test_annotated-normalized_c.tsv', 'training_norm_trn_c.tsv'
    data_set = tsv.split('_')[0]
    with open(tsv) as iob_file:
        lines = '\n'.join([l.strip() for l in iob_file if not l.startswith('FILE')])
        iob_sentences = [[(iob[1], iob[2], iob[3]) for iob in [tok.split('\t') for tok in sent.split('\n')] if len(iob) > 1] for sent in lines.split('\n\n')]
        for iob_sent in iob_sentences:
            c += len(iob_sent)
            iob_list = sent2ioblist(iob_sent)
            nes = iob_list[0]
            if iob_list[1]:
                faulty_annotations = faulty_annotations + [f for f in iob_list[1]]
            if nes[0]:
                for n in range(len(nes[0])):
                    ne_tuples.append((data_set, nes[0][n], nes[1][n], nes[2][n], nes[3][n], nes[4][n], nes[5][n]))
                    
df = pd.DataFrame(ne_tuples, columns = ['data_set', 'type', 'subtype', 'tokens', 'pos', 'n_tokens', 'n_characters'])

countfigures(dftypes(df), 0, 35, 5, 'character')
countfigures(dftypes(df), 0, 12, 2, 'token')

print(counttable(df, 'tokens', 8))
print(counttable(df, 'characters', 25))
