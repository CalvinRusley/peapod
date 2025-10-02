#!/usr/bin/env python
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import numba as nb
import pandas as pd
from peapod import utilities as utils
from peapod import methods

#####################################################################################################################
################################################## General purpose ##################################################
#####################################################################################################################


def aln_quality(test_positions, ref_positions, force=False):
    """ Calculates the sensitivity and precision of an alignment relative to a reference alignment. 

        Parameters:
            test_positions (positions object): alignment to check
            ref_positions (positions object): reference alignment
            force (boolean): overrides a name-based check of whether the alignments are both of the same sequences (default False)
            
        Returns:
            sensitivity (float): true positives / (true positives + false negatives)
            precision (float): true positives / (true positives + false positives)
    """
    if ((test_positions.deflines0 != ref_positions.deflines0) or (test_positions.deflines1 != ref_positions.deflines1)) and not force:
        raise Exception('Alignments are not of the same sequences.')
    test_aln_positions = set([tuple(x) for x in test_positions.coordinates])
    ref_aln_positions = set([tuple(x) for x in ref_positions.coordinates])
    true_pos = len(test_aln_positions & ref_aln_positions)
    sensitivity = true_pos/len(ref_aln_positions)
    precision = true_pos/len(test_aln_positions)
    return sensitivity, precision


#####################################################################################################################
##################################################### HOMSTRAD ######################################################
#####################################################################################################################


def import_homstrad_pairs(homstrad_pair_dir):
    """ Import a directory of pairs of sequences (one pair per HOMSTRAD family).

        Parameters:
            homstrad_pair_dir (filepath): directory 
            
        Returns:
            dictionary (keys are HOMSTRAD family names, values are profiles dictionaries, each with 2 sequences)
    """
    fastafiles = [join(homstrad_pair_dir, f) for f in listdir(homstrad_pair_dir) if isfile(join(homstrad_pair_dir, f))]
    output = {}
    for fasta in fastafiles:
        homstrad_family_name = os.path.basename(fasta).replace('.faa','')
        output[homstrad_family_name] = utils.fasta_to_profiles_dict(fasta,unalign=True) #unaligning just in case
    return output


def homstrad_benchmark_tools_against_ref(tool_positions,ref_name):
    """ Compare aligned HOMSTRAD positions by a set of tools against a selected set of reference pairwise alignments.

        Parameters:
            tool_positions (dictionary): dictionary where tool names are keys, values are dictionaries 
                (each with HOMSTRAD family names as keys, positions objects as values)
            ref_name (string): name of tool to use as a reference
            
        Returns:
            pandas dataframe
    """
    stats = []
    query_positions = tool_positions.copy()
    ref_positions = query_positions.pop(ref_name, None)
    for tool in query_positions.keys():
        print(tool)
        for idx, name in enumerate(query_positions[tool].keys()):
            pair = [ref_positions[name].deflines0[0],ref_positions[name].deflines1[0]]
            print(name)
            print(str(idx)+' '+str(pair))
            if query_positions[tool][name].coordinates.shape[0]==0:
                sensitivity = 0
                precision = 0
            else:
                sensitivity, precision = aln_quality(query_positions[tool][name], ref_positions[name])
            stats.append([name,pair,tool,sensitivity,precision,ref_name])
    return pd.DataFrame(stats,columns=['family','pair','tool','sensitivity','precision','reference']).replace(to_replace='-', value='0')


def run_method_on_homstrad(homstrad_dict, method_func, gap_func=None):
    """ Test a method on the pairwise HOMSTRAD benchmark. If you're using an embedding based approach, make sure to embed the profile dictionaries in homstrad_dict ahead of time.

        Parameters:
            homstrad_dict (dictionary): output of benchmark.import_homstrad_pairs(), previously embedded if that's what you're using
            method_func (function): a function that takes two profiles as the only required arguments, and returns only a positions object
            gap_func (function): a numba njitted gap function
            
        Returns:
            dictionary (HOMSTRAD family names as keys, positions objects as values)
    """
    output = {}
    for idx, name in enumerate(homstrad_dict.keys()):
        pair = list(homstrad_dict[name].keys())
        print(name)
        print(str(pair))
        profile0 = homstrad_dict[name][pair[0]]
        profile1 = homstrad_dict[name][pair[1]]
        if gap_func == None:
            alignment = method_func(profile0,profile1)
        else:
            alignment = method_func(profile0,profile1,gap_func)
        output[name] = alignment
    return output








#####################################################################################################################
############################################## Very specific functions ##############################################
#####################################################################################################################


def parse_collated_clustalesque_hell(file):
    all_dicts = {}
    with open(file) as f:
        for line in f:
            if line.startswith(">"):
                split_line = line.split()
                name = split_line[0].replace('>','')
                seq0 = split_line[1]
                seq1 = split_line[2]
                tool = split_line[3]
                pair = (seq0,seq1)
                sensitivity = split_line[5]
                precision = split_line[7]
                if tool not in all_dicts:
                    all_dicts[tool] = {}
                    all_dicts[tool][name] = {}
                elif name not in all_dicts[tool]:
                    all_dicts[tool][name] = {}
                all_dicts[tool][name][pair] = {'sensitivity':sensitivity,'precision':precision,'seq_dict':{}}
                all_dicts[tool][name][pair]['seq_dict'].setdefault(seq0, '')
                all_dicts[tool][name][pair]['seq_dict'].setdefault(seq1, '')
                visited = {seq0:False,seq1:False}
            elif line.rstrip():
                split_line = line.split()
                defline = split_line[0]
                sequence = split_line[2].rstrip().upper()
                line_start_position = int(split_line[1])-1
                if (line_start_position>0):
                    preexisting_line = all_dicts[tool][name][pair]['seq_dict'][defline]
                    visit_state = np.array(list(visited.values()))
                    if not visit_state.any(): #if both are false, append
                        # if tool =='HOMSTRAD':
                        #     print('prepending triggered for '+name+' with '+tool)
                        other_defline = [x for x in all_dicts[tool][name][pair]['seq_dict'].keys() if x!=defline][0]
                        all_dicts[tool][name][pair]['seq_dict'][defline] += 'X' * (line_start_position)
                        all_dicts[tool][name][pair]['seq_dict'][other_defline] += '-' * (line_start_position)
                    elif (not visit_state.all()) and (visit_state.any()):#only one is false, prepend
                        other_defline = [x for x in all_dicts[tool][name][pair]['seq_dict'].keys() if x!=defline][0]
                        this_seq_existing = all_dicts[tool][name][pair]['seq_dict'][defline]
                        other_seq_existing = all_dicts[tool][name][pair]['seq_dict'][other_defline]
                        all_dicts[tool][name][pair]['seq_dict'][defline] = ('X' * (line_start_position)) + this_seq_existing
                        all_dicts[tool][name][pair]['seq_dict'][other_defline] = ('-' * (line_start_position)) + other_seq_existing
                # if (''.join(set(sequence)) != '-'):
                all_dicts[tool][name][pair]['seq_dict'][defline] += sequence
                visited[defline] = True
    return all_dicts

def benchmark_to_profiles_positions_stats(file):
    benchmark_dict = parse_collated_clustalesque_hell(file)
    positions_output = {}
    profiles_output = {}
    stats = []
    for tool in benchmark_dict.keys():
        if tool=='HOMSTRAD':
            for i in benchmark_dict[tool].keys():
                for j in benchmark_dict[tool][i].keys():
                    seq_dict = benchmark_dict[tool][i][j]['seq_dict']
                    positions_output[i] = utils.aln_seq_dict_to_pairwise_positions(seq_dict,[j])[j]
                    profiles_output[i] = utils.seq_dict_to_profiles_dict(seq_dict,unalign=True)
        else:
            for i in benchmark_dict[tool].keys():
                for j in benchmark_dict[tool][i].keys():
                    name = i
                    pair = j
                    sensitivity = benchmark_dict[tool][i][j]['sensitivity']
                    precision = benchmark_dict[tool][i][j]['precision']
                    stats.append([name,pair,tool,sensitivity,precision])
    return profiles_output, positions_output, stats

def benchmark_to_positions_profiles(file):
    benchmark_dict = parse_collated_clustalesque_hell(file)
    positions_output = {}
    profiles_output = {}
    for tool in benchmark_dict.keys():
        positions_output[tool] = {}
        profiles_output[tool] = {}
        for i in benchmark_dict[tool].keys():
            for j in benchmark_dict[tool][i].keys():
                seq_dict = benchmark_dict[tool][i][j]['seq_dict']
                positions_output[tool][i] = utils.aln_seq_dict_to_pairwise_positions(seq_dict,[j])[j]
                profiles_output[tool][i] = utils.seq_dict_to_profiles_dict(seq_dict,unalign=False)
    return positions_output, profiles_output



# HOMSTRAD

def import_homstrad_pairwise_positions(homstrad_dir):
    family_dirs = [join(homstrad_dir, f) for f in listdir(homstrad_dir) if not isfile(join(homstrad_dir, f))]
    ali_files = []
    for family in family_dirs:
        ali_files.extend([join(family, f) for f in listdir(family) if isfile(join(family, f)) and f.endswith('.ali')])
    ref_positions = {}
    ref_profiles = []
    for ali_file in ali_files:
        name = ali_file.rpartition('/')[2].replace('.ali','')
        seq_dict = utils.ali_to_seq_dict(ali_file)
        first = list(seq_dict.keys())[0]
        last = list(seq_dict.keys())[-1]
        ref_positions[(first,last)] = utils.aln_seq_dict_to_pairwise_positions(seq_dict,[(first,last)])
        seq_dict_subset = {k:seq_dict[k] for k in (first,last) if k in seq_dict}
        ref_profiles.append(utils.seq_dict_to_profiles_dict(seq_dict_subset,unalign=True))
    return ref_positions, ref_profiles

def embed_homstrad(ref_profiles,model):
    for profile_dict in ref_profiles:
        methods.batch_embed(profile_dict,model)

def check_positions_against_homstrad(query_positions,ref_positions):
    output = [[],[],[]]
    for idx, query in enumerate(query_positions):
        sensitivity, precision = bm.aln_quality(query, ref_positions[query.deflines0,query.deflines1], force=False)
        output[idx,0] = sensitivity
        output[idx,1] = precision
        output[0].append((query.deflines0,query.deflines1))
        output[1].append(sensitivity)
        output[2].append(precision)
    return pd.DataFrame(output).T.rename(columns={0:'pair',1:'sensitivity',2:'precision'})





# SCOPe40

# For SCOPe, proteins in different classes or folds are safe "unrelated" sequences, while stuff within the same superfamily or family are safe "related" sequences. If two proteins share the same class/fold, but not the same superfamily or family, don't evaluate because it's not certain they're related or unrelated

def import_scope(scope_fasta):
    profile_dict = utils.seq_dict_to_profiles_dict(utils.fasta_to_seq_dict(scope_fasta))
    scope_classifications = {}
    for key in profile_dict.keys():
        scope_classifications[key] = {}
        classifications = key.split()[1].split('.')
        scope_classifications[key]['class'] = classifications[0]
        scope_classifications[key]['fold'] = classifications[1]
        scope_classifications[key]['superfamily'] = classifications[2]
        scope_classifications[key]['family'] = classifications[3]
    return profile_dict, scope_classifications

def defensible_comparisons(class_dict):
    all_pairwise_comb = list(combinations(class_dict.keys(), 2))
    same_fam = []
    same_superfam = []
    same_fold = []
    same_class = []
    diff_class = []
    for comb in all_pairwise_comb:
        seq0 = class_dict[comb[0]]
        seq1 = class_dict[comb[1]]
        if (seq0['class'] == seq1['class']):
            if (seq0['fold'] == seq1['fold']):
                if (seq0['superfamily'] == seq1['superfamily']):
                    if (seq0['family'] == seq1['family']):
                        same_fam.append(comb)
                    else:
                        same_superfam.append(comb)
                else:
                    same_fold.append(comb)
            else:
                same_class.append(comb)
        else:
            diff_class.append(comb)
    return same_fam, same_superfam, same_fold, same_class, diff_class




