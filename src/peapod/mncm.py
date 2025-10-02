#!/usr/bin/env python
import numpy as np
from peapod import utilities as utils

def _pairwise_best_hit(S,score_threshold):
    if score_threshold == None:
        boolean_max_matrix = (S.array.max(axis=1,keepdims=1) == S.array) & (S.array.max(axis=0,keepdims=1) == S.array)
    else:
        boolean_max_matrix = (S.array.max(axis=1,keepdims=1) == S.array) & (S.array.max(axis=0,keepdims=1) == S.array) & (S.array > score_threshold)
    return np.transpose(boolean_max_matrix.nonzero())

def _labeling(pbh,jmax):
    copy = pbh.copy()
    edges = np.concatenate((copy,np.zeros((copy.shape[0],1))),axis=1)
    origin_sorted = edges[edges[:, 0].argsort()]
    jlabels = np.zeros(jmax)
    for i in range(origin_sorted.shape[0]):
        j = int(origin_sorted[i,1])
        origin_sorted[i,2] = 1 + max(jlabels[0:j],default=0)
        currentjlabel = jlabels[j]
        jlabels[j] = max(currentjlabel,origin_sorted[i,2])
    k_sorted = origin_sorted[origin_sorted[:, 2].argsort()]
    split_by_k = np.split(k_sorted, np.unique(k_sorted[:, 2], return_index=True)[1][1:])
    edge_dict = {}
    #if split_by_k.shape[0] > 0:
    for array in split_by_k:
        edge_dict[int(array[0,2])] = array[:,0:2].tolist()
    return edge_dict

def _weighted_labeling(pbh,jmax,S):
    copy = np.zeros((pbh.shape[0],pbh.shape[1]+1))
    for index in range(pbh.shape[0]):
        copy[index,0] = pbh[index,0]
        copy[index,1] = pbh[index,1]
        copy[index,2] = S.array[pbh[index,0],pbh[index,1]]
    edges = np.concatenate((copy,np.zeros((copy.shape[0],1))),axis=1)
    origin_sorted = edges[edges[:, 0].argsort()]
    #print(origin_sorted)
    #print('\n:origin_sorted after label loop:')
    jlabels = np.zeros(jmax)
    for i in range(origin_sorted.shape[0]):
        j = int(origin_sorted[i,1])
        origin_sorted[i,3] = origin_sorted[i,2] + max(jlabels[0:j],default=0)
        currentjlabel = jlabels[j]
        jlabels[j] = max(currentjlabel,origin_sorted[i,3])
    #print(origin_sorted)
    #print('\nk_sorted:')
    k_sorted = origin_sorted[origin_sorted[:, 3].argsort()]
    #print(k_sorted)
    #k_ranks_sorted = np.concatenate((k_sorted,np.arange(1,k_sorted.shape[0]+1)[:, np.newaxis]),axis=1)
    split_by_k = np.split(k_sorted, np.unique(k_sorted[:, 3], return_index=True)[1][1:])
    edge_dict = {}
    for index, array in enumerate(split_by_k):
        edge_dict[index] = array[:,0:2].tolist()
    #print('\nedge_dict:')
    #print(edge_dict)
    return edge_dict


def _find_best(S, solns):
    #print(solns)
    scores = []
    for mncm in solns:
        total = 0
        for edge in mncm:
            total += S.array[int(edge[0]),int(edge[1])]
        #print(total)
        scores.append(total)
    #print(scores)
    index = scores.index(max(scores))
    final_edges = solns[index]
    return (final_edges)

def _not_crossing_tuple(e1, e2):
    #input: two tuples
    #output: boolean, True if the edges do not cross and False otherwise
    if ((e1[0] < e2[0]) and (e1[1] < e2[1])) or ((e1[0] > e2[0]) and (e1[1] > e2[1])):
        return True
    else:
        #print(str(e2)+' conflicts with '+str(e1))
        return False    

def _pick_mncm(edgelabelbuckets,k,S,soln=[],soln_found=False):
    if soln != [] and soln_found:
        return (soln, soln_found)
    solns = []
    maxes = edgelabelbuckets.get(k, [])
    all_soln = soln
    if len(maxes)==0:
        soln_found=True
        return (all_soln,soln_found)
    elif len(maxes)==1 or k==1:
        edge = maxes[0]
        if soln == None:
            return (all_soln, soln_found)
        elif soln == [] or all([_not_crossing_tuple(edge, soln_edge) for soln_edge in soln]):
            new_soln = soln + [edge]
            all_soln, soln_found = _pick_mncm(edgelabelbuckets, k-1, S, new_soln, soln_found)
            return (all_soln, soln_found)
        else:
            return (all_soln, soln_found)
    else:
        for maximum in maxes:
            if all(_not_crossing_tuple(soln_edge,maximum) for soln_edge in soln):
                new_soln = all_soln+[maximum]
                to_add, soln_found = _pick_mncm(edgelabelbuckets, k-1, S, new_soln, soln_found)
                solns = solns + [to_add]
        if len(solns) == 1:
            soln_found = True
            [final] = solns
            soln_found = True
            return(final, soln_found)
        else:
            best = _find_best(S,solns)
            soln_found = True
            return (best, soln_found)


def _pick_wmncm(edgelabelbuckets,k_list,S,soln = [],soln_found = False): 
    if soln != [] and soln_found:
        return (soln, soln_found)    
    solns = []
    all_soln = soln
    if len(k_list) == 0:
        soln_found = True
        return (all_soln, soln_found)
    k_max = max(k_list)
    k_list_next = [k for k in k_list if k != k_max]
    maxes = edgelabelbuckets[k_max]
    if len(maxes) >1:
        maxes = maxes[0]
    if len(maxes)==1 or k_max==1 or len(k_list)==1:
        edge = maxes[0]
        if soln == None:
            return (all_soln, soln_found)
        elif soln == [] or all([_not_crossing_tuple(edge, soln_edge) for soln_edge in soln]):
            new_soln = soln + [edge]
            all_soln, soln_found = _pick_wmncm(edgelabelbuckets, k_list_next, S, new_soln, soln_found)
            return (all_soln, soln_found)
        else:
            all_soln, soln_found = _pick_wmncm(edgelabelbuckets, k_list_next, S, soln, soln_found)
            return (all_soln, soln_found)
    else:
        for maximum in maxes:
            if all([_not_crossing_tuple(soln_edge,maximum) for soln_edge in soln]):
                new_soln = all_soln+ [maximum]
                to_add, soln_found = _pick_wmncm(edgelabelbuckets, k_list_next, S, new_soln, soln_found)
                solns = solns + [to_add]
        if len(solns) == 1:
            soln_found = True
            [final] = solns
            soln_found = True
            return(final, soln_found)
        else:
            best = _find_best(S,solns)
            soln_found = True
            return(best, soln_found)


def _pseudoscore(S,positions):
    total = 0
    for position in positions:
        total += S.array[int(position[0]),int(position[1])]
    return total


def mncm(S,score_threshold=None):
    """ Based on a scoring matrix, return the set of pairwise best hit positions that constitute the maximum non-crossing set.

        Parameters:
            S (matrix object): similarity matrix
            score_threshold (float): score below which paired positions will be excluded from consideration.

        Returns:
            positions object
    """
    jmax = max(S.array.shape)
    pbh = _pairwise_best_hit(S,score_threshold)
    if pbh.shape[0] == 0:
        position_array = np.array([])
        score = 0
    else:
        edge_buckets = _labeling(pbh,jmax)
        maxk = max(edge_buckets.keys())
        positions = _pick_mncm(edge_buckets,maxk,S)[0]
        positions.reverse()
        score = _pseudoscore(S,positions)
        position_array = np.array(positions)
    return utils.positions(S.deflines0,S.deflines1,position_array,score=score)


def wmncm(S,score_threshold=None):
    """ Based on a scoring matrix, return the set of pairwise best hit positions that constitute the weighted maximum non-crossing set.

        Parameters:
            S (matrix object): similarity matrix
            score_threshold (float): score below which paired positions will be excluded from consideration.

        Returns:
            positions object
    """
    jmax = max(S.array.shape)
    pbh = _pairwise_best_hit(S,score_threshold)
    if pbh.shape[0] == 0:
        position_array = np.array([])
        score = 0
    else:
        edge_buckets = _weighted_labeling(pbh,jmax,S)
        k_list = list(edge_buckets.keys())
        positions = _pick_wmncm(edge_buckets,k_list,S)[0]
        positions.reverse()
        score = _pseudoscore(S,positions)
        position_array = np.array(positions)
    return utils.positions(S.deflines0,S.deflines1,position_array,score=score)