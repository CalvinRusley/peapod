#!/usr/bin/env python
import numpy as np
import numba as nb
from numba.core import types
from numba.typed import Dict
from peapod import utilities as utils
import torch
from scipy import spatial


#####################################################################################################################
############################################# Minkowski p-norm distance #############################################
#####################################################################################################################

@nb.njit
def _root1(x):
    return x

@nb.njit
def _root2(x):
    return np.sqrt(x)

@nb.njit
def _root3(x):
    return np.cbrt(x)

@nb.njit
def _root4(x):
    return np.sqrt(np.sqrt(x))


optimized_func = {
    1:_root1,
    2:_root2,
    3:_root3,
    4:_root4
}


@nb.njit
def _cdist_opt(emb0,emb1,pnumerator,pdenominator,numerator_func,denominator_func):
    #slightly faster than pytorch in certain non-euclidean cases
    m = np.empty((emb0.shape[0],emb1.shape[0]), np.float64)
    for i in range(emb0.shape[0]):
        for j in range(emb1.shape[0]):
            m[i,j] = numerator_func(np.sum(denominator_func(np.abs(emb0[i]-emb1[j])**pnumerator))**pdenominator)
    return m


def minkowski(profile0, profile1, pnumerator=2, pdenominator=1, l=1):
    """ Take as input 2 embedded profiles and returns the similarity matrix based on the exponentiated 
        minkowski p-norm (or quasinorm) (i.e. e^(-distance)). Matrix dimensions are [len(profile0), len(profile1)].
        Code adapted from Pantolini et al. 2024 (https://git.scicore.unibas.ch/schwede/EBA).

        Parameters:
            profile0 (profile object): first profile
            profile1 (profile object): second profile
            pnumerator (int): numerator of p-norm (minkowski) distance (default 2)
            pdenominator (int): denominator of p-norm (minkowski) distance (default 1)
            l (float): scaling factor to apply before exponentiation
            
        Returns:
            matrix (matrix object): similarity matrix
    """
    if (pnumerator/pdenominator) < 1:
        print('WARNING: p values less than one are quasinorms and the triangle inequality does not hold. See Mirkes et al. 2020 (https://doi.org/10.3390/e22101105) for more information.')
    optimized_roots = optimized_func.keys()
    if profile0.n==1 and profile1.n==1:
        emb0 = profile0.embeddings
        emb1 = profile1.embeddings
    else:
        print('WARNING: You are attempting to align more than two sequences. This has not yet been implemented in PEAPOD.')
    if ((pnumerator/pdenominator) == 2) or (pnumerator not in optimized_roots) or (pdenominator not in optimized_roots):
        sim = torch.exp(-l*torch.cdist(torch.from_numpy(emb0), torch.from_numpy(emb1), p=(pnumerator/pdenominator)))
    else:
        numerator_func = optimized_func[pnumerator]
        denominator_func = optimized_func[pdenominator]
        sim = torch.exp(-l*torch.from_numpy(_cdist_opt(emb0,emb1,pnumerator,pdenominator,numerator_func,denominator_func)))
    return utils.matrix(profile0.deflines,profile1.deflines,sim.numpy())


#####################################################################################################################
################################################# Cosine similarity #################################################
#####################################################################################################################

def cosine(profile0, profile1):
    """ Calculates the cosine similarity matrix. Code modified from Pantolini et al. 2024 (https://git.scicore.unibas.ch/schwede/EBA).

        Parameters:
            profile0 (profile object): first profile
            profile1 (profile object): second profile

        Returns:
            matrix (matrix object): similarity matrix
    """
    if profile0.n==1 and profile1.n==1:
        emb0 = profile0.embeddings
        emb1 = profile1.embeddings
    else:
        print('WARNING: You are attempting to align more than two sequences. This has not yet been implemented in PEAPOD.')
    sim = torch.tensor(1-spatial.distance.cdist(emb0, emb1, 'cosine'))
    return utils.matrix(profile0.deflines,profile1.deflines,sim.numpy())



@nb.njit
def _array_pair_to_nb_dict(keys,values):
    d = {}
    for idx in range(keys.shape[0]):
        key = keys[idx]
        value = values[idx]
        d[key] = value
    return d



#####################################################################################################################
################################################ Substitution matrix ################################################
#####################################################################################################################

def load_substitution_matrix(tsvfilename):
    """ Load a tsv file of melted substitution matrix values as a numba typed dictionary.

        Parameters:
            tsvfilename (string): filepath to file containing melted substitution matrix values

        Returns:
            matrix (matrix object): similarity matrix
    """
    keys = []
    values = []
    with open(tsvfilename) as f:
        for line in f:
            (key, value)=line.split()
            keys.append(key)
            values.append(float(value))
    return _array_pair_to_nb_dict(np.array(keys),np.array(values, dtype='float64'))


@nb.njit
def _substitution_lookup(length0,length1,seq0,seq1,substitution_matrix_dict):
    m = np.empty((length0,length1), np.float64)
    for i, aacid1 in enumerate(seq0):
        for j, aacid2 in enumerate(seq1):
            m[i,j] = substitution_matrix_dict[aacid1+aacid2]
    return m


def substitution(profile0, profile1, substitution_matrix_dict):
    """ Create a similarity matrix based on values from a substitution matrix.

        Parameters:
            profile0 (profile object): first profile
            profile1 (profile object): second profile
            substitution_matrix_dict (numba typed.Dict): dictionary of substitution scores

        Returns:
            matrix (matrix object): similarity matrix
    """
    if (profile0.n > 1) or (profile1.n > 1):
        raise Exception("Either one or both of the profiles provided are not single sequences. This function is not yet supported.")
    seq0 = profile0.aaseqs[0]
    seq1 = profile1.aaseqs[0]
    m = _substitution_lookup(profile0.length,profile1.length,seq0,seq1,substitution_matrix_dict)
    return utils.matrix(profile0.deflines,profile1.deflines,m)




#####################################################################################################################
################################################### Mean distance ###################################################
#####################################################################################################################


def mean_embedding_dist(profile0, profile1, p=2):
    """ Calculates the p-norm (minkowski) distance between the mean pools of two profile embeddings.

        Parameters:
            profile0 (profile object): first profile
            profile1 (profile object): second profile
            p (int): norm of distance to compute (default 2, a.k.a. Euclidean distance)

        Returns:
            matrix (matrix object): matrix object with the distance (float) replacing the array
    """
    emb0mean = profile0.embeddings.mean(axis=0)
    emb1mean = profile1.embeddings.mean(axis=0)
    dist = torch.cdist(torch.from_numpy(emb0mean).unsqueeze(0), torch.from_numpy(emb1mean).unsqueeze(0), p=p).numpy()[0][0]
    return utils.matrix(profile0.deflines,profile1.deflines,dist)
    



#####################################################################################################################
########################################## Similarity matrix manipulation ###########################################
#####################################################################################################################



def enhance_signal(S):
    """ Enhances the signal of a similarity matrix using the Z-score with relation to the row and column of each cell. 
        Code modified from Pantolini et al. 2024 (https://git.scicore.unibas.ch/schwede/EBA).

        Parameters:
            S (matrix object): similarity matrix

        Returns:
            matrix (matrix object): enhanced similarity matrix
    """
    sim = S.array
    columns_avg = torch.sum(torch.from_numpy(sim),0)/torch.from_numpy(sim).shape[0]
    rows_avg = torch.sum(torch.from_numpy(sim),1)/torch.from_numpy(sim).shape[1]
    
    columns_std = torch.std(torch.from_numpy(sim),0)
    rows_std = torch.std(torch.from_numpy(sim),1)

    z_rows = (torch.from_numpy(sim)-rows_avg.unsqueeze(1))/rows_std.unsqueeze(1)
    z_columns = (torch.from_numpy(sim)-columns_avg)/columns_std
    
    enhanced = (z_rows+z_columns)/2
    
    return utils.matrix(S.deflines0,S.deflines1,enhanced.numpy())



def shift(S, shift):
    """ Shifts the values in a similarity matrix.

        Parameters:
            S (matrix object): similarity matrix
            shift (float): amount to add to every value in the array

        Returns:
            matrix (matrix object): similarity matrix
    """
    sim = S.array
    return utils.matrix(S.deflines0,S.deflines1,np.subtract(sim, -1*shift))