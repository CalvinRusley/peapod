#!/usr/bin/env python
import math
import numpy as np
import numba as nb

@nb.njit
def _ungap_matrix(S,alignment1,alignment2):
    return S[alignment1, :][:, alignment2]

@nb.njit
def _mean_path_score(ungapped_matrix):
    return np.mean(ungapped_matrix)*ungapped_matrix.shape[0]

@nb.njit
def _sum_elements_squared(ungapped_matrix):
    sumelementsquares = np.sum(np.square(ungapped_matrix))
    return sumelementsquares

@nb.njit
def _sum_of_crossterms(ungapped_matrix):
    N = ungapped_matrix.shape[0]
    matrixsum = np.sum(ungapped_matrix)
    columnssums = np.sum(ungapped_matrix, axis=0)
    rowsums = np.sum(ungapped_matrix, axis=1)
    crosstermsum = 0
    for index0 in range(N):
        for index1 in range(N):
            sij = ungapped_matrix[index0][index1]
            slicedsum = matrixsum - rowsums[index0] - columnssums[index1] + sij
            crosstermsum += sij * slicedsum
    return crosstermsum

@nb.njit
def _variance(ungapped_matrix,mu):
    #Calculating sqrt((sum of elements squared/N)+((sum of sij*cross terms over all i and j)/(N * N-1))-µ^2)
    N = ungapped_matrix.shape[0]
    if not (N==ungapped_matrix.shape[1]):
        print('ERROR: UNGAPPED MATRIX IS NOT SQUARE')
        print('axis 0 is ' + str(N))
        print('axis 1 is ' + str(ungapped_matrix.shape[1]))
    first_term = (_sum_elements_squared(ungapped_matrix)/N) + (_sum_of_crossterms(ungapped_matrix)/(N*(N-1)))
    variance_squared = first_term - (mu**2)
    return math.sqrt(variance_squared)

def pozitiv(S,positions):
    """ Computes the mean and standard deviation necessary for the POZITIV score of an alignment as per Booth et al. 2004.

        Parameters:
            S (matrix object): similarity matrix
            positions (positions object): aligned positions

        Returns:
            mu (float): mean
            sigma (float): standard deviation
    """
    if (S.deflines0 != positions.deflines0) or (S.deflines1 != positions.deflines1):
        raise Exception('One or both dimensions of the similarity matrix you supplied do not match the deflines of the positions.')
    alignment1 = np.array(positions.coordinates[:,0]).astype(np.int64)
    alignment2 = np.array(positions.coordinates[:,1]).astype(np.int64)
    if len(alignment1)>1:
        ungapped_matrix = _ungap_matrix(S.array,alignment1,alignment2)
        mu = _mean_path_score(ungapped_matrix)
        sigma = _variance(ungapped_matrix,mu)
    else:
        mu = positions.score
        sigma = 1
    return mu, sigma

def gapless_score(S, positions):
    """ Computes the score of an alignment ignoring gap penalties.

        Parameters:
            S (matrix object): similarity matrix
            positions (positions object): aligned positions

        Returns:
            score (float): alignment score ignoring gap penalties
    """
    if (S.deflines0 != positions.deflines0) or (S.deflines1 != positions.deflines1):
        raise Exception('One or both dimensions of the similarity matrix you supplied do not match the deflines of the positions.')
    alignment1 = np.array(positions.coordinates[:,0]).astype(np.int64)
    alignment2 = np.array(positions.coordinates[:,1]).astype(np.int64)
    return np.trace(_ungap_matrix(S.array,alignment1,alignment2))

