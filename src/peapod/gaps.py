#!/usr/bin/env python
import numpy as np
import numba as nb

### A collection of example gap functions

#####################################################################################################################
################################################ Affine gap functions ###############################################
#####################################################################################################################

@nb.njit
def affinegap_prostt5(gap):
    return 15

@nb.njit
def affinegap_ankhbase(gap):
    return 7

@nb.njit
def affinegap_ankhlarge(gap):
    return 5

@nb.njit
def affinegap_prott5(gap):
    return 14

@nb.njit
def affinegap_esm2(gap):
    return 12

@nb.njit
def affinegap_esm1b(gap):
    return 10

@nb.njit
def no_penalty_gap(gap):
    return 0

@nb.njit
def affinegap_blosum(gap):
    return 3 + (0.5*gap)




@nb.njit
def affinegap_prostt5_rc(gap):
    return 16+(4*gap)

@nb.njit
def affinegap_blosum45_rc(gap):
    return 16+gap


#####################################################################################################################
############################################## Logarithmic gap functions ############################################
#####################################################################################################################

@nb.njit
def loggap(gap):
    return 2 + (1*np.log(gap))


#####################################################################################################################
########################################### Logarithmic affine gap functions ########################################
#####################################################################################################################

@nb.njit
def logaffinegap(gap):
    return 3 + (2*gap) + (1*np.log(gap))



