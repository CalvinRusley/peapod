#!/usr/bin/env python
import numpy as np
import numba as nb

from peapod import utilities as utils
from peapod import alignment as aln
from peapod import similarity as sim
from peapod import pozitiv as poz
from peapod import plms
from peapod import visualize as viz
from peapod import benchmark as bm
from peapod import batchcorrection as bc
from peapod import fft
from peapod import mncm
import pandas as pd
from scipy import stats
import datetime


# These are the messier wrapper functions for doing more specific things

def batch_embed(profile_dict,model,padding=False):
    print("Embedding " + str(len(profile_dict.keys())) + " protein sequences...")
    tally = 0
    for key in profile_dict.keys():
        print(datetime.datetime.now())
        profile_dict[key].embed(model,padding=padding)
        tally+=1
        print(tally)


def global_alignment(profile0,profile1,gap_func):
    batch_corrected = bc.batch_correct([profile0,profile1])
    S = sim.shift(sim.enhance_signal(sim.minkowski(batch_corrected[0],batch_corrected[1])),-1)
    alignment, _ = aln.global_aln(S, gap_func)
    return alignment


def local_alignment(profile0,profile1,gap_func):
    batch_corrected = bc.batch_correct([profile0,profile1])
    S = sim.shift(sim.enhance_signal(sim.minkowski(batch_corrected[0],batch_corrected[1])),-1)
    alignment, _ = aln.local_aln(S, gap_func)
    return alignment


def locally_anchored_global_alignment(profile0,profile1,gap_func):
    batch_corrected = bc.batch_correct([profile0,profile1])
    S = sim.shift(sim.enhance_signal(sim.minkowski(batch_corrected[0],batch_corrected[1])),-1)
    alignment = aln.local_anchor_global_aln(S, gap_func)
    return alignment


def fft_anchored_global_alignment(profile0,profile1,gap_func):
    batch_corrected = bc.batch_correct([profile0,profile1])
    S = sim.shift(sim.enhance_signal(sim.minkowski(batch_corrected[0],batch_corrected[1])),-1)
    fft_anchors = fft.get_anchors(S,batch_corrected[0],batch_corrected[1])
    if fft_anchors == None:
        print("No anchors!")
        output, _ = aln.global_aln(S, gap_func)
    else:
        output = aln.anchored_global_aln(S, fft_anchors, gap_func)
    return output


def mncm_anchored_global_alignment(profile0,profile1):
    batch_corrected = bc.batch_correct([profile0,profile1])
    S = sim.shift(sim.enhance_signal(sim.minkowski(batch_corrected[0],batch_corrected[1])),-1)
    mncm_anchors = mncm.mncm(S) 
    output = aln.anchored_global_aln(S, mncm_anchors, gap_func)
    return output


def wmncm_anchored_global_alignment(profile0,profile1):
    batch_corrected = bc.batch_correct([profile0,profile1])
    S = sim.shift(sim.enhance_signal(sim.minkowski(batch_corrected[0],batch_corrected[1])),-1)
    wmncm_anchors = mncm.wmncm(S) 
    output = aln.anchored_global_aln(S, wmncm_anchors, gap_func)
    return output


def pairwise_cluster_alignment(profile0,profile1):
    batch_corrected = bc.batch_correct([profile0,profile1])
    S = sim.shift(sim.enhance_signal(sim.minkowski(batch_corrected[0],batch_corrected[1])),-1)
    alignment = aln.pairwise_clustering_aln(S,weighted=False)
    return alignment


def weighted_pairwise_cluster_alignment(profile0,profile1):
    batch_corrected = bc.batch_correct([profile0,profile1])
    S = sim.shift(sim.enhance_signal(sim.minkowski(batch_corrected[0],batch_corrected[1])),-1)
    alignment = aln.pairwise_clustering_aln(S,weighted=True)
    return alignment


