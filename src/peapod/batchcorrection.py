#!/usr/bin/env python
import numpy as np
from peapod import utilities as utils
from inmoose.pycombat import pycombat_norm

def batch_correct(profile_list,ref_batch = 0):
    """ Performs batch effect correction using an empirical Bayes framework as implemented in inmoose. 

        Parameters:
            profile_list (list of profile objects): a list of profiles to perform batch correction on, including the reference profile
            ref_batch (int): index of the reference profile (default 0)
            
        Returns:
            new_profiles (list of profile objects): a list of batch-corrected profiles
    """
    batch_assignments = []
    for i, profile in enumerate(profile_list):
        batch_assignments.extend([i for _ in range(profile.length)])
    concat_emb = np.concatenate([profile.embeddings.T for profile in profile_list],axis=1)
    #each sequence is a batch, each amino acid is a sample
    batch_corrected = pycombat_norm(concat_emb, batch_assignments, ref_batch = ref_batch)
    new_profiles = []
    for i, profile in enumerate(profile_list):
        new_profile = utils.profile(profile.deflines,profile.aaseqs)
        new_profile.embedding_override(batch_corrected[:, np.where(np.array(batch_assignments) == i)[0].tolist()].T)
        new_profiles.append(new_profile)
    return new_profiles
