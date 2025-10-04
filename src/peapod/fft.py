#!/usr/bin/env python
import numpy as np
import numba as nb
import scipy
import math
from peapod import utilities as utils


def _k_range(profile0,profile1,window):
    kmax = profile0.length - window
    kmin = -(profile1.length - window)
    return kmin, kmax

@nb.njit
def _zero_pad(profile0_length,profile0_embeddings,profile1_length,profile1_embeddings,emb_dim,kmin,kmax):
    longer = max(profile0_length,profile1_length)
    padded_profile0 = np.zeros((longer+abs(kmin)+kmax,emb_dim))
    padded_profile1 = padded_profile0.copy()
    padded_profile0[abs(kmin):abs(kmin)+profile0_length,:] = profile0_embeddings
    padded_profile1[0:profile1_length,:] = profile1_embeddings
    return padded_profile0, padded_profile1

def _zero_pad_wrapper(profile0,profile1,kmin,kmax):
    profile0_length = profile0.length
    profile1_length = profile1.length
    profile0_emb = profile0.embeddings
    profile1_emb = profile1.embeddings
    emb_dim = profile0.embeddings.shape[1]
    return _zero_pad(profile0_length,profile0_emb,profile1_length,profile1_emb,emb_dim,kmin,kmax)

def _discrete_corr(vector0,vector1):
    return scipy.fft.ifft(np.multiply(scipy.fft.fft(vector0),np.conj(scipy.fft.fft(vector1)))).real

def _calc_correlations(profile0,profile1,window):
    emb_dim = profile0.embeddings.shape[1]
    kmin, kmax = _k_range(profile0,profile1,window)
    padded_emb0, padded_emb1 = _zero_pad_wrapper(profile0,profile1,kmin,kmax)
    correlations = np.zeros(padded_emb0.shape)
    for i in range(emb_dim):
        correlations[:,i] = _discrete_corr(padded_emb0[:,i], padded_emb1[:,i])
    return correlations.sum(axis=1), kmin

@nb.njit
def _pick_topn_offsets(k_scores,kmin,topn):
    if topn < 1:
        raise Exception("You have asked for less than 1 top scoring offset. We're not really sure what to make of that.")
    top_ind = np.argpartition(k_scores, -topn)[-topn:]
    return top_ind+kmin

def _get_spans(boolean_vector, max_span):
    #spans ends are exclusive!
    a = np.pad(boolean_vector, (0, 1))
    b = np.pad(boolean_vector, (1, 0))
    starts = np.array(np.nonzero(a & ~b))
    ends = np.array(np.nonzero(~a & b))
    too_large = (ends-starts) > max_span
    #print(np.where(too_large)[0])
    small_indices = np.where(np.logical_not(too_large))[0]
    spans = np.concatenate([starts,ends]).T
    small_spans = spans[small_indices,:]
    if (too_large > 0).any():
        new_spans = []
        new_spans.append(small_spans)
        for idx in np.where(too_large)[0]:
            length = spans[idx,1]-spans[idx,0]
            subspan_length = math.floor(length/math.ceil(length/max_span))
            subspan_no = (length%subspan_length)+1
            split_span = np.array([[a.min(),a.max()+1] for a in np.array_split(range(spans[idx,0],spans[idx,1]),subspan_no)])
            new_spans.append(split_span)
        concat_spans = np.concatenate(new_spans)
    else:
        concat_spans = small_spans
    return concat_spans

def _find_similar_segments(S, k_list, window, threshold, max_span):
    similar_segments = []
    for k in k_list:
        diagonal = np.diag(S.array,k=-int(k))
        above_threshold = np.zeros(diagonal.shape[0],dtype='int')
        for i in range(diagonal.shape[0]+1-window):
            if diagonal[i:i+window].mean() >= threshold:
                above_threshold[i:i+window] = 1
        if any(above_threshold > 0):
            spans = _get_spans(above_threshold,max_span)
            scored_spans = np.concatenate([spans,np.zeros((spans.shape[0],2))],axis=1)
            for j in range(spans.shape[0]):
                scored_spans[j,2] = diagonal[int(scored_spans[j,0]):int(scored_spans[j,1])].sum()
                scored_spans[j,3] = k
            similar_segments.append(scored_spans)
    # print("# of similar segments: "+str(len(similar_segments)))
    if len(similar_segments)==0:
        output = np.array([])
    elif len(similar_segments)==1:
        output = similar_segments[0]
    else:
        output = np.concatenate(similar_segments)
    return output

#@nb.njit
def _segments_to_ranked_bounds(sim_segments):
    n = sim_segments.shape[0]
    bounds = np.zeros((n,10))
    bounds[:,0:4] = sim_segments
    for nth in range(n):
        k = bounds[nth,3]
        if k>0:
            start_dim0 = sim_segments[nth,0]+k
            start_dim1 = sim_segments[nth,0]
            end_dim0 = sim_segments[nth,1]+k
            end_dim1 = sim_segments[nth,1]
        else:
            start_dim0 = sim_segments[nth,0]
            start_dim1 = sim_segments[nth,0]+abs(k)
            end_dim0 = sim_segments[nth,1]
            end_dim1 = sim_segments[nth,1]+abs(k)
        bounds[nth,4] = start_dim0 #set start dim0
        bounds[nth,5] = start_dim1 #set start dim1
        bounds[nth,6] = end_dim0-1 #set end dim0
        bounds[nth,7] = end_dim1-1 #set end dim1
    startdim0_ranked = bounds[bounds[:, 4].argsort()]
    startdim0_ranked[:,8] = np.array(range(n))
    final_ranked = startdim0_ranked[startdim0_ranked[:, 5].argsort()]
    final_ranked[:,9] = np.array(range(n))
    # print("ranked bounds:")
    # print(final_ranked)
    return final_ranked

#@nb.njit
def _ranked_bounds_to_matrix(ranked_bounds):
    #double check functionality
    n = ranked_bounds.shape[0]
    matrix = np.zeros((n,n,10))
    for nth in range(n):
        i = ranked_bounds[nth,8]
        j = ranked_bounds[nth,9]
        # print("i,j: " +str(i)+","+str(j))
        # print(ranked_bounds[nth,:].squeeze())
        matrix[int(i),int(j),:] = ranked_bounds[nth,:].squeeze()
    return matrix

@nb.njit
def _fill_corr_span_matrix_local(matrix,gap_func):
    n = matrix.shape[0]
    scoring = np.zeros((n+1,n+1,3),dtype=np.float32)
    for i in range(n):
        for j in range(n):
            gap_source_i = -1
            gap_source_j = -1
            max_gapped_score = 0
            # print('For position '+str(i)+','+str(j)+'...')
            for x in range(i):
                for y in range(j):
                    # if (matrix[x,y,2] != 0) and (matrix[x,y,6]<matrix[i,j,4]) and (matrix[x,y,7]<matrix[i,j,5]):
                    #     print('The span starting at '+str(x)+','+str(y)+' is a candidate because:')
                    #     print(str(matrix[x,y,6])+'<'+str(matrix[i,j,4]))
                    #     print(str(matrix[x,y,7])+'<'+str(matrix[i,j,5]))
                    #if previous cell's score is not zero, and doesn't overlap, calculate the gaps and the score from gapping
                    if (matrix[x,y,2] != 0) and (matrix[x,y,6]<matrix[i,j,4]) and (matrix[x,y,7]<matrix[i,j,5]):
                        if matrix[i,j,4]-matrix[x,y,6]-1 > 0:
                            dim0_gap = gap_func(matrix[i,j,4]-matrix[x,y,6]-1)
                        else:
                            dim0_gap = 0
                        if matrix[i,j,5]-matrix[x,y,7]-1 > 0:
                            dim1_gap = gap_func(matrix[i,j,5]-matrix[x,y,7]-1)
                        else:
                            dim1_gap = 0
                        gap_score = scoring[x,y,0] + matrix[i,j,2] - dim0_gap - dim1_gap
                        #if the gapping score is greater than the max gapping score, update the max gapping score and source
                        if gap_score>max_gapped_score:
                            max_gapped_score = gap_score
                            gap_source_i = x
                            gap_source_j = y
            if max_gapped_score>matrix[i,j,2]:
                source_i = gap_source_i
                source_j = gap_source_j
                score = max_gapped_score
            else:
                source_i = -1
                source_j = -1
                score = matrix[i,j,2]
            scoring[i,j,0] = score
            scoring[i,j,1] = source_i
            scoring[i,j,2] = source_j
    return scoring

def _segment_traceback_local(matrix,scored_matrix):
    Wslice = scored_matrix[:,:,0]
    dims = Wslice.shape
    start = np.unravel_index(np.argmax(Wslice),dims)
    chosen_segments = []
    i = int(start[0])
    j = int(start[1])
    W = scored_matrix[i,j,0]
    #while (i >= 0) and (j >= 0):
    while (W > 0):
        chosen_segments.append(matrix[i,j,:][:,np.newaxis])
        nexti = int(scored_matrix[i,j,1])
        nextj = int(scored_matrix[i,j,2])
        i = nexti
        j = nextj
        W = scored_matrix[i,j,0]
    if len(chosen_segments)>0:
        output = np.concatenate(chosen_segments,axis=1).T
    else:
        output = np.array([])
    return output

def _similar_segments_to_coords(similar_segments):
    coordinates = []
    for i in range(similar_segments.shape[0]):
        dim0_vals = np.array(range(int(similar_segments[i,0]),int(similar_segments[i,1])))[:,np.newaxis]
        k = int(similar_segments[i,3])
        if k>0:
            coordinates.append(np.concatenate([dim0_vals+int(similar_segments[i,3]),dim0_vals],axis=1))
        else:
            coordinates.append(np.concatenate([dim0_vals,dim0_vals+abs(int(similar_segments[i,3]))],axis=1))
    return np.concatenate(coordinates)

def _segments_to_topn_anchors(S,segments,topn):
    if topn < 1:
        raise Exception('You have asked for less than 1 top hit from each anchor.')
    anchors = []
    for segment in segments:
        k = segment[3]
        diag_start = int(segment[0])
        diag_end = int(segment[1])
        segment_scores = np.diag(S.array,k=-int(k))[diag_start:diag_end]
        topn_diag_idx = np.argpartition(segment_scores, -topn)[-topn:]+diag_start
        if k>0:
            dim0s = topn_diag_idx+k
            dim1s = topn_diag_idx
        else:
            dim0s = topn_diag_idx
            dim1s = topn_diag_idx+abs(k)
        anchors.append(np.concatenate([dim0s[:,np.newaxis],dim1s[:,np.newaxis]],axis=1))
    concat_anchors = np.concatenate(anchors,axis=0)
    total_score = 0
    for anchor in concat_anchors:
        total_score+=S.array[int(anchor[0]),int(anchor[1])]
    return utils.positions(S.deflines0,S.deflines1,concat_anchors,score=total_score)




def correlations_table(profile0,profile1,window):
    """ Computes the correlations of different offsets of two profiles. Used in conjunction
        with plot_offset_correlations() from peapod's visualize module.

        Parameters:
            profile0 (profile object): first profile
            profile1 (profile object): second profile
            window (int): window size (included because window size of downstream methods affects the offsets to check)

        Returns:
            table (numpy array): array of offsets and their correlations
    """
    k_scores, kmin = _calc_correlations(profile0,profile1,window)
    table = np.zeros((k_scores.shape[0],2))
    table[:,0] = k_scores
    table[:,1] = np.array(range(k_scores.shape[0]))+kmin
    return table


def get_all_corr_segments(S,profile0,profile1,window,score_threshold,max_span,segment_count):
    """ Returns correlated segments between two profiles as a positions object for plotting purposes.

        Parameters:
            S (matrix object): similarity matrix
            profile0 (profile object): first profile to align
            profile1 (profile object): second profile to align
            window (int): length of sliding window size with which to evaluate scores of similarity matrix diagonals
            score_threshold (float): minimum average score for segments to be considered similar
            max_span (int): maximum length of a similar segment, above which they are split into smaller segments
            segment_count (int): number of top-correlated diagonals to evaluate

        Returns:
            positions (positions object): positions in correlated segments
    """
    if (S.deflines0 != profile0.deflines) or (S.deflines1 != profile1.deflines):
        raise Exception('One or both dimensions of the similarity matrix you supplied do not match the deflines of the profiles.')
    k_scores, kmin = _calc_correlations(profile0,profile1,window)
    k_list = _pick_topn_offsets(k_scores,kmin,segment_count)
    sim_segments = _find_similar_segments(S,k_list,window,score_threshold,max_span)
    if sim_segments.size == 0:
        output = None
    else:
        output = utils.positions(profile0.deflines,profile1.deflines,_similar_segments_to_coords(sim_segments))
    return output

def get_chosen_path_segments(S,profile0,profile1,gap_func,window,score_threshold,max_span,segment_count):
    """ Returns correlated segments between two profiles as a positions object for plotting purposes.

        Parameters:
            S (matrix object): similarity matrix
            profile0 (profile object): first profile to align
            profile1 (profile object): second profile to align
            gap_func (numba jitted function): gap function
            window (int): length of sliding window size with which to evaluate scores of similarity matrix diagonals
            score_threshold (float): minimum average score for segments to be considered similar
            max_span (int): maximum length of a similar segment, above which they are split into smaller segments
            segment_count (int): number of top-correlated diagonals to evaluate

        Returns:
            positions (positions object): positions in correlated segments in chosen path
    """
    if (S.deflines0 != profile0.deflines) or (S.deflines1 != profile1.deflines):
        raise Exception('One or both dimensions of the similarity matrix you supplied do not match the deflines of the profiles.')
    k_scores, kmin = _calc_correlations(profile0,profile1,window)
    k_list = _pick_topn_offsets(k_scores,kmin,segment_count)
    sim_segments = _find_similar_segments(S,k_list,window,score_threshold,max_span)
    # print("similar segments:")
    # print(sim_segments)
    # print(sim_segments.size)
    if sim_segments.size == 0:
        output = None
    else:
        segment_matrix = _ranked_bounds_to_matrix(_segments_to_ranked_bounds(sim_segments))
        scored_segment_matrix = _fill_corr_span_matrix_local(segment_matrix,gap_func)
        # print("scored segment matrix")
        # print(scored_segment_matrix)
        chosen_segments = _segment_traceback_local(segment_matrix,scored_segment_matrix)
        output = utils.positions(profile0.deflines,profile1.deflines,_similar_segments_to_coords(chosen_segments))
    return output

def get_anchors(S,profile0,profile1,gap_func,window=10,score_threshold=5,max_span=50,segment_count=20,anchors_per_segment=2):
    """ Returns correlated segments between two profiles as a positions object for plotting purposes.

        Parameters:
            S (matrix object): similarity matrix
            profile0 (profile object): first profile to align
            profile1 (profile object): second profile to align
            gap_func (numba jitted function): gap function
            window (int): length of sliding window size with which to evaluate scores of similarity matrix diagonals (default 10)
            score_threshold (float): minimum average score for segments to be considered similar (default 5)
            max_span (int): maximum length of a similar segment, above which they are split into smaller segments (default 50)
            segment_count (int): number of top-correlated diagonals to evaluate (default 20)
            anchors_per_segment (int): number of anchors per similar segment to report (default 2)

        Returns:
            positions (positions object): fft-derived anchors
    """
    if (S.deflines0 != profile0.deflines) or (S.deflines1 != profile1.deflines):
        raise Exception('One or both dimensions of the similarity matrix you supplied do not match the deflines of the profiles.')
    k_scores, kmin = _calc_correlations(profile0,profile1,window)
    k_list = _pick_topn_offsets(k_scores,kmin,segment_count)
    sim_segments = _find_similar_segments(S,k_list,window,score_threshold,max_span)
    if sim_segments.size == 0:
        output = None
    else:
        segment_matrix = _ranked_bounds_to_matrix(_segments_to_ranked_bounds(sim_segments))
        scored_segment_matrix = _fill_corr_span_matrix_local(segment_matrix,gap_func)
        chosen_segments = _segment_traceback_local(segment_matrix,scored_segment_matrix)
        output = _segments_to_topn_anchors(S,chosen_segments,anchors_per_segment)
    return output
