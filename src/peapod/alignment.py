#!/usr/bin/env python
import numpy as np
import numba as nb
from peapod import utilities as utils
from peapod import mncm
from peapod import fft

#####################################################################################################################
############################################### Fill Scoring Matrices ###############################################
#####################################################################################################################

### Local ###
@nb.njit
def _profile_fill_matrix_local(S, gap_func):
    #fill a m x n x 6 matrix with the third dimension ordered as D, V, H, W, i, and j
    #D - score arriving from the diagonal
    #V - score arriving from a vertical gap
    #H - score arriving from a horizontal gap
    #W - maximum arriving score
    #i - i coordinate from which maximum score arrives
    #j - j coordinate from which maximum score arrives
    mn = S.shape
    m = mn[0]
    n = mn[1]
    #initialize m x n x 6 matrix and fill with zeroes
    matrix = np.zeros((m+1,n+1,6),dtype=np.float64)
    gap_candidates_rows = np.full((m, n), 1)
    gap_candidates_cols = np.full((m, n), 1)
    for i in range(m):
        for j in range(n):
            #calculate D and save source i, j
            previousW = matrix[i-1,j-1,3]
            matrix[i,j,0] = max(0,previousW+S[i,j])
            
            #calculate V and save source i, j of max
            #need to handle empty arrays:
            VpreviousDs = matrix[i,0:j,0]
            if len(VpreviousDs):
                VDmax_index = np.argmax(VpreviousDs * gap_candidates_rows[i,0:j])
                gap_candidates_rows[i,0:VDmax_index] = np.full((VDmax_index), 0)
                candidate_indices_rows = np.nonzero(gap_candidates_rows[i,0:j])
                Vscores = np.zeros(len(candidate_indices_rows[0]))
                for idx, k in enumerate(candidate_indices_rows[0]):
                    Vscores[idx] = matrix[i,k,0]-gap_func(j-k)
                Vmax_score_idx = np.argmax(Vscores)
                if Vscores[Vmax_score_idx] > 0:
                    matrix[i,j,1] = Vscores[Vmax_score_idx]
                    sourcej = candidate_indices_rows[0][Vmax_score_idx] 
                    # else statement not needed for assigning a value to sourcej 
                    # because with the use of argmax here, ties go to the diagonal, 
                    # thus sourcej will never be called if it hasn't been set. This 
                    # doesn't apply to global alignments because of the need for initial gap penalties

            #calculate H and save source i, j of max
            HpreviousDs = matrix[0:i,j,0]
            if len(HpreviousDs):
                HDmax_index = np.argmax(HpreviousDs * gap_candidates_cols[0:i,j])
                gap_candidates_cols[0:HDmax_index,j] = np.full((HDmax_index), 0)
                candidate_indices_cols = np.nonzero(gap_candidates_cols[0:i,j])
                Hscores = np.zeros(len(candidate_indices_cols[0]))
                for idx, k in enumerate(candidate_indices_cols[0]):
                    Hscores[idx] = matrix[k,j,0]-gap_func(i-k)
                Hmax_score_idx = np.argmax(Hscores)
                if Hscores[Hmax_score_idx] > 0:
                    matrix[i,j,2] = Hscores[Hmax_score_idx]
                    sourcei = candidate_indices_cols[0][Hmax_score_idx] 
                    # else statement not needed for assigning a value to sourcei
                    # because with the use of argmax here, ties go to the diagonal, 
                    # thus sourcei will never be called if it hasn't been set. This 
                    # doesn't apply to global alignments because of the need for initial gap penalties

            #take maximum of D, H, V and set W
            index_of_max = np.argmax(matrix[i,j,0:3])
            #document source i, j of chosen path for traceback
            matrix[i,j,3] = matrix[i,j,index_of_max]
            if index_of_max==0:
                matrix[i,j,4] = i-1
                matrix[i,j,5] = j-1
            elif index_of_max==1:
                matrix[i,j,4] = i
                matrix[i,j,5] = sourcej
            else:
                matrix[i,j,4] = sourcei
                matrix[i,j,5] = j
    return matrix


### Global ###
@nb.njit
def _profile_fill_matrix_global(S, gap_func, initial_score):
    """ Take as input a similarity matrix (matrix.array) and a gap penalty function and fills a 
        global sequence alignment matrix as per Mott 1999 (an extension of Smith-Waterman 1981).

        :param S: similarity matrix
        :param gap_func: monotonically increasing gap penalty function

        :type S: numpy array
        :type gap_func: python function
    """
    # adding weights will require passing profiles as arguments
    #fill a m x n x 6 matrix with the third dimension ordered as D, V, H, W, i, and j
    #D - score arriving from the diagonal
    #V - score arriving from a vertical gap
    #H - score arriving from a horizontal gap
    #W - maximum arriving score
    #i - i coordinate from which maximum score arrives
    #j - j coordinate from which maximum score arrives
    mn = S.shape
    m = mn[0]
    n = mn[1]
    #initialize m x n x 6 matrix and fill with zeroes
    matrix = np.zeros((m+1,n+1,6),dtype=np.float64)
    matrix[0:m,-1,3] = (gap_func(np.arange(1,m+1, dtype=np.float64))*-1) + initial_score
    matrix[-1,0:n,3] = (gap_func(np.arange(1,n+1, dtype=np.float64))*-1) + initial_score
    matrix[-1,-1,3] = initial_score
    gap_candidates_rows = np.full((m, n), 1)
    gap_candidates_cols = np.full((m, n), 1)
    for i in range(m):
        for j in range(n):
            #calculate D and save source i, j
            previousW = matrix[i-1,j-1,3]
            matrix[i,j,0] = previousW+S[i,j]
            
            #calculate V and save source i, j of max
            #need to handle empty arrays:
            VpreviousDs = matrix[i,0:j,0]
            if len(VpreviousDs):
                Vscores = np.zeros(j)
                for k in range(j):
                    Vscores[k] = matrix[i,k,0]-(gap_func(j-k))
                Vmax_score_idx = np.argmax(Vscores)
                matrix[i,j,1] = Vscores[Vmax_score_idx]
                sourcej = Vmax_score_idx
            else:
                sourcej = -1

            #calculate H and save source i, j of max
            HpreviousDs = matrix[0:i,j,0]
            if len(HpreviousDs):
                Hscores = np.zeros(i)
                for k in range(i):
                    Hscores[k] = matrix[k,j,0]-(gap_func(i-k))
                Hmax_score_idx = np.argmax(Hscores)
                matrix[i,j,2] = Hscores[Hmax_score_idx]
                sourcei = Hmax_score_idx
            else:
                sourcei = -1


            #take maximum of D, H, V and set W
            index_of_max = np.argmax(matrix[i,j,0:3])
            #document source i, j of chosen path for traceback
            matrix[i,j,3] = matrix[i,j,index_of_max]
            if index_of_max==0:
                matrix[i,j,4] = i-1
                matrix[i,j,5] = j-1
            elif index_of_max==1:
                matrix[i,j,4] = i
                matrix[i,j,5] = sourcej
            else:
                matrix[i,j,4] = sourcei
                matrix[i,j,5] = j
    return matrix





#####################################################################################################################
################################################ Traceback functions ################################################
#####################################################################################################################

@nb.njit
def _mark_duplicate_positions_fast(positions):
    copy = positions.copy()
    for index in range(1,positions.shape[0]):
        if (positions[index,0]==positions[index-1,0]) or (positions[index,1]==positions[index-1,1]):
            copy[index,0] = -1
            copy[index,1] = -1
    return copy

@nb.njit
def _loop_nonneg_positions(positions):
    nonneg = 0
    for index in range(positions.shape[0]):
        if (positions[index,0]>=0) & (positions[index,1]>=0):
            nonneg+=1
    nonnegpositions = np.zeros((nonneg,2),dtype=np.int64)
    nonnegindex = 0
    for index in range(positions.shape[0]):
        if (positions[index,0]>=0) & (positions[index,1]>=0):
            nonnegpositions[nonnegindex,0] = positions[index,0]
            nonnegpositions[nonnegindex,1] = positions[index,1]
            nonnegindex+=1
    return nonnegpositions


### Local ###
@nb.njit
def _fast_local_traceback(matrix,start,maxdim):
    positions = np.zeros((maxdim,2),dtype=np.int64)-1
    i = int(start[0])
    j = int(start[1])
    #print('Start: '+str(i)+','+str(j))
    W = matrix[i,j,3]
    aln_score = W
    positions_index = 0
    while (W > 0):
        #print(str(i)+','+str(j))
        positions[positions_index,0] = i
        positions[positions_index,1] = j
        pi = int(matrix[i,j,4])
        pj = int(matrix[i,j,5])
        i = pi
        j = pj
        W = matrix[i,j,3]
        #print(W)
        positions_index += 1
    return aln_score, positions
    
def _fast_local_traceback_wrapper(matrix):
    Wslice = matrix[:,:,3]
    dims = Wslice.shape
    start = np.unravel_index(np.argmax(Wslice),dims)
    maxdim = dims[0]+dims[1]
    aln_score, positions = _fast_local_traceback(matrix,start,maxdim)
    marked_duplicate_positions = _mark_duplicate_positions_fast(_loop_nonneg_positions(positions)[::-1])
    return aln_score, _loop_nonneg_positions(marked_duplicate_positions)#.transpose().tolist()

def local_score_without_traceback(matrix):
    Wslice = matrix[0:matrix.shape[0],0:matrix.shape[1],3]
    start = np.unravel_index(np.argmax(Wslice),Wslice.shape)
    return matrix[int(start[0]),int(start[1]),3]



### Global ###
@nb.njit
def _fast_global_traceback(matrix,start,maxdim):
    positions = np.zeros((maxdim,2),dtype=np.int64)-1
    i = int(start[0])
    j = int(start[1])
    W = matrix[i,j,3]
    aln_score = W
    positions_index = 0
    while (i >= 0) and (j >= 0):
        #print(str(i)+','+str(j))
        positions[positions_index,0] = i
        positions[positions_index,1] = j
        pi = int(matrix[i,j,4])
        pj = int(matrix[i,j,5])
        i = pi
        j = pj
        W = matrix[i,j,3]
        positions_index += 1
    return aln_score, positions
    
def _fast_global_traceback_wrapper(matrix):
    Wslice = matrix[:,:,3]
    dims = Wslice.shape
    maxdim = dims[0]+dims[1]
    aln_score, positions = _fast_global_traceback(matrix,(dims[0]-2,dims[1]-2),maxdim)
    marked_duplicate_positions = _mark_duplicate_positions_fast(_loop_nonneg_positions(positions)[::-1])
    return aln_score, _loop_nonneg_positions(marked_duplicate_positions)#.transpose().tolist()







#####################################################################################################################
############################################## Putting it all together ##############################################
#####################################################################################################################

def local_aln(S, gap_func):
    """ Given a similarity matrix and a gap penalty function, performs a 
        local alignment as per Mott 1999.

        Parameters:
            S (matrix object): similarity matrix
            gap_func (function): monotonic gap function, njitted (numba)

        Returns:
            positions object
    """
    sim = S.array
    matrix = _profile_fill_matrix_local(sim, gap_func)
    aln_score, alignment = _fast_local_traceback_wrapper(matrix)
    return utils.positions(S.deflines0,S.deflines1,alignment,score=aln_score), utils.matrix(S.deflines0,S.deflines1,matrix[0:-1,0:-1,3])


def global_aln(S, gap_func, initial_score=0):
    """ Given a similarity matrix and a gap penalty function, performs a 
        global alignment as per Mott 1999.

        Parameters:
            S (matrix object): similarity matrix
            gap_func (function): monotonic gap function, njitted (numba)
            initial_score (float): only used for anchored alignments, where running total score should affect sub-alignment (default 0)

        Returns:
            positions object
    """
    sim = S.array
    matrix = _profile_fill_matrix_global(sim, gap_func, initial_score = initial_score)
    aln_score, alignment = _fast_global_traceback_wrapper(matrix)
    return utils.positions(S.deflines0,S.deflines1,alignment,score=aln_score), utils.matrix(S.deflines0,S.deflines1,matrix[0:-1,0:-1,3])




#####################################################################################################################
############################################### Multistage alignments ###############################################
#####################################################################################################################

def _slice_similarity(S,anchor,next_anchor):
    return utils.matrix(S.deflines0,S.deflines1,S.array[int(anchor[0])+1:int(next_anchor[0]),int(anchor[1])+1:int(next_anchor[1])])

def _eval_between_anchors(S,anchor,next_anchor,gap_func,initial_score=0):
    gap = next_anchor-anchor
    if (gap != np.array([1,1])).all():
        if (gap == np.array([1,1])).any():
            gap_dim = np.where(gap == np.array([1,1]))[0][0]
            running_score = initial_score-gap_func(next_anchor[gap_dim]-anchor[gap_dim]-1)
            new_coords = np.array([])
        else:
            split_alignment, _ = global_aln(_slice_similarity(S,anchor,next_anchor), gap_func, initial_score=initial_score)
            running_score = split_alignment.score
            new_coords = split_alignment.coordinates + np.array(anchor) + np.array([1,1])
    else:
        running_score=initial_score
        new_coords = np.array([])
    return running_score, new_coords


def local_anchor_global_aln(S, gap_func):
    """ Given a similarity matrix and a gap penalty function, performs a 
        global alignment anchored by a local alignment.

        Parameters:
            S (matrix object): similarity matrix
            gap_func (function): monotonic gap function, njitted (numba)

        Returns:
            positions object
    """
    local_alignment, local_matrix = local_aln(S, gap_func)
    anchors = np.zeros((4,2))
    anchors[0,:] = np.array([-1,-1])
    anchors[1,:] = local_alignment.coordinates[0]
    anchors[2,:] = local_alignment.coordinates[-1]
    anchors[3,:] = np.array(S.array.shape)-1
    splits_aligned_coords = [local_alignment.coordinates]
    running_score = 0
    #nterminal
    nterm_score, nterm_coords = _eval_between_anchors(S,anchors[0,:],anchors[1,:],gap_func,initial_score=running_score)
    running_score += nterm_score
    if nterm_coords.shape[0] != 0:
        splits_aligned_coords.append(nterm_coords)

    #include local score
    running_score += local_alignment.score
    
    #cterminal
    cterm_score, cterm_coords = _eval_between_anchors(S,anchors[2,:],anchors[3,:],gap_func,initial_score=running_score)
    running_score += cterm_score
    if cterm_coords.shape[0] != 0:
        splits_aligned_coords.append(cterm_coords)
    
    final_coords = np.concatenate(splits_aligned_coords)
    sorted_coords = final_coords[final_coords[:, 0].argsort()]
    return utils.positions(S.deflines0,S.deflines1,sorted_coords,score=running_score)




def anchored_global_aln(S, positions, gap_func):
    """ Given a similarity matrix a set of anchors (positions object), and a 
        gap penalty function, performs an anchored global alignment.

        Parameters:
            S (matrix object): similarity matrix
            positions (positions object): a set of anchor coordinates
            gap_func (function): monotonic gap function, njitted (numba)

        Returns:
            positions object
    """
    if (S.deflines0 != positions.deflines0) or (S.deflines1 != positions.deflines1):
        raise Exception('One or both dimensions of the similarity matrix you supplied do not match the deflines of the positions.')
    anchors = np.zeros((positions.coordinates.shape[0]+2,3))
    anchors[0,0:2] = np.array([-1,-1])
    anchors[-1,0:2] = np.array(S.array.shape)
    anchors[1:-1,0:2] = positions.coordinates[positions.coordinates[:, 0].argsort()] #just to make sure it's sorted
    for idx in range(1,anchors.shape[0]-1):
        anchors[idx,2] = S.array[int(anchors[idx,0]),int(anchors[idx,1])]
    splits_aligned_coords = [positions.coordinates]
    running_score = 0
    for idx in range(anchors.shape[0]-1):
        anchor = anchors[idx,0:2]
        next_anchor = anchors[idx+1,0:2]
        running_score, new_coords = _eval_between_anchors(S,anchor,next_anchor,gap_func,initial_score=running_score)
        running_score += anchors[idx+1,2]
        if new_coords.shape[0] != 0:
            splits_aligned_coords.append(new_coords)
    final_coords = np.concatenate(splits_aligned_coords)
    sorted_coords = final_coords[final_coords[:, 0].argsort()]
    return utils.positions(positions.deflines0,positions.deflines1,sorted_coords,score=running_score)





def anchored_local_aln(S,positions,gap_func):
    """ Given a similarity matrix a set of anchors (positions object), and a 
        gap penalty function, performs an anchored local alignment.

        Parameters:
            S (matrix object): similarity matrix
            positions (positions object): a set of anchor coordinates
            gap_func (function): monotonic gap function, njitted (numba)

        Returns:
            positions object
    """
    new_S_array = S.array.copy()
    for idx in range(positions.coordinates.shape[0]):
        anchor = positions.coordinates[idx,:]
        new_S_array[0:int(anchor[0]),int(anchor[1])+1:S.array.shape[1]] = -np.inf
        new_S_array[int(anchor[0])+1:S.array.shape[0],0:int(anchor[1])] = -np.inf
    return local_aln(utils.matrix(S.deflines0,S.deflines1,new_S_array), gap_func)


def _recursive_mncm(S,weighted=False,score_threshold=0):
    if weighted==True:
        positions = mncm.wmncm(S,score_threshold=score_threshold)
    else:
        positions = mncm.mncm(S,score_threshold=score_threshold)
    if positions.coordinates.shape[0]==0:
        sorted_coords = np.array([])
    else:
        anchors = np.zeros((positions.coordinates.shape[0]+2,positions.coordinates.shape[1]))
        anchors[0,:] = np.array([-1,-1])
        anchors[-1,:] = np.array(S.array.shape)
        anchors[1:-1,:] = positions.coordinates[positions.coordinates[:, 0].argsort()] #just to make sure it's sorted
        sub_aligned_coords = [positions.coordinates]
        for idx in range(anchors.shape[0]-1):
            anchor = anchors[idx,:]
            next_anchor = anchors[idx+1,:]
            gap = next_anchor-anchor
            if (gap != np.array([1,1])).all():
                if not (gap == np.array([1,1])).any():
                    sub_alignment = _recursive_mncm(_slice_similarity(S,anchor,next_anchor),weighted=weighted,score_threshold=score_threshold)
                    if sub_alignment.coordinates.shape[0] != 0:
                        sub_aligned_coords.append(sub_alignment.coordinates + np.array(anchor) + np.array([1,1]))
        final_coords = np.concatenate(sub_aligned_coords)
        sorted_coords = final_coords[final_coords[:, 0].argsort()]
    return utils.positions(S.deflines0,S.deflines1,sorted_coords)

def pairwise_clustering_aln(S,weighted=False,score_threshold=0):
    """ Given a similarity matrix and a minimum score threshold, performs a 
        recursive pairwise-best-hit and (W)MNCM alignment (which isinherently
        global and produces a gapless score).
        Inspired by vcMSA (McWhite et al., 2022).

        Parameters:
            S (matrix object): similarity matrix
            weighted (boolean): if True, perform WMNCM instead of MNCM (default False)
            score_threshold (float): minimum score between positions to include in the alignment (default 0)

        Returns:
            positions object
    """
    #inherently global, produces a gapless score
    positions = _recursive_mncm(S,weighted=weighted,score_threshold=score_threshold)
    score = 0
    for idx in range(positions.coordinates.shape[0]):
        coord = positions.coordinates[idx,:]
        score += S.array[int(coord[0]),int(coord[1])]
    return utils.positions(positions.deflines0,positions.deflines1,positions.coordinates,score=score)


def fft_anchored_pairwise_clustering_aln(S,
                                         profile0,
                                         profile1,
                                         gap_func,
                                         window=10,
                                         fft_score_threshold=5,
                                         max_span=50,
                                         segment_count=20,
                                         anchors_per_segment=2,
                                         weighted=False,
                                         clustering_score_threshold=0):
    """ Given a similarity matrix and a minimum score threshold, performs a 
        recursive pairwise-best-hit and (W)MNCM alignment (which isinherently
        global and produces a gapless score), but the initial anchors are 
        computed using a Fast Fourier Transform approach inspired by MAFFT.
        Inspired by vcMSA (McWhite et al., 2022).

        Parameters:
            S (matrix object): similarity matrix
            profile0 (profile object): first profile to align
            profile1 (profile object): second profile to align
            gap_func (function): monotonic gap function, njitted (numba)
            window (integer): sliding window size on which to evaluate highly correlated diagonals during FFT (default 10)
            fft_score_threshold (float): average score above which a window is considered similar during FFT (default 5)
            max_span (integer): maximum length of a single similar segment, above which they are split into smaller segments (default 50)
            segment_count (integer): number of correlated diagonals to search for similar segments during FFT (default 20)
            anchors_per_segment (integer): number of top scoring positions within a chosen similar segment to use as alignment anchors (default 2)
            weighted (boolean): if True, perform WMNCM instead of MNCM (default False)
            clustering_score_threshold (float): minimum score between positions to include in the alignment (default 0)

        Returns:
            positions object
    """
    if (profile0.deflines != S.deflines0) or (profile1.deflines != S.deflines1):
        raise Exception('One or both profiles you supplied do not match the deflines of the similarity matrix.')
    fft_anchors = fft.get_anchors(S,profile0,profile1,gap_func,window,fft_score_threshold,max_span,segment_count,anchors_per_segment)
    if fft_anchors == None:
        print("No anchors!")
        sorted_coords = _recursive_mncm(S,weighted=weighted,score_threshold=clustering_score_threshold).coordinates
    else:
        anchors = np.zeros((fft_anchors.coordinates.shape[0]+2,fft_anchors.coordinates.shape[1]))
        anchors[0,:] = np.array([-1,-1])
        anchors[-1,:] = np.array(S.array.shape)
        anchors[1:-1,:] = fft_anchors.coordinates[fft_anchors.coordinates[:, 0].argsort()] #just to make sure it's sorted
        sub_aligned_coords = [fft_anchors.coordinates]
        for idx in range(anchors.shape[0]-1):
            anchor = anchors[idx,:]
            next_anchor = anchors[idx+1,:]
            gap = next_anchor-anchor
            if (gap != np.array([1,1])).all():
                if not (gap == np.array([1,1])).any():
                    sub_alignment = _recursive_mncm(_slice_similarity(S,anchor,next_anchor),weighted=weighted,score_threshold=clustering_score_threshold)
                    if sub_alignment.coordinates.shape[0] != 0:
                        sub_aligned_coords.append(sub_alignment.coordinates + np.array(anchor) + np.array([1,1]))
        final_coords = np.concatenate(sub_aligned_coords)
        sorted_coords = final_coords[final_coords[:, 0].argsort()]
    score = 0
    for idx in range(sorted_coords.shape[0]):
        coord = sorted_coords[idx,:]
        score += S.array[int(coord[0]),int(coord[1])]
    return utils.positions(S.deflines0,S.deflines1,sorted_coords,score=score)



#####################################################################################################################
###################################### Compare alignment to substitution stats ######################################
#####################################################################################################################

def percent_positive(profile0, profile1, positions, substitution_matrix_dict):
    """ Given two profiles, the aligned positions between them, and a 
        substitution matrix, returns the percent of aligned positions with 
        positive scores under the provided substitution matrix.
        
        Parameters:
            profile0 (profile object): first profile
            profile1 (profile object): second profile
            positions (positions object): aligned positions
            substitution_matrix_dict (numba typed.Dict): dictionary of substitution scores
            
        Returns:
            percent_positive (int): percentage of aligned positions with positive scores
    """
    if (len(positions.deflines0) != 1) or (len(positions.deflines1) != 1):
        raise Exception("Either one or both of the profiles you've aligned are not single sequences. Percent positive scores are only computed for pairwise comparisons.")
    if (profile0.deflines != positions.deflines0) or (profile1.deflines != positions.deflines1):
        raise Exception('One or both profiles you supplied do not match the deflines of the positions.')
    aaseq0list = list(profile0.aaseqs[0])
    aaseq1list = list(profile1.aaseqs[0])
    positive_tally = 0
    for index in range(positions.coordinates.shape[0]):
        aa0 = aaseq0list[positions.coordinates[index,0].astype(np.int64)]
        aa1 = aaseq1list[positions.coordinates[index,1].astype(np.int64)]
        if float(substitution_matrix_dict[aa0+aa1]) > 0:
            positive_tally += 1
    return positive_tally/positions.coordinates.shape[0]


def percent_identity(profile0, profile1, positions):
    """ Given two profiles and the aligned positions between them, 
        returns the percent of aligned positions with identical residues.
        
        Parameters:
            profile0 (profile object): first profile
            profile1 (profile object): second profile
            positions (positions object): aligned positions
            
        Returns:
            percent_identical (int): percentage of aligned positions that are identical
    """
    if (len(positions.deflines0) != 1) or (len(positions.deflines1) != 1):
        raise Exception("Either one or both of the profiles you've aligned are not single sequences. Percent identity scores are only computed for pairwise comparisons.")
    if (profile0.deflines != positions.deflines0) or (profile1.deflines != positions.deflines1):
        raise Exception('One or both profiles you supplied do not match the deflines of the positions.')
    aaseq0list = list(profile0.aaseqs[0])
    aaseq1list = list(profile1.aaseqs[0])
    identity_tally = 0
    for index in range(len(positions.coordinates)):
        aa0 = aaseq0list[positions.coordinates[index,0].astype(np.int64)]
        aa1 = aaseq1list[positions.coordinates[index,1].astype(np.int64)]
        if aa0==aa1:
            identity_tally += 1
    return identity_tally/positions.coordinates.shape[0]

