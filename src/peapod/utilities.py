#!/usr/bin/env python
import os.path
import pickle
from itertools import islice
from itertools import combinations
import numpy as np
import numba as nb


class profile:
    """ Profiles can be one sequence or many aligned sequences. It is created upon importing
        protein fasta and at each stage of a progressive alignment.
    """
    def __init__(self,deflines,aaseqs,positions=None):
        self.n = len(deflines) #defines the number of sequences in the profile
        self.deflines = np.array(deflines) #stores the deflines of the sequences in the profile
        self.aaseqs = np.array([list(x) for x in aaseqs]) #stores the aaseqs of the profile in a length*n array
        self.length = self.aaseqs.shape[1] #defines the length of the profile
        self.positions = np.array(range(self.length)) #stores the positions of the profile in a length*n array
        self.weights = 1

    def embed(self,model,padding=False):
        if self.n==1: #stores embedding in a length*embedding_dim*n array
            if padding:
                pad = 'XXXXXXXXXX'
                padded_aa = pad+''.join([num for num in self.aaseqs[0]])+pad
                self.embeddings = model.extract(padded_aa)[10:-10,:]
            else:
                self.embeddings = model.extract(''.join([num for num in self.aaseqs[0]]))
        else:
            raise Exception('This profile is multiple aligned sequences, not a single sequence.')
    
    def embedding_override(self,embedding):
        self.embeddings = embedding
        # if self.n != embedding.shape[2]:
        #     raise Exception('The provided embedding does not match the number of sequences in the profile.')
        # else:
        #     self.embeddings = embedding
    
    def assignweights(self,weights):
        self.weights = weights

    def slice(self,start=0,end=-1):
        new_aaseqs = self.aaseqs[start:end,:]
        new_positions = self.positions[start:end,:]
        new_length = new_aaseqs.shape[0]
        self.aaseqs = new_aaseqs
        self.positions = new_positions
        self.length = new_length
        if hasattr(self, 'embedding'):
            new_embeddings = self.embeddings[start:end,:,:]
            self.embeddings = new_embeddings
        self.sliced = True

        

class positions:
    """ Positions objects are created as an intermediate between aligning two profiles and merging
        those profiles, or when selecting anchors for an alignment.
    """
    def __init__(self,deflines0,deflines1,coordinates,score=None):
        #it might be best to insist on a score, because I can't think of a scenario where it wouldn't be needed
        self.deflines0 = deflines0
        self.deflines1 = deflines1
        self.coordinates = coordinates
        self.score = score

class matrix:
    """ This matrix class is really just a way to assign additional information to a numpy array.
        Is that proper? No clue.
    """
    def __init__(self,deflines0,deflines1,array):
        self.deflines0 = deflines0
        self.deflines1 = deflines1
        self.array = array


    




def pickler(cucumber, picklefile):
    """ Pickles an object.

        Parameters:
            cucumber (object): object to pickle
            picklefile (filepath): filepath to save the pickle to
    """
    with open(picklefile, 'wb') as f:
        pickle.dump(cucumber, f)

def unpickler(picklefile):
    """ Unpickles an object.

        Parameters:
            picklefile (filepath): filepath to pickle file to unpickle

        Returns:
            object
    """
    with open(picklefile, 'rb') as f:
        cucumber = pickle.load(f)
    return cucumber


def chunk_dict(dictionary, SIZE=100):
    """ Splits a dictionary into chunks.

        Parameters:
            dictionary (dictionary): dictionary to split
            SIZE (integer): number of key-value pairs per dictionary chunk

        Returns:
            dictionary of chunks
    """
    it = iter(dictionary)
    for i in range(0, len(dictionary), SIZE):
        yield {k:dictionary[k] for k in islice(it, SIZE)}


##functions for writing a new line of data to output
def write_line_to_output(data,output):
    """ Writes a single line of data to an output file as a tab-delimited row. Intended to minimize memory requirements.

        Parameters:
            data (list): list of values corresponding to fields in the output file
            output (filepath): filepath of output file (tsv format)
    """
    with open(output, 'a') as output_file:
        output_string = [str(x) for x in data]
        output_file.write('\t'.join(output_string)+'\n')



##functions for importing various file types

def fasta_to_seq_dict(file):
    """ Imports a fasta file as a sequence dictionary.

        Parameters:
            file (filepath): filepath to a fasta file

        Returns:
            sequence dictionary (deflines are keys, sequences are values)
    """
    seq_dict = {}
    with open(file) as f:
        for line in f:
            if line.startswith(">"):
                sequence_name = line.lstrip(">").rstrip()
                seq_dict.setdefault(sequence_name, '')
            else:
                seq_dict[sequence_name] += line.rstrip().upper()
    return seq_dict


def seq_dict_to_profiles_dict(seq_dict,unalign=False):
    """ Turns a sequence dictionary into a dictionary of profiles.

        Parameters:
            seq_dict (dictionary): sequence dictionary
            unalign (boolean): if True, remove '-' and '/' characters from sequence strings (default False)

        Returns:
            profile dictionary (deflines are keys, profiles are values)
    """
    profiles_dict = {}
    if unalign:
        for key in seq_dict.keys():
            profiles_dict[key] = profile([key],[seq_dict[key].replace('*','').replace('-','').replace('/','')])
    else:
        for key in seq_dict.keys():
            profiles_dict[key] = profile([key],[seq_dict[key].replace('*','')])
    return profiles_dict


def fasta_to_profiles_dict(file,unalign=False):
    """ Imports a fasta file as a profiles dictionary.

        Parameters:
            file (filepath): filepath to a fasta file
            unalign (boolean): if True, remove '-' and '/' characters from sequence strings (default False)

        Returns:
            profile dictionary (deflines are keys, profiles are values)
    """
    seq_dict = fasta_to_seq_dict(file)
    return seq_dict_to_profiles_dict(seq_dict,unalign=unalign)


def usalign_to_positions(file):
    """ Turns a USalign/TMalign file into a positions object.

        Parameters:
            file (filepath): filepath to a USalign/TMalign output file

        Returns:
            positions object
    """
    with open(file) as fp:
        for i, line in enumerate(fp):
            if i == 9:
                deflines0 = os.path.basename(line.split(":")[1].replace(' ','')).partition('.')[0]
            if i == 10:
                deflines1 = os.path.basename(line.split(":")[1].replace(' ','')).partition('.')[0]
            if i == 15:
                splitline15 = line.split()
                struc0_norm_TMscore = float(splitline15[1])
            if i == 16:
                splitline16 = line.split()
                struc1_norm_TMscore = float(splitline16[1])
            if i == 20:
                #data_dict['seq0'] = line.replace('-','').replace("\n", "")
                aaseq0_aln = list(line)
            if i == 22:
                #data_dict['seq1'] = line.replace('-','').replace("\n", "")
                aaseq1_aln = list(line)
    coordinates = []
    aaseq0_idx = -1
    aaseq1_idx = -1
    for aln_idx in range(len(aaseq0_aln)):
        if (aaseq0_aln[aln_idx] != '-'):
            aaseq0_idx += 1
        if (aaseq1_aln[aln_idx] != '-'):
            aaseq1_idx += 1
        if (aaseq0_aln[aln_idx] != '-') and (aaseq1_aln[aln_idx] != '-'):
            coordinates.append([aaseq0_idx,aaseq1_idx])
    return positions([deflines0],[deflines1],np.array(coordinates),score=(struc0_norm_TMscore,struc1_norm_TMscore))



##functions for importing various file types that don't yet feel standardized enough


def ali_to_seq_dict(file):
    """ Imports an ali file as a sequence dictionary.

        Parameters:
            file (filepath): filepath to an ali file

        Returns:
            sequence dictionary (deflines are keys, sequences are values)
    """
    ali_dict = {}
    after_header = False
    with open(file) as f:
        for line in f:
            if line.startswith(">"):
                after_header = True
            if after_header:
                if line.startswith(">"):
                    sequence_name = line.lstrip(">").rstrip()
                    ali_dict.setdefault(sequence_name, {'seq':'','info':''})
                elif ':' in line:
                    ali_dict[sequence_name]['info'] += line.rstrip()
                else:
                    ali_dict[sequence_name]['seq'] += line.rstrip()
            # else:
            #     print(line)
    seq_dict = {}
    for key in ali_dict.keys():
        defline = key+'___'+ali_dict[key]['info']
        sequence = ali_dict[key]['seq']
        seq_dict[defline] = sequence
    return seq_dict


def aln_seq_dict_to_pairwise_positions(seq_dict,pairwise_combinations):
    #pairwise_combinations = list(combinations(seq_dict.keys(), 2))
    pairwise_positions = {}
    for combination in pairwise_combinations:
        coordinates = []
        aaseq0_aln = seq_dict[combination[0]]
        aaseq1_aln = seq_dict[combination[1]]
        aaseq0_idx = -1
        aaseq1_idx = -1
        for aln_idx in range(len(aaseq0_aln)):
            if (aaseq0_aln[aln_idx] != '-') and (aaseq0_aln[aln_idx] != '/'):
                aaseq0_idx += 1
            if (aaseq1_aln[aln_idx] != '-') and (aaseq0_aln[aln_idx] != '/'):
                aaseq1_idx += 1
            if (aaseq0_aln[aln_idx] != '-') and (aaseq1_aln[aln_idx] != '-') and (aaseq0_aln[aln_idx] != '/') and (aaseq1_aln[aln_idx] != '/'):
                coordinates.append([aaseq0_idx,aaseq1_idx])
        pairwise_positions[combination] = positions([combination[0]],[combination[1]],np.array(coordinates))
    return pairwise_positions


def pairwise_alignment_fasta_to_positions(file):
    seq_dict = fasta_to_seq_dict(file)
    pairwise_combinations = list(combinations(seq_dict.keys(), 2))
    return aln_seq_dict_to_pairwise_positions(seq_dict,pairwise_combinations)

def import_outside_pairwise_fasta_aln(filelist):
    output = {}
    for file in filelist:
        name = os.path.basename(file).rpartition('_')[0]
        output[name] = list(pairwise_alignment_fasta_to_positions(file).values())[0]
    return output






