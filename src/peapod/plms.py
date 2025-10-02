#!/usr/bin/env python
import os
import torch
import re
import esm
import ankh
#from transformers import BertModel, BertTokenizer
from transformers import T5Tokenizer, T5EncoderModel
from transformers import logging

# This code is modified from https://git.scicore.unibas.ch/schwede/EBA


class seq_feature_extractor_base:
    """Base class defining the interface for feature extraction based
    on a sequence.
    """
    def __init__(self):
        pass

    def full_seq_features(self):
        """Must be implemented by child class. Returns True if the extracted
        features relate to the full input sequence. Returns False if extracted
        features are on a per-residue basis.
        """
        raise NotImplementedError('full_seq_features not implemented')

    def dim(self):
        """Must be implemented by child class and returns the number of
        features.
        """
        raise NotImplementedError('dim not implemented')

    def device(self):
        """Must be implemented by child class and returns to which device
        the return tensors are sent.
        """
        raise NotImplementedError('device not implemented')

    def extract(self, seqres):
        """Must be implemented by child class and returns a torch tensor with
        dimensionality (len(seqres), self.dim()) or (self.dim()). Depending
        on return value of self.full_seq_features().
        """
        raise NotImplementedError('extract not implemented')


class seq_feature_extractor(seq_feature_extractor_base):
    """Concatenates features from several :class:`seq_feature_extractor_base` 
    instances.
    """

    def __init__(self):
        """Starts as an empty container, :class:`seq_feature_extractor_base` 
        instances can be added with :func:`register`.
        """
        self.feature_extractors = list()
        self.dim = 0

    def register(self, feature_extractor):
        """Adds a new :class:`seq_feature_extractor_base` and enforces 
        consistency of feature_extractor.full_seq_features() and 
        feature_extractor.device() with already registered extractors.

        :param feature_extractor: Feature extractor to add
        :type feature_extractor: :class:`seq_feature_extractor_base`

        :raises: :class:`ValueError` if *feature_extractor* is inconsistent
                 with already registered extractors.
        """
        # enforce consistency with already present extractors
        if len(self.feature_extractors) > 0:
            if feature_extractor.full_seq_features() != self.feature_extractors[0].full_seq_features():
                raise ValueError('All feature extractors must either be per-residue extractors of full seq extractors')
            if feature_extractor.device() != self.feature_extractors[0].device():
                raise ValueError('All feature extractors must send return tensors to same device')
        self.feature_extractors.append(feature_extractor)
        self.dim += feature_extractor.dim()


    def full_seq_features(self):
        """Returns the return value of :func:`full_seq_features` of the 
        registered extractors. 

        :raises: :class:`RuntimeError` if no extractor has been registered yet
        """
        if len(self.feature_extractors) == 0:
            raise RuntimeError("Not feature extractors registered yet!")
        return self.feature_extractors[0].full_seq_features()

    def dim(self):
        """Returns the summed number of features of all registered feature 
        extractors.
        """
        return self.dim

    def device(self):
        """Returns the device to which the features are sent

        :raises: :class:`RuntimeError` if no extractor has been registered yet
        """
        if len(self.feature_extractors) == 0:
            raise RuntimeError("Not feature extractors registered yet!")
        return self.feature_extractors[0].device()

    def extract(self, seqres):
        """Returns the concatenated features from all registered extractors.
        Depending on :func:`full_seq_features` this will be a torch tensor of
        shape (len(*seqres*), :func:`dim`) or (:func:`dim`)
        
        :raises: :class:`RuntimeError` if no extractor has been registered yet
        """
        if len(self.feature_extractors) == 0:
            raise RuntimeError("Not feature extractors registered yet!")
        return_tensors = [fe.extract(seqres) for fe in self.feature_extractors]
        return torch.cat(return_tensors, 1)


class bert_embeddings(seq_feature_extractor_base):
    def __init__(self, bert_model, bert_tokenizer, embedding_type = 'residue',
                 device = None):
        """Extracts embeddings of protein sequences using bert models that 
        can be loaded by the Python transformers module (see example script).

        :param bert_model: Bert model (see example script)
        :param bert_tokenizer: Bert tokenizer (see example script)
        :param embedding_type: Must be one in ['residue', 'cls', 'avg'].
                               'residue': Extracts per residue embeddings,
                               'cls': Extracts the full seq embedding and 
                               returns the Bert 'cls' (classification) vector,
                               'avg': Similar as 'cls' but returns avg per 
                               residue embeddings instead
        :param device: Sends Bert model to this device if given. The torch
                       tensors returned by extract are on the same device.
        :type bert_model: :class:`transformers.models.bert.modeling_bert.BertModel`
        :type bert_tokenizer: :class:`transformers.models.bert.tokenization_bert.BertTokenizer`
        :type embedding_type: :class:`str`
        :type device: :class:`str`/:class:`torch.device`
        """
        self._bert_model = bert_model
        self._bert_tokenizer = bert_tokenizer
        if embedding_type not in ['residue', 'cls', 'avg']:
            raise RuntimeError('embedding_type must be in [\'residue\', \'cls\', \'avg\']')
        
        self._embedding_type = embedding_type
        self._device = device
        if self._device is not None:
            self._bert_model.to(self._device)
        self._bert_model.eval()

    def full_seq_features(self):
        """Returns True if *embedding_type* in constructor is in ['cls', 'avg'],
        False otherwise.
        """
        return self._embedding_type in ['cls', 'avg']

    def dim(self):
        """Returns dimensionality of embedding space of the BERT model
        """
        return self._bert_model.config.hidden_size

    def device(self):
        """Returns the device to which the embeddings are sent
        """
        return self._device

    def extract(self, seqres):
        """Returns BERT embedding as a torch tensor (sent to *device*)
        according to *embedding_type*.

        :param seqres: SEQRES to encode
        :type seqres: :class:`str`
        """
        tokenizer_in = ' '.join(seqres)
        tokenizer_in = [re.sub(r"[UZOB]", "X", tokenizer_in)]
        ids = self._bert_tokenizer.batch_encode_plus(tokenizer_in, add_special_tokens=True)
        input_ids = torch.tensor(ids['input_ids'])
        attention_mask = torch.tensor(ids['attention_mask'])
        
        if self._device is not None:
            input_ids = input_ids.to(self._device)
            attention_mask = attention_mask.to(self._device)

        with torch.no_grad():
            embedding = self._bert_model(input_ids=input_ids,attention_mask=attention_mask)[0]
            
        if self._embedding_type == 'residue':
            return embedding[0][1:len(seqres)+1][:].cpu().numpy()
        elif self._embedding_type == 'cls':
            return embedding[0][0][:].cpu().numpy()
        elif self._embedding_type == 'avg':
            return torch.mean(embedding[0][1:len(seqres)+1][:], 0).cpu().numpy()
        else:
            # this should never happen as we check for valid embedding type in 
            # constructor
            raise RuntimeError('Invalid embedding type')


class esm_embeddings(seq_feature_extractor_base):
    def __init__(self, esm_model, esm_batch_converter, embedding_type = 'residue',
                 device = None):
        """Extracts embeddings of protein sequences using esm models that 
        can be loaded by the Python transformers module (see example script).??

        :param esm_model: Bert model (see example script)
        :param esm_batch_converter: esm batch converter (see example script)
        :param embedding_type: Must be one in ['residue', 'avg'].
                               'residue': Extracts per residue embeddings,
                               'avg': Similar as 'cls' but returns avg per 
                               residue embeddings instead
        :param device: Sends Bert model to this device if given. The torch
                       tensors returned by extract are on the same device.
        :type esm_model: :class: `esm.model.ProteinBertModel`
        :type esm_batch_converter: :class: `esm.data.BatchConverter`
        :type embedding_type: :class:`str`
        :type device: :class:`str`/:class:`torch.device`
        """

        self._esm_model = esm_model
        self._esm_batch_converter = esm_batch_converter
        if embedding_type not in ['residue', 'avg']:
            raise RuntimeError('embedding_type must be in [\'residue\', \'avg\']')
        
        self._embedding_type = embedding_type
        self._device = device
        if self._device is not None:
            self._esm_model.to(self._device)
        self._esm_model.eval()

    def full_seq_features(self):
        """Returns True if *embedding_type* in constructor is in 'avg',
        False otherwise.
        """
        return self._embedding_type=='avg'

    def dim(self):
        """Returns dimensionality of embedding space of the esm model
        """
        return self._esm_model.emb_layer_norm_before.normalized_shape[0]
        #return self._esm_model.emb_layer_norm_after

    def device(self):
        """Returns the device to which the embeddings are sent
        """
        return self._device

    def extract(self, seqres, esm_layer=33):
        """Returns esm embedding as a torch tensor (sent to *device*)
        according to *embedding_type*.

        :param seqres: SEQRES to encode
        :param esm_layer: layer used to extract embeddings
        
        :type seqres: :class:`str`
        :type esm_layer: :class:`int`
        """
        data = [('sequence', seqres)]

        batch_labels, batch_strs, batch_tokens = self._esm_batch_converter(data)

        if self._device is not None:
            batch_tokens = batch_tokens.to(self._device)

        with torch.no_grad():
            results = self._esm_model(batch_tokens, repr_layers=[esm_layer], return_contacts=True)
        
        token_representations = results["representations"][esm_layer]
        assert(len(seqres) + 2 == token_representations.shape[1])

        if self._embedding_type == 'residue':
            return token_representations[0][1:-1].cpu().numpy()
        
        elif self._embedding_type == 'avg':
            # Generate per-sequence representations via averaging
            # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
            sequence_representations = []
            for i, (_, seq) in enumerate(data):
                sequence_representations.append(token_representations[i, 1 : len(seq) + 1].mean(0))

            return sequence_representations[0].cpu().numpy()
        
        else:
            # this should never happen as we check for valid embedding type in 
            # constructor
            raise RuntimeError('Invalid embedding type')


class ankh_embeddings(seq_feature_extractor_base):
    def __init__(self, ankh_model, ankh_tokenizer, embedding_type = 'residue',
                 device = None):
        """Extracts embeddings of protein sequences using bert models that 
        can be loaded by the Python transformers module (see example script).

        :param bert_model: Bert model (see example script)
        :param bert_tokenizer: Bert tokenizer (see example script)
        :param embedding_type: Must be one in ['residue', 'cls', 'avg'].
                               'residue': Extracts per residue embeddings,
                               'cls': Extracts the full seq embedding and 
                               returns the Bert 'cls' (classification) vector,
                               'avg': Similar as 'cls' but returns avg per 
                               residue embeddings instead
        :param device: Sends Bert model to this device if given. The torch
                       tensors returned by extract are on the same device.
        :type bert_model: :class:`transformers.models.bert.modeling_bert.BertModel`
        :type bert_tokenizer: :class:`transformers.models.bert.tokenization_bert.BertTokenizer`
        :type embedding_type: :class:`str`
        :type device: :class:`str`/:class:`torch.device`
        """
        self._ankh_model = ankh_model
        self._ankh_tokenizer = ankh_tokenizer
        if embedding_type not in ['residue', 'avg']:
            raise RuntimeError('embedding_type must be in [\'residue\', \'avg\']')
        
        self._embedding_type = embedding_type
        self._device = device
        if self._device is not None:
            self._ankh_model.to(self._device)
        self._ankh_model.eval()

    def full_seq_features(self):
        """Returns True if *embedding_type* in constructor is in ['avg'],
        False otherwise.
        """
        return self._embedding_type in ['avg']

    def dim(self):
        """Returns dimensionality of embedding space of the ANKH model
        """
        return self._ankh_model.config.hidden_size

    def device(self):
        """Returns the device to which the embeddings are sent
        """
        return self._device

    def extract(self, seqres):
        """Returns ANKH embedding as a torch tensor (sent to *device*)
        according to *embedding_type*.

        :param seqres: SEQRES to encode
        :type seqres: :class:`str`
        """
        
        protein_sequences = [list(seq) for seq in [seqres]]
        outputs = self._ankh_tokenizer.batch_encode_plus(protein_sequences, add_special_tokens=True, padding=True, is_split_into_words=True, return_tensors="pt")
        
        if self._device is not None:
            outputs = outputs.to(self._device)

        with torch.no_grad():
            embedding = self._ankh_model(input_ids=outputs['input_ids'], attention_mask=outputs['attention_mask'])[0]

        shift_left = 0
        shift_right = -1
        perresidue_embedding = embedding[0][shift_left:shift_right]
            
        if self._embedding_type == 'residue':
            return perresidue_embedding.cpu().numpy()
        elif self._embedding_type == 'avg':
            return torch.mean(perresidue_embedding, 0).cpu().numpy()
        else:
            # this should never happen as we check for valid embedding type in 
            # constructor
            raise RuntimeError('Invalid embedding type')



#########################################################################
#########################################################################
#########################################################################


def load_extractor(model_name, embedding_type, device=None):
    ''' Returns a feature extractor through menzi: esm1b_t33_650M_UR50S per residue or average; bert: per residue, average or cls.

        :param model_name: the model you want to load: ESM-b1, ESM2 or protT5
        :param embedding_type: the kind of representation you want: residue, avg (also cls in case of bert)
        :param device: device in case you want to use GPU

        :type model_name: string
        :type embedding_type: string
        :type device: device object
    '''
    
    if model_name == 'ESM1b':
        esm_model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        esm_batch_converter = alphabet.get_batch_converter()
        esm_model.eval()
    
        return esm_embeddings(esm_model, esm_batch_converter, embedding_type=embedding_type, device=device)

    if model_name == 'ESM2':
        esm_model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        # esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        esm_batch_converter = alphabet.get_batch_converter()
        esm_model.eval()
    
        return esm_embeddings(esm_model, esm_batch_converter, embedding_type=embedding_type, device=device)

    elif model_name == 'ProtT5':
        logging.set_verbosity_error()
        bert_tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False )
        bert_model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")

        return bert_embeddings(bert_model, bert_tokenizer, embedding_type=embedding_type, device=device)

    elif model_name == 'ProstT5':
        logging.set_verbosity_error()
        tokenizer = T5Tokenizer.from_pretrained("Rostlab/ProstT5", do_lower_case=False )
        model = T5EncoderModel.from_pretrained("Rostlab/ProstT5")

        return bert_embeddings(model, tokenizer, embedding_type=embedding_type, device=device)

    elif model_name == 'ankh-large':
        logging.set_verbosity_error()
        model, tokenizer = ankh.load_large_model()
        return ankh_embeddings(model, tokenizer, embedding_type=embedding_type, device=device)

    elif model_name == 'ankh-base':
        logging.set_verbosity_error()
        model, tokenizer = ankh.load_base_model()
        return ankh_embeddings(model, tokenizer, embedding_type=embedding_type, device=device)

    else:
        raise RuntimeError('Invalide model name, use ESM1b, ESM2, ProtT5, ProstT5, ankh-base, or ankh-large')
