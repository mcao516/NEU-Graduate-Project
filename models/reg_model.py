#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn

from models.embeding import Embedding_layer
from models.context_encoder import ContextEncoder
from models.profile_encoder import BidirectionalEncoder
from models.decoder import PointerAttentionDecoder


class REGModel(nn.Module):
    def __init__(self, config):
        super(REGModel, self).__init__()
        self.config = config
        
        # build modules: embedding layer, encoders and decoder
        self.wordEmbed = self._build_embedding_layer()
        self.con_encoder = self._build_context_encoder()
        self.des_encoder = self._build_profile_encoder()
        self.pointerDecoder = self._build_decoder()

    def _build_embedding_layer(self):
        """Create embedding layer.
        """
        return Embedding_layer(self.config.nwords, 
                               self.config.dim_word, 
                               self.config.nchars, 
                               self.config.dim_char, 
                               self.config.hidden_size_char, 
                               self.config.embeddings, 
                               drop_out=self.config.drop_out,
                               char_emb_type=self.config.char_emb_type)
    
    def _build_context_encoder(self):
        """Create context encoder.
        """
        return ContextEncoder(self.config.dim_word + self.config.hidden_size_char, 
                              self.config.hidden_size, self.wordEmbed, 
                              drop_out=self.config.drop_out)

    def _build_profile_encoder(self):
        """Build profile encoder.
        """
        return BidirectionalEncoder(self.config.dim_word + self.config.hidden_size_char, 
                                    self.config.hidden_size, self.wordEmbed, 
                                    drop_out=self.config.drop_out)
        
    def _build_decoder(self):
        """Build pointer decoder.
        """
        pointerDecoder = PointerAttentionDecoder(self.config.dim_word, 
                                                 self.config.hidden_size, 
                                                 self.config.nwords, 
                                                 self.wordEmbed)

        pointerDecoder.setValues(self.config.start_id, self.config.stop_id, 
                                 self.config.unk_id, self.config.npronouns, 
                                 self.config.beam_size, self.config.min_dec_steps, 
                                 self.config.max_dec_steps)
        return pointerDecoder


    def forward(self, context_input, des_input, dec_input=None, decode_flag=False, beam_search=True):
        """Encode context text and entity profile, then generate reference expression.
        """
        # encode context text
        pre_context, pos_context = context_input
        h_n, c_n = self.con_encoder(pre_context, pos_context)

        # encode entity profile
        des_word_ids, des_seq_lens, des_mask, des_char_ids, \
            des_word_lens, max_des_oovs, des_extented_ids = des_input

        hidden_outs, _, _, mask = self.des_encoder(des_word_ids, des_seq_lens, 
            des_mask, des_char_ids, des_word_lens)

        # decode
        self.pointerDecoder.max_article_oov = max_des_oovs
        self.pointerDecoder.beam_search = beam_search
        
        if decode_flag:
            refex, switches = self.pointerDecoder(hidden_outs, (h_n, c_n), mask, 
                des_extented_ids, None, None, None, None, decode=True)
            return refex, switches
        else:
            assert dec_input is not None, "Description input can NOT be none in training model!"
            _input, target, dec_lens, dec_mask = dec_input
            total_loss = self.pointerDecoder(hidden_outs, (h_n, c_n), mask, 
                des_extented_ids, _input, target, dec_lens, dec_mask, decode=False)
            
            return total_loss


