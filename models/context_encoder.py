#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

from copy import deepcopy
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class ContextEncoder(nn.Module):
    """Sinal directional LSTM network, encoding pre- and pos-context.
    """
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 wordEmbed, 
                 drop_out=0.1,
                 pack_sequence=True):
        """Initialize Context Encoder

        Args:
            input_size: input embedding size.
            hidden_size: LSTM hidden size.
            wordEmbed: nn.Module, embedding layer.
        """
        super(ContextEncoder, self).__init__()
        self.word_embed = wordEmbed # embedding layer
        self.pack_sequence = pack_sequence

        self.pre_rnn = nn.LSTM(input_size, 
                               hidden_size, 
                               batch_first=True, 
                               bidirectional=True)
        self.pos_rnn = deepcopy(self.pre_rnn)

        self.output_cproj = nn.Linear(hidden_size*4, hidden_size)
        self.output_hproj = deepcopy(self.output_cproj)
        self.tanh = nn.Tanh()

        self.dropout = nn.Dropout(p=drop_out)


    def _encode(self, inputs, seq_lens=None):
        """
        Args:
            inputs: [batch, seq_len, embedding_dim]
            seq_lens: [batch]
        """
        if seq_lens is not None and self.pack_sequence:
            # sort inputs by sequence length
            lens_sorted, idx_sort = torch.sort(seq_lens, dim=0, descending=True)
            _, idx_unsort = torch.sort(idx_sort, dim=0)
            inputs = inputs.index_select(0, idx_sort)

            inputs = pack_padded_sequence(inputs, lens_sorted, batch_first=True)

        # (h, c): ([2, batch_size, hidden], [2, batch_size, hidden])
        _, (h, c) = self.pre_rnn(inputs)

        # restore order
        h, c = torch.cat((h[0], h[1]), -1), torch.cat((c[0], c[1]), -1)
        h, c = h.index_select(0, idx_unsort), c.index_select(0, idx_unsort)

        return h, c


    def forward(self, _pre, _pos):
        """Encoding context sequences.
        
        Args:
            _pre: (prec_word_ids, prec_seq_lens, prec_char_ids, prec_word_lens)
            _pos: (posc_word_ids, posc_seq_lens, posc_char_ids, posc_word_lens)
        """
        prec_word_ids, prec_seq_lens, prec_char_ids, prec_word_lens = _pre
        posc_word_ids, posc_seq_lens, posc_char_ids, posc_word_lens = _pos

        # [batch, max_seq_len, word_dim + char_hidden]
        embed_pre = self.word_embed(prec_word_ids, prec_char_ids, prec_word_lens)
        embed_pos = self.word_embed(posc_word_ids, posc_char_ids, posc_word_lens)
        
        pre_h, pre_c = self._encode(embed_pre, prec_seq_lens)
        pos_h, pos_c = self._encode(embed_pos, posc_seq_lens)
        
        # final_hidd_proj/final_cell_proj: [batch_size, hidden]
        h_cat = self.tanh(self.output_hproj(torch.cat((pre_h, pos_h), 1)))
        c_cat = self.tanh(self.output_cproj(torch.cat((pre_c, pos_c), 1)))

        h_cat, c_cat = self.dropout(h_cat), self.dropout(c_cat)

        return h_cat, c_cat
