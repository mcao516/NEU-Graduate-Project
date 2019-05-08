#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import torch

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# encoder net for the article
class BidirectionalEncoder(nn.Module):
    """Bidirectional LSTM encoder.
    """
    def __init__(self, input_size, hidden_size, wordEmbed, drop_out=0.1):
        super(BidirectionalEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.word_embed = wordEmbed
        self.encoder = nn.LSTM(input_size, 
                               hidden_size, 
                               batch_first=True, 
                               bidirectional=True)

        self.output_cproj = nn.Linear(hidden_size * 2, hidden_size)
        self.output_hproj = copy.deepcopy(self.output_cproj)
        
        self.dropout = nn.Dropout(p=drop_out)


    def forward(self, input_word_ids, input_seq_lens, input_seq_mask, 
                input_char_ids, input_word_lens):
        """Encode description, return outputs and the last hidden state
           
        Args:
            input_word_ids: [batch_size, max_seq_len]
            input_seq_lens: [batch_size]
            input_seq_mask: [batch_size, max_seq_len]
            input_char_ids: [batch_size, max_seq_len, max_word_len]
            input_word_lens: [batch_size, max_seq_len]
        """
        # _input: [batch, max_seq_len]
        batch_size, max_len = input_word_ids.size(0), input_word_ids.size(1)
        # embed_wd: [batch, max_seq_len, word_dim + char_dim*2]
        embed_wd = self.word_embed(input_word_ids, input_char_ids, input_word_lens)
        
        # sorting the batch for packing
        lengths_sorted, idx_sort = torch.sort(input_seq_lens, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        embed_wd = embed_wd.index_select(0, idx_sort)
        
        input_packed = pack_padded_sequence(embed_wd, lengths_sorted, batch_first=True)
        
        # outputs: [batch, max_seq_len, hidden_size * 2]
        # final_state: ([2, batch, hidden_size], [2, batch, hidden_size])
        outputs_packed, final_state = self.encoder(input_packed)
        outputs_padded, _ = pad_packed_sequence(outputs_packed, batch_first=True)
        outputs = outputs_padded.index_select(0, idx_unsort)
        
        h_n = final_state[0].index_select(1, idx_unsort)
        c_n = final_state[1].index_select(1, idx_unsort)
        assert outputs.size(0) == batch_size
        assert outputs.size(1) == max_len
        assert outputs.size(2) == self.hidden_size * 2
        
        # inv_mask: [batch, max_seq_len, hidden_size * 2]
        mask = input_seq_mask.eq(0).detach()
        inv_mask = mask.eq(0).unsqueeze(2).expand(batch_size, max_len, 
            self.hidden_size * 2).float().detach()
        hidden_out = outputs * inv_mask
        
        # final_hidd_proj: [batch_size, hidden_size]
        final_hidd_proj = self.output_hproj(torch.cat((h_n[0], h_n[1]), 1))
        final_cell_proj = self.output_cproj(torch.cat((c_n[0], c_n[1]), 1))

        del embed_wd, input_packed

        # apply dropout
        hidden_out = self.dropout(hidden_out)
        final_hidd_proj = self.dropout(final_hidd_proj)
        final_cell_proj = self.dropout(final_cell_proj)

        return hidden_out, final_hidd_proj, final_cell_proj, mask