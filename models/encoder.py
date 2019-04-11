import torch
import torch.nn.functional as F

from torch import nn
from torch.nn import LSTM, Linear, LSTMCell, Module
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ContextEncoder(nn.Module):
    """Sinal directional LSTM network, encoding pre- and pos-context.
    """
    def __init__(self, input_size, hidden_size, wordEmbed, drop_out=0.5):
        """Initialize Context Encoder
        Args:
            input_size: input embedding size
            hidden_size: output hidden size
        """
        super(ContextEncoder,self).__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.word_embed  = wordEmbed # embedding layer
        self.drop_out = drop_out
        
        self._build_model(self.drop_out)
    
    def _build_model(self, drop_out):
        """Build context-encoder model"""
        self.pre_rnn = nn.LSTM(self.input_size, self.hidden_size, batch_first=True, bidirectional=True)
        self.pos_rnn = nn.LSTM(self.input_size, self.hidden_size, batch_first=True, bidirectional=True)
        self.output_cproj = nn.Linear(self.hidden_size*4, self.hidden_size)
        self.output_hproj = nn.Linear(self.hidden_size*4, self.hidden_size)
        self.tanh = nn.Tanh()

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, _pre, _pos):
        """Encoding context sequences
        
        Args:
            _pre: (prec_word_ids, prec_seq_lens, prec_char_ids, prec_word_lens)
            _pos: (posc_word_ids, posc_seq_lens, posc_char_ids, posc_word_lens)
        """
        prec_word_ids, prec_seq_lens, prec_char_ids, prec_word_lens = _pre
        posc_word_ids, posc_seq_lens, posc_char_ids, posc_word_lens = _pos
        # embed_pre: [batch, max_seq_len, word_dim+char_hidden*2]
        embed_pre = self.word_embed(prec_word_ids, prec_char_ids, prec_word_lens)
        # embed_pos: [batch, max_seq_len, word_dim+char_hidden*2]
        embed_pos = self.word_embed(posc_word_ids, posc_char_ids, posc_word_lens)
        
        # sort lengths
        prec_lens_sorted, pre_idx_sort = torch.sort(prec_seq_lens, dim=0, descending=True)
        posc_lens_sorted, pos_idx_sort = torch.sort(posc_seq_lens, dim=0, descending=True)
        _, pre_idx_unsort = torch.sort(pre_idx_sort, dim=0)
        _, pos_idx_unsort = torch.sort(pos_idx_sort, dim=0)
        # sort embedded sentences
        embed_pre = embed_pre.index_select(0, pre_idx_sort)
        embed_pos = embed_pos.index_select(0, pos_idx_sort)
        
        pre_packed = pack_padded_sequence(embed_pre, prec_lens_sorted, batch_first=True)
        pos_packed = pack_padded_sequence(embed_pos, posc_lens_sorted, batch_first=True)
        
        # pre_state: ([2, batch_size, hidden], [2, batch_size, hidden])
        _, pre_state = self.pre_rnn(pre_packed)
        _, pos_state = self.pos_rnn(pos_packed)
        
        # restore to the initial order
        pre_h, pre_c = torch.cat((pre_state[0][0], pre_state[0][1]), -1), torch.cat((pre_state[1][0], pre_state[1][1]), -1)
        pos_h, pos_c = torch.cat((pos_state[0][0], pos_state[0][1]), -1), torch.cat((pos_state[1][0], pos_state[1][1]), -1)
        # pre_idx_resort = pre_idx_unsort.view(-1, 1).expand(pre_h.size(0), pre_h.size(1))
        # pos_idx_resort = pos_idx_unsort.view(-1, 1).expand(pos_h.size(0), pos_h.size(1))
        # pre_h = pre_h.gather(0, pre_idx_resort)
        # pre_c = pre_c.gather(0, pre_idx_resort)
        # pos_h = pos_h.gather(0, pos_idx_resort)
        # pos_c = pos_c.gather(0, pos_idx_resort)
        pre_h = pre_h.index_select(0, pre_idx_unsort)
        pre_c = pre_c.index_select(0, pre_idx_unsort)
        pos_h = pos_h.index_select(0, pos_idx_unsort)
        pos_c = pos_c.index_select(0, pos_idx_unsort)
        
        # final_hidd_proj/final_cell_proj: [batch_size, hidden]
        final_hidd_proj = self.tanh(self.output_hproj(torch.cat((pre_h, pos_h), 1)))
        final_cell_proj = self.tanh(self.output_cproj(torch.cat((pre_c, pos_c), 1)))
        
        del embed_pre, embed_pos, pre_packed, pos_packed
        del pre_state, pos_state
        del pre_h, pre_c, pos_h, pos_c

        final_hidd_proj = self.dropout(final_hidd_proj)
        final_cell_proj = self.dropout(final_cell_proj)

        return final_hidd_proj, final_cell_proj


# encoder net for the article
class BidirectionalEncoder(nn.Module):
    """Bidirectional LSTM encoder.
    """
    def __init__(self, input_size, hidden_size, wordEmbed, drop_out=0.5):
        super(BidirectionalEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.word_embed = wordEmbed # embedding layer

        self._build_model(drop_out) # build model...
        
    def _build_model(self, drop_out):
        self.encoder = nn.LSTM(self.input_size, self.hidden_size, 
            batch_first=True, bidirectional=True)
        self.output_cproj = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.output_hproj = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.dropout = nn.Dropout(p=0.1)

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