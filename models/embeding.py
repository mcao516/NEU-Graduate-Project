import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class Embedding_layer(nn.Module):
    """Embedding layer of the model.
    """
    def __init__(self, vocab_size, word_dim, char_size, char_dim, char_hidden_dim, pre_trained=None, padding_idx=0, drop_out=0.5):
        """Initialize embedding layer. If pre_trained is provided, 
           initialize the model using pre-trained embeddings.
        """
        super(Embedding_layer, self).__init__()
        self.vocab_size = vocab_size
        self.char_size = char_size
        self.word_dim = word_dim
        self.char_dim = char_dim
        self.char_hidden_dim = char_hidden_dim
        self.pre_trained = pre_trained
        self.padding_idx = padding_idx
        self.drop_out = drop_out
        
        self._build_model(self.drop_out) # build model...
    
    def _build_model(self, drop_out):
        self.word_embedding = nn.Embedding(self.vocab_size, self.word_dim,
                                           padding_idx=self.padding_idx)
        self.char_embedding = nn.Embedding(self.char_size, self.char_dim,
                                           padding_idx=self.padding_idx)
        
        self.char_rnn = nn.LSTM(self.char_dim, self.char_hidden_dim,
                                batch_first=True, bidirectional=True)

        self.char_proj = nn.Linear(self.char_hidden_dim*2, self.char_hidden_dim)
        
        # apply pre-trained embeddings
        if self.pre_trained is not None:
            self.word_embedding.weight.data.copy_(torch.from_numpy(self.pre_trained))
        # add dropout layer
        self.dropout = nn.Dropout(p=drop_out)
        
    def forward(self, word_ids, char_ids=None, word_lens=None):
        """
        Args:
            word_ids: [batch_size, max_seq_len]
            char_ids: [batch_size, max_seq_len, max_word_len]
            word_lens: [batch_size, max_seq_len]
        Return:
            embedded: [batch_size, max_seq_len, embedding_dim]
        """
        # word_embeddings: [batch_size, max_seq_len, word_dim]
        word_embeddings = self.word_embedding(word_ids)
        # word_embeddings = self.dropout(word_embeddings)
        # no character-level embeddings
        if char_ids is None:
            return word_embeddings
        
        batch_size, max_seq_len, max_word_len = char_ids.size()
        # char_embedded: [batch_size, max_seq_len, max_word_len, char_dim]
        char_embedded = self.char_embedding(char_ids)
        # char_embedded = self.dropout(char_embedded)
        char_embedded = char_embedded.view(batch_size*max_seq_len, max_word_len, -1)
        
        assert word_lens.size(0) == batch_size and word_lens.size(1) == max_seq_len
        word_lens = word_lens.view(batch_size*max_seq_len)
        lengths_sorted, idx_sort = torch.Tensor.sort(word_lens, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        assert char_embedded.size(0) == idx_sort.size(0)
        char_embedded = char_embedded.index_select(0, idx_sort)
        char_packed = pack_padded_sequence(char_embedded, lengths_sorted, batch_first=True)
        
        # h_n: [2, batch_size*max_seq_len, char_hidden_dim]
        _, (h_n, _) = self.char_rnn(char_packed)
        assert h_n[0].size(0) == idx_unsort.size(0)
        fw_hn = h_n[0].index_select(0, idx_unsort)
        bw_hn = h_n[1].index_select(0, idx_unsort)
        # idx_unsort_expand = idx_unsort.view(-1, 1).expand(batch_size*max_seq_len, h_n[0].size(-1))
        # fw_hn = h_n[0].gather(0, idx_unsort_expand)
        # bw_hn = h_n[1].gather(0, idx_unsort_expand)
        assert fw_hn.size(0) == batch_size*max_seq_len    
    
        # char_hidden: [batch_size, max_seq_len, 2*char_hidden_dim]
        char_hiddens = torch.cat((fw_hn, bw_hn), -1).view(batch_size, max_seq_len, -1)
        assert char_hiddens.size(2) == self.char_hidden_dim*2

        char_hiddens = torch.tanh(self.char_proj(char_hiddens))
        # char_hiddens = self.dropout(char_hiddens)
        assert char_hiddens.size(2) == self.char_hidden_dim

        # [batch_size, max_seq_len, word_dim+char_hidden_dim]
        final_embeddings = torch.cat((word_embeddings, char_hiddens), -1)
        final_embeddings = self.dropout(final_embeddings)
        return final_embeddings