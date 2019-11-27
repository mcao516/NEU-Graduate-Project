import torch
import torch.nn as nn
import torch.nn.functional as F


class CharLSTM(nn.Module):
    """Character-level embedding using bidirectional LSTM"""

    def __init__(self,
                 char_size,
                 char_dim,
                 char_hidden_dim,
                 padding_idx=0,
                 dropout=0.2):
        """Constructs CharLSTM model.

        Args:
            char_size: total characters in the vocabulary.
            char_dim: character embedding size.
            char_hidden_dim: hidden size of character LSTM.
            dropout: dropout rate.
        """
        super(CharLSTM, self).__init__()
        self.char_hidden_dim = char_hidden_dim
        self.char_embedding = nn.Embedding(char_size, char_dim,
                                           padding_idx=padding_idx)
        self.char_rnn = nn.LSTM(char_dim, char_hidden_dim,
                                batch_first=True, bidirectional=True)
        self.char_proj = nn.Linear(char_hidden_dim * 2, char_hidden_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, char_ids, word_lens):
        """
        Args:
            char_ids: [batch_size, max_seq_len, max_word_len]
            word_lens: [batch_size, max_seq_len]

        Return:
            c_emb: [batch_size, max_seq_len, char_hidden_dim]
        """
        batch_size, seq_len, word_len = char_ids.size()

        # char_embedded: [batch_size, max_seq_len, max_word_len, char_dim]
        char_embs = self.char_embedding(char_ids)
        char_embs = char_embs.view(batch_size * seq_len, word_len, -1)

        # h_n: [2, batch_size*max_seq_len, char_hidden_dim]
        _, (h_n, _) = self.char_rnn(char_embs)
        c_hidden = torch.cat((h_n[0], h_n[1]), -1)
        c_hidden = c_hidden.view(batch_size, seq_len, -1)

        c_emb = torch.tanh(self.char_proj(c_hidden))
        c_emb = self.dropout(c_emb)

        assert c_emb.shape == (batch_size, seq_len, self.char_hidden_dim)
        return c_emb


class HighwayMLP(nn.Module):
    """Implement highway network."""

    def __init__(self,
                 input_size,
                 activation=nn.functional.relu,
                 gate_activation=torch.sigmoid):

        super(HighwayMLP, self).__init__()

        self.act, self.gate_act = activation, gate_activation

        self.mlp = nn.Linear(input_size, input_size)
        self.transform = nn.Linear(input_size, input_size)

    def forward(self, x):
        """
        Args:
            x: [*, input_size]

        Return:
            out: [*, input_size]
        """
        mlp_out = self.act(self.mlp(x))
        gate_out = self.gate_act(self.transform(x))
        return gate_out * mlp_out + (1 - gate_out) * x


class CharCNN(nn.Module):
    """Character-level embedding with convolutional neural networks.
    """
    def __init__(self,
                 char_size,
                 char_dim,
                 output_size=50,
                 filter_num=10,
                 max_filter_size=5,
                 padding_idx=0,
                 dropout=0.2):
        """Constructs CharCNN model.

        Args:
            char_size: total characters in the vocabulary.
            char_dim: character embedding size.
            output_size: output character embedding size.
            filter_num: number of filters (each size).
            max_filter_size: the largest filter width.
            padding_idx: index used for padding.
            dropout: dropout rate.
        """
        super(CharCNN, self).__init__()

        self.char_dim = char_dim
        self.filter_num = filter_num
        self.max_filter_size = max_filter_size

        self.embed = nn.Embedding(char_size, char_dim, padding_idx=padding_idx)

        self.filters = nn.ModuleList()
        for size in range(1, max_filter_size + 1):
            self.filters.append(nn.Conv1d(char_dim, filter_num, kernel_size=size))

        self.highway = HighwayMLP(max_filter_size * filter_num)
        self.proj = nn.Linear(max_filter_size * filter_num, output_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, char_ids, word_lens):
        """
        Args:
            char_ids: [batch_size, max_seq_len, max_word_len]
            word_lens: [batch_size, max_seq_len]

        Return:
            c_emb: [batch_size, max_seq_len, char_hidden_dim]
        """
        batch_size, max_seq_len, max_word_len = char_ids.shape

        # embedding
        char_embedded = self.embed(char_ids)
        char_embedded = char_embedded.view(-1, max_word_len, self.char_dim)
        char_embedded = char_embedded.permute(0, 2, 1)  # [batch_size*max_seq_len, char_dim, max_word_len]

        # convolution layer & max pooling
        outputs = []
        for i, conv in enumerate(self.filters):
            if i + 1 <= max_word_len:
                conv_out = conv(char_embedded)
                out = F.max_pool2d(conv_out, kernel_size=(1, conv_out.shape[-1]))
                out = out.squeeze(-1)  # [batch_size*max_seq_len, filter_num]
                outputs.append(out)

        outputs = torch.cat(outputs, -1)
        outputs = self._pad_outputs(outputs)
        assert outputs.shape == (batch_size * max_seq_len, self.max_filter_size * self.filter_num)

        # highway network
        highway_out = self.highway(outputs)

        # proj
        final_out = torch.relu(self.proj(highway_out))
        final_out = final_out.view(batch_size, max_seq_len, -1)
        final_out = self.dropout(final_out)

        return final_out

    def _pad_outputs(self, x):
        """In case the max word length is less than the max filter width,
           use this function to pad the output.

        Args:
            x: tensor, [batch_size * max_seq_len, N * filter_num] (N <= max_filter_size)

        return:
            out: tensor, [batch_size * max_seq_len, max_filter_size * filter_num]
        """
        bm, input_size = x.shape
        dim_to_pad = self.filter_num * self.max_filter_size - input_size
        assert dim_to_pad >= 0

        if dim_to_pad == 0:
            return x
        else:
            padder = torch.zeros((bm, dim_to_pad), dtype=x.dtype, device=x.device)
            out = torch.cat((x, padder), -1)
            return out


class Embedding_layer(nn.Module):
    """Embedding layer of the model.
    """
    def __init__(self,
                 vocab_size,
                 word_dim,
                 char_size,
                 char_dim,
                 char_hidden_dim,
                 pre_trained=None,
                 padding_idx=0,
                 drop_out=0.5,
                 char_emb_type='rnn'):
        """Initialize embedding layer.
           If pre_trained is not None, initialize weights using pre-trained embeddings.

        Args:
            vocab_size: total words in the vocabulary.
            word_dim: word embedding size.
            char_size: total characters in the vocabulary.
            char_dim: character embedding size.
            char_hidden_dim: hidden size of character LSTM.
            pre_trained: numpy array, pre-trained word embeddings.
            padding_idx: index used to pad sequences.
            dropout: dropout rate.
        """
        super(Embedding_layer, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, word_dim,
                                           padding_idx=padding_idx)
        # apply pre-trained embeddings
        if pre_trained is not None:
            self.word_embedding.weight.data.copy_(torch.from_numpy(pre_trained))

        if char_emb_type == 'rnn':
            self.char_embed = CharLSTM(char_size, char_dim, char_hidden_dim)
        elif char_emb_type == 'cnn':
            self.char_embed = CharCNN(char_size, char_dim, output_size=char_hidden_dim)
        else:
            raise ValueError("Unknown character embedding type: {}".format(char_emb_type))

        # add dropout layer
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, word_ids, char_ids=None, word_lens=None):
        """All sequences and words are padded.

        Args:
            word_ids: [batch_size, seq_len]
            char_ids: [batch_size, seq_len, word_len]
            word_lens: [batch_size, seq_len]

        Return:
            final_emb: [batch_size, seq_len, word_dim + (char_hidden_dim)]
        """
        # word_embeddings: [batch_size, max_seq_len, word_dim]
        w_emb = self.word_embedding(word_ids)

        # no character-level embeddings
        if char_ids is None:
            final_emb = w_emb
        else:
            c_emb = self.char_embed(char_ids, word_lens)
            final_emb = torch.cat((w_emb, c_emb), -1)

        final_emb = self.dropout(final_emb)

        return final_emb
