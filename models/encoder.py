import torch
from torch import nn
from torch.nn import LSTM, Linear, LSTMCell, Module
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Embedding_layer(nn.Module):
    """Embedding layer of the model.
    """
    def __init__(self, vocab_size, embedding_dim, pre_trained=None):
        """Initialize embedding layer. If pre_trained is provided, 
           initialize the model using pre-trained embeddings.
        """
        super(Embedding_layer, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if pre_trained is not None:
            # apply pre-trained embeddings
            self.embedding.weight.data.copy_(torch.from_numpy(pre_trained))
    
    def forward(self, input):
        return self.embedding(input)


# encoder net for the article
class Encoder(Module):
    """Encoder network, encoding the description and context.
    """
    def __init__(self, input_size, hidden_size, wordEmbed):
        super(Encoder,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.word_embed = wordEmbed # embedding layer
        self.encoder = LSTM(self.input_size, self.hidden_size, 
                            batch_first=True, bidirectional=True)
        self.output_cproj = Linear(self.hidden_size * 2, self.hidden_size)
        self.output_hproj = Linear(self.hidden_size * 2, self.hidden_size)

    def forward(self, _input, _input_mask):
        # _input: [batch, max_seq_len]
        batch_size, max_len = _input.size(0), _input.size(1)
        embed_wd = self.word_embed(_input) # embed_fwd: [batch, seq, embedding_sim]
        
        # get mask for location of PAD
        mask = _input_mask.eq(0).detach()
        
        # outputs: [batch, seq, hidden*2]
        # final_state: ([2, batch, hidden], [2, batch, hidden])
        outputs, final_state = self.encoder(embed_wd)

        # inverse of mask
        inv_mask = mask.eq(0).unsqueeze(2).expand(batch_size, max_len, 
                                                  self.hidden_size * 2).float().detach()
        hidden_out = outputs * inv_mask
        
        # final_hidd_proj: [batch_size, hidden_size]
        h_n, c_n = final_state[0], final_state[1]
        final_hidd_proj = self.output_hproj(torch.cat((h_n[0], h_n[1]), 1))
        final_cell_proj = self.output_cproj(torch.cat((c_n[0], c_n[1]), 1))
        
        return hidden_out, final_hidd_proj, final_cell_proj, mask