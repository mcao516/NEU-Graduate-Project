import torch
import torch.nn.functional as F

from torch import nn


class Hypothesis(object):
    def __init__(self, token_id, hidden_state, cell_state, log_prob):
        self.full_prediction = token_id # list
        self._h = hidden_state
        self._c = cell_state
        self.log_prob = log_prob
        self.survivability = self.log_prob/float(len(self.full_prediction))

    def extend(self, token_id, hidden_state, cell_state, log_prob):
        """Extend a beam path, add new token_id, update hidden and cell state,
           and modify the probility.
        """
        return Hypothesis(token_id=self.full_prediction + [token_id],
                          hidden_state=hidden_state, cell_state=cell_state,
                          log_prob= self.log_prob + log_prob)


class PointerAttentionDecoder(nn.Module):
    """Pointer-generator attention decoder.
    """
    def __init__(self, input_size, hidden_size, vocab_size, wordEmbed, device):
        super(PointerAttentionDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.word_embed = wordEmbed
        self.device = device

        self.beam_search = True # if doing beam search
        self.max_article_oov = None # max number of OOVs in a batch of data
        
        self._build_model() # build the model
        
    def _build_model(self):
        # lstm decoder
        self.decoderRNN = nn.LSTM(self.input_size, self.hidden_size, batch_first=True)
        
        # params for attention
        # v tanh(W_h h + W_s s + b)
        self.Wh = nn.Linear(self.hidden_size*2, self. hidden_size*2)
        self.Ws = nn.Linear(self.hidden_size, self.hidden_size*2)
        self.v  = nn.Linear(self.hidden_size*2, 1)

        # parameters for p_gen
        # sigmoid(w_h h* + w_s s + w_x x + b)
        self.w_h = nn.Linear(self.hidden_size*2, 3) # attention context vector
        self.w_s = nn.Linear(self.hidden_size, 3) # hidden state
        self.w_x = nn.Linear(self.input_size, 3) # input vector
        self.w_c = nn.Linear(self.hidden_size, 3) # context encoder final hidden state
        
        # params for output proj
        self.V = nn.Linear(self.hidden_size*3, self.vocab_size)

        # dropout layer
        # self.dropout = nn.Dropout(p=0.5)

    def setValues(self, start_id, stop_id, unk_id, nprons, beam_size, min_decode=3, max_decode=10):
        # start/stop tokens
        self.start_id = start_id
        self.stop_id = stop_id
        self.unk_id = unk_id
        # decoding parameters
        self.nprons = nprons
        self.beam_size = beam_size
        self.min_length = min_decode
        self.max_decode_steps = max_decode
        
    def forward(self, enc_states, enc_final_state, enc_mask, article_inds, 
                _input, targets, dec_lens, dec_mask, decode=False):
        """enc_states [batch, max_seq_len, 2*hidden_size]:
               Output states of descirption bidirectional encoder.
           enc_final_states ([batch, hidden_size], ...):
               Final state of context encoder.
           enc_mask [batch_size, max_enc_len]:
               0 or 1 mask for descirption decoder output states.
           article_inds [batch_size, enc_seq_len]:
               Description encoder input with temporary OOV ids repalce 
               each UNK token
           _input [batch_size, dec_seq_len]:
               Decoder inputs, unk token for unknow words
           targets [batch_size, dec_seq_len]:
               Decoder targets, temporary OOV ids for unknow words
           dec_lens [batch_size]:
               Lengths of encoder inputs
           dec_mask [batch_size, des_seq_len]:
               Padding mask for encoder input
           decode Boolean:
               flag for train/eval mode
        """
        if decode is True:
            if self.beam_search:
                return self.decode(enc_states, enc_final_state, enc_mask, article_inds)
            else:
                return self.greedy_decoding(enc_states, enc_final_state, enc_mask, article_inds)
        
        # for attention calculation
        # enc_proj: [batch_size, max_enc_len, 2*hidden]
        batch_size, max_enc_len, enc_size = enc_states.size()
        enc_proj = self.Wh(enc_states.view(batch_size*max_enc_len, enc_size)).view(batch_size, max_enc_len, -1)	
        
        # embed_input: [batch_size, dec_seq_len, embedding_dim]
        embed_input = self.word_embed(_input)
        # state: ([1, batch_size, hidden_size], ...)
        state = enc_final_state[0].unsqueeze(0), enc_final_state[1].unsqueeze(0)
        # hidden: [batch_size, dec_seq_len, hidden_size] 
        hidden, _ = self.decoderRNN(embed_input, state)
        
        lm_loss = []
        
        max_dec_len = _input.size(1)
        # step through decoder hidden states
        for _step in range(max_dec_len):
            _h = hidden[:, _step, :] # _h: [batch_size, hidden_size]
            target = targets[:, _step].unsqueeze(1) # target: [batch_size, 1]
            # mask: [batch_size, 1]
            target_mask_0 = dec_mask[:, _step].unsqueeze(1) 
            # dec_proj: [batch_size, max_enc_len, 2*hidden_size]
            dec_proj = self.Ws(_h).unsqueeze(1).expand_as(enc_proj)
            # dropout
            # enc_proj = self.dropout(enc_proj)
            # dec_proj = self.dropout(dec_proj)
            # attn_scores: [batch_size, max_enc_len]
            e_t = self.v(torch.tanh(enc_proj + dec_proj).view(batch_size*max_enc_len, -1))
            attn_scores = e_t.view(batch_size, max_enc_len)
            # mask to -INF before applying softmax
            attn_scores.masked_fill_(enc_mask, -float('inf'))
            # attn_scores: [batch_size, max_enc_len]
            attn_scores = F.softmax(attn_scores, 1)
            del e_t
            
            # context: [batch_size, 2*hidden_size]
            context = attn_scores.unsqueeze(1).bmm(enc_states).squeeze(1)
            # p_vocab: [batch_size, vocab_size]
            p_vocab = F.softmax(self.V(torch.cat((_h, context), 1)), 1)
            # dropout
            enc_final_state_proj = self.w_c(enc_final_state[0])
            _h_proj = self.w_s(_h)
            # p_switch: [batch_size, 3]
            p_switch = torch.nn.functional.softmax(self.w_h(context) + enc_final_state_proj + self.w_x(embed_input[:, _step, :]) + _h_proj, dim=1)
            p_switch = p_switch.view(-1, 3)

            p_gen = torch.cat((p_switch[:, 0].view(-1, 1).expand(batch_size, self.vocab_size-self.nprons), p_switch[:, 1].view(-1, 1).expand(batch_size, self.nprons)), dim=1)
            assert p_gen.size(0) == batch_size and p_gen.size(1) == self.vocab_size
            p_copy = p_switch[:, 2].view(-1, 1) # [batch_size, 1]

            # weighted_Pvocab: [batch_size, vocab_sze]
            weighted_Pvocab = p_gen * p_vocab 
            weighted_attn = p_copy * attn_scores # [batch_size, max_enc_len]
            assert weighted_attn.size(0) == batch_size and weighted_attn.size(1) == max_enc_len

            if self.max_article_oov > 0:
                # create OOV (but in-article) zero vectors
                ext_vocab = torch.zeros((batch_size, self.max_article_oov), 
                                        requires_grad=True, device=self.device)
                combined_vocab = torch.cat((weighted_Pvocab, ext_vocab), 1)
                del ext_vocab
            else:
                combined_vocab = weighted_Pvocab

            del weighted_Pvocab, p_vocab
            assert article_inds.data.min() >= 0 and article_inds.data.max() < \
                (self.vocab_size + self.max_article_oov), \
                'Recheck OOV indexes! {}/{}'.format(self.max_article_oov, article_inds)

            # scatter article word probs to combined vocab prob.
            # No need to subtract, masked_fill_ 0 ?
            # article_inds_masked = article_inds.masked_fill_(enc_mask, 0)
            article_inds_masked = article_inds
            # combined_vocab: [batch_size, vocab_size + max_oov_num]
            combined_vocab = combined_vocab.scatter_add(1, article_inds_masked, weighted_attn)
            # output: [batch_size, 1]
            output = combined_vocab.gather(1, target) # target: [batch_size, 1]

            # unk_mask: [batch_size, 1]
            unk_mask = target.eq(self.unk_id).detach()
            output.masked_fill_(unk_mask, 1.0)

            lm_loss.append(output.log().mul(-1)*target_mask_0.float())
            
        # add individual losses
        total_masked_loss = torch.cat(lm_loss, 1).sum(1).div(dec_lens.float())
        return total_masked_loss

    def decode_step(self, enc_states, enc_h_n, state, _input, enc_mask, article_inds):
        """One step of decoding
        Args:
            enc_states: [batch, max_seq_len, hidden_size]
            enc_h_n: [1, hidden_size], last hidden state of context encoder hidden state
            state: [beam_size, hidden_size], previous time step hidden state
            _input: current time step input
        Returns:
            combined_vocab: [beam_size, vocab+extra_oov]
            _h, _c: ([beam_size, hidden_size], ...)
        """
        batch_size, max_enc_len, enc_size = enc_states.size()
        enc_proj = self.Wh(enc_states.view(batch_size*max_enc_len, enc_size)).view(batch_size, max_enc_len, -1)
        
        assert _input.max().item() < self.vocab_size, 'Word id {} is out of index'.format(_input.max().item())
        embed_input = self.word_embed(_input)
        _h, _c = self.decoderRNN(embed_input, state)[1]
        _h = _h.squeeze(0)
        dec_proj = self.Ws(_h).unsqueeze(1).expand_as(enc_proj)
        
        e_t = self.v(torch.tanh(enc_proj + dec_proj).view(batch_size*max_enc_len, -1))
        attn_scores = e_t.view(batch_size, max_enc_len)
        del e_t
        attn_scores.masked_fill_(enc_mask, -float('inf'))
        attn_scores = F.softmax(attn_scores, 1)

        context = attn_scores.unsqueeze(1).bmm(enc_states)
        # output proj calculation
        p_vocab = F.softmax(self.V(torch.cat((_h, context.squeeze(1)), 1)), 1)
        # p_switch: [batch_size, 3]
        p_switch = torch.nn.functional.softmax(self.w_h(context.squeeze(1)) + self.w_s(_h) + self.w_x(embed_input[:, 0, :]) + self.w_c(enc_h_n), dim=1)
        p_switch = p_switch.view(-1, 3)
        
        # [batch_size, self.vocab_size]
        # general_words, pronouns, copying
        p_gen = torch.cat((p_switch[:, 0].view(-1, 1).expand(batch_size, self.vocab_size-self.nprons), p_switch[:, 1].view(-1, 1).expand(batch_size, self.nprons)), dim=1)
        assert p_gen.size(0) == batch_size and p_gen.size(1) == self.vocab_size
        p_copy = p_switch[:, 2].view(-1, 1) # [batch_size, 1]

        # weighted_Pvocab: [batch_size, vocab_sze]
        weighted_Pvocab = p_gen * p_vocab 
        weighted_attn = p_copy * attn_scores # [batch_size, max_enc_len]
        assert weighted_attn.size(0) == batch_size and weighted_attn.size(1) == max_enc_len

        if self.max_article_oov > 0:
            # create OOV (but in-article) zero vectors
            ext_vocab = torch.zeros((batch_size, self.max_article_oov), device=self.device, requires_grad=True)
            combined_vocab = torch.cat((weighted_Pvocab, ext_vocab), 1)
            del ext_vocab
        else:
            combined_vocab = weighted_Pvocab
        assert article_inds.data.min() >=0 and article_inds.data.max() < (self.vocab_size+ self.max_article_oov), 'Recheck OOV indexes!'

        combined_vocab = combined_vocab.scatter_add(1, article_inds, weighted_attn)
        
        return combined_vocab, _h, _c.squeeze(0), p_switch.cpu().data.numpy()

    # Beam Search Decoding
    def decode(self, enc_states, enc_final_state, enc_mask, article_inds):
        """Parameters:
           enc_states [1, enc_seq_len, 2*hidden_size]: 
               Description encoder output states.
           enc_final_states ([1, hidden_size], ...):
               Context encoder final state
           enc_mask [1, enc_seq_len]:
               Description padding mask, 1 for pad token
           article_inds [1, enc_seq_len]: 
               Description encoder input with temporary OOV ids repalce 
               each OOV token
        """
        with torch.no_grad():
            assert enc_states.size(0) == enc_mask.size(0) == article_inds.size(0) == 1, "In decoding mode, the input batch size must be to 1"
            # _input: [batch_size(beam_size), seq_len]
            assert type(self.start_id) == int
            _input = torch.tensor([[self.start_id]], dtype=torch.long, device=self.device)
            init_state = enc_final_state[0].unsqueeze(0), enc_final_state[1].unsqueeze(0)
            enc_h_n = enc_final_state[0]
            decoded_outputs = []
            
            # all_hyps: list of current beam hypothesis. 
            all_hyps = [Hypothesis([self.start_id], None, None, 0)]
            # start decoding
            for _step in range(self.max_decode_steps):
                # ater first step, input is of batch_size=curr_beam_size
                # curr_beam_size <= self.beam_size due to pruning of beams that have terminated
                # adjust enc_states and init_state accordingly
                curr_beam_size  = _input.size(0)
                # [1, seq_len, 2*hidden] => [curr_beam_size, seq_len, 2*hidden]
                beam_enc_states = enc_states.expand(curr_beam_size, enc_states.size(1), enc_states.size(2)).contiguous().detach()
                # [1, enc_seq_len] => [curr_beam_size, enc_seq_len]
                beam_article_inds = article_inds.expand(curr_beam_size, article_inds.size(1)).detach()
                beam_enc_mask = enc_mask.expand(curr_beam_size, enc_mask.size(1)).detach()
                vocab_probs, next_h, next_c = self.decode_step(beam_enc_states, 
                    enc_h_n, init_state, _input, beam_enc_mask, beam_article_inds)

                # does bulk of the beam search
                # decoded_outputs: list of all ouputs terminated with stop tokens and of minimal length
                all_hyps, decode_inds, decoded_outputs, init_h, init_c = self.getOverallTopk(vocab_probs, next_h, next_c, all_hyps, decoded_outputs)

                # convert OOV words to unk tokens for lookup
                decode_inds.masked_fill_((decode_inds >= self.vocab_size), self.unk_id)
                decode_inds = decode_inds.t()
                _input = torch.tensor(decode_inds, device=self.device)
                init_state = (torch.tensor(init_h.unsqueeze(0), device=self.device), 
                              torch.tensor(init_c.unsqueeze(0), device=self.device))

            non_terminal_output = sorted(all_hyps, key=lambda x:x.survivability, reverse=True)
            sorted_outputs = sorted(decoded_outputs, key=lambda x:x.survivability, reverse=True)
            
            all_outputs = [item.full_prediction for item in sorted_outputs] + [item.full_prediction for item in non_terminal_output]
            return all_outputs

    def getOverallTopk(self, vocab_probs, _h, _c, all_hyps, results):
        """vocab_probs [curr_beam_size, vocab+oov_size]
        """
        # return top-k values i.e. top-k over all beams i.e. next step input ids
        # return hidden, cell states corresponding to topk
        probs, inds = vocab_probs.topk(k=self.beam_size, dim=1)
        probs = probs.log().data # [curr_beam_size, beam_size]
        inds = inds.data # [curr_beam_size, beam_size]

        candidates = []
        assert len(all_hyps) == probs.size(0), '# Hypothesis and log-prob size dont match'
        # cycle through all hypothesis in full beam
        for i, hypo in enumerate(probs.tolist()):
            for j, _ in enumerate(hypo):
                new_cand = all_hyps[i].extend(token_id=inds[i,j].item(),
                    hidden_state=_h[i].unsqueeze(0), cell_state=_c[i].unsqueeze(0), log_prob=probs[i,j])
                candidates.append(new_cand)
        # sort in descending order
        candidates = sorted(candidates, key=lambda x:x.survivability, reverse=True)
        new_beam, next_inp = [], []
        next_h, next_c = [], []
        # prune hypotheses and generate new beam
        for h in candidates:
            if h.full_prediction[-1] == self.stop_id:
                # weed out small sentences that likely have no meaning
                if len(h.full_prediction) >= self.min_length:
                    results.append(h)
            else:
                new_beam.append(h)
                next_inp.append(h.full_prediction[-1])
                next_h.append(h._h.data)
                next_c.append(h._c.data)
            if len(new_beam) >= self.beam_size:
                break
        assert len(new_beam) >= 1, 'Non-existent beam'
        return new_beam, torch.LongTensor([next_inp]), results, torch.cat(next_h, 0), torch.cat(next_c, 0)
    
    # greedy decoding
    def greedy_decoding(self, enc_states, enc_final_state, enc_mask, article_inds):
        """Parameters:
           enc_states [1, max_seq_len, hidden_size*2]:
               Output states for one description
           enc_final_state ([1, hidden_size], ...):
               Final state of one context(pre- and pos-).
           enc_mask [1, max_seq_len]:
               Padding mask for input description
           article_inds [1, max_seq_len]:
               Description as encoder input with temporary OOV ids repalce 
               each OOV token
        """
        with torch.no_grad():
            assert enc_states.size(0) == enc_mask.size(0) == article_inds.size(0) == 1, "In decoding mode, the input batch size must be to 1"
            _input = torch.tensor([[self.start_id]], dtype=torch.long, device=self.device)
            init_state = enc_final_state[0].unsqueeze(0), enc_final_state[1].unsqueeze(0)
            enc_h_n = enc_final_state[0]
            decode_outputs, switches = [self.start_id], [[0, 0, 0]]
            for _ in range(self.max_decode_steps):
                vocab_probs, next_h, next_c, switch_variable = self.decode_step(enc_states, 
                    enc_h_n, init_state, _input, enc_mask, article_inds)
                probs, inds = vocab_probs.topk(k=1) # probs: [1, 1]
                decode_outputs.append(inds.item())
                switches.append(switch_variable)
                
                if inds.item() == self.stop_id:
                    break
                
                assert inds.size(0) == inds.size(1) == 1
                if inds.max().item() >= self.vocab_size:
                    _input = torch.tensor([[self.unk_id]], device=self.device)
                else:
                    _input = inds.detach()
                init_state = (next_h.unsqueeze(0), next_c.unsqueeze(0))
                
            return decode_outputs, switches