import numpy as np
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class GRUEncoder(nn.Module):
    def __init__(self, config):
        super(GRUEncoder, self).__init__()
        self.bsize = config['bsize']
        self.word_emb_dim =  config['word_emb_dim']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.pool_type = config['pool_type']
        self.dpout_model = config['dpout_model']
        self.use_cuda = config['use_cuda']
        self.num_layer = config['num_layer']
        self.final_hidden_attention = config['final_hidden_attention']
        
        self.enc_lstm = nn.GRU(self.word_emb_dim, self.enc_lstm_dim, self.num_layer, bidirectional=True, dropout=self.dpout_model)
        
        if self.final_hidden_attention: self.attention = nn.Linear(2*self.enc_lstm_dim, 1)

        
        if self.use_cuda:
            self.init_lstm = Variable(torch.FloatTensor(self.num_layer*2, self.bsize, self.enc_lstm_dim).zero_()).cuda()
        else:
            self.init_lstm = Variable(torch.FloatTensor(self.num_layer*2, self.bsize, self.enc_lstm_dim).zero_())


    def forward(self, sent_tuple):
        # sent_len: [max_len, ..., min_len] (batch)
        # sent: Variable(seqlen x batch x worddim)
        
        sent, sent_len = sent_tuple
        bsize = sent.size(1)
        
        if bsize != self.init_lstm.size(1):
            if self.use_cuda:
                self.init_lstm = Variable(torch.FloatTensor(self.num_layer*2, bsize, self.enc_lstm_dim).zero_()).cuda()
            else:
                self.init_lstm = Variable(torch.FloatTensor(self.num_layer*2, bsize, self.enc_lstm_dim).zero_())
        # Sort by length (keep idx)
        sent_len, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        if self.use_cuda:
            sent = sent.index_select(1, Variable(torch.cuda.LongTensor(idx_sort)))
        else:
            sent = sent.index_select(1, Variable(torch.LongTensor(idx_sort)))
        
        # Handling padding in Recurrent Networks
        sent_packed = pack_padded_sequence(sent, sent_len)
        sent_output, hidden = self.enc_lstm(sent_packed, self.init_lstm)
        
        if not self.final_hidden_attention:
            #### use last hidden state ####
            emb = torch.cat((hidden[0], hidden[1]), 1) # batch x 2*nhid
        else:
            ##### use attention #####
            padded_out_gru, lengths = pad_packed_sequence(sent_output, padding_value=int(0), batch_first=True)
            unnormalize_weight = torch.tanh(torch.squeeze(self.attention(padded_out_gru), 2)) # seq_len x batch_size
            unnormalize_weight = torch.softmax(unnormalize_weight, dim=1)
            unnormalize_weight = pack_padded_sequence(unnormalize_weight, lengths, batch_first=True)
            unnormalize_weight, lengths = pad_packed_sequence(unnormalize_weight, padding_value=0.0, batch_first=True)
            normalize_weight = torch.nn.functional.normalize(unnormalize_weight, p=1, dim=1)
            normalize_weight = normalize_weight.view(normalize_weight.size(0), 1, -1)
            emb = torch.squeeze(normalize_weight.bmm(padded_out_gru), 1)

        idx_unsort = np.argsort(idx_sort)
        if self.use_cuda:
            emb = emb.index_select(0, Variable(torch.cuda.LongTensor(idx_unsort)))
        else:
            emb = emb.index_select(0, Variable(torch.LongTensor(idx_unsort)))
            
        return emb
    


class NLINet(nn.Module):
    def __init__(self, config):
        super(NLINet, self).__init__()
        
        self.fc_dim = config['fc_dim']
        self.n_classes = config['n_classes']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.encoder_type = config['encoder_type']
        self.dpout_fc = config['dpout_fc']
        self.dpout_embed = config['dpout_embed']
        embed_freeze = config['embed_freeze']
        embed_matrix = config['embed_matrix']
        vocab_size, embed_size = embed_matrix.shape
        
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.embed.weight = nn.Parameter(torch.from_numpy(embed_matrix).type(torch.FloatTensor), requires_grad=not embed_freeze)
        self.embed_dropout = nn.Dropout(self.dpout_embed)
        
        self.encoder = eval(self.encoder_type)(config)
        self.inputdim = 8*self.enc_lstm_dim
        
        # classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dpout_fc),
            nn.Linear(self.inputdim, self.fc_dim),
            nn.Tanh(),
            nn.Dropout(p=self.dpout_fc),
            nn.Linear(self.fc_dim, self.n_classes),
            nn.Softmax(dim=-1),
        )


    def forward(self, s1, s2):
        embed1 = self.embed_dropout(self.embed(s1[0])).transpose_(0, 1)
        embed2 = self.embed_dropout(self.embed(s2[0])).transpose_(0, 1)

        u = self.encoder((embed1, s1[1]))
        v = self.encoder((embed2, s2[1]))
        features = torch.cat((u, v, torch.abs(u-v), u*v), 1)
        output = self.classifier(features)
        return output
    
    
    def encode(self, s1):
        embed1 = self.embed_dropout(self.embed(s1[0])).transpose_(0, 1)
        encode = self.encoder((embed1, s1[1]))
        return encode