import numpy as np
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F


class CNNEncoder(nn.Module):
    def __init__(self, config):
        super(CNNEncoder, self).__init__()
        self.bsize = config['bsize']
        self.use_cuda = config['use_cuda']
        self.word_emb_dim =  config['word_emb_dim']
        self.enc_hidden_dim = config['enc_hidden_dim']
        window_sizes = config['window_sizes'] if 'window_sizes' in config else [2,3,4]
        
        self.convs = nn.ModuleList([
            nn.Conv2d(1, self.enc_hidden_dim, [window_size, self.word_emb_dim], padding=(window_size-1, 0))
            for window_size in window_sizes
        ])

        
    def forward(self, sent_tuple):
        sent, sent_len = sent_tuple

        # convolution + max pool layer for each window size
        x = sent.contiguous().transpose_(0,1)
        x = torch.unsqueeze(x, 1)       # [B, C, T, E] Add a channel dim.
        xs = []
        for conv in self.convs:
            x2 = F.relu(conv(x))        # [B, F, T, 1]
            x2 = torch.squeeze(x2, -1)  # [B, F, T]
            x2 = F.max_pool1d(x2, x2.size(2))  # [B, F, 1]
            xs.append(x2)
        x = torch.cat(xs, 2)            # [B, F, window]

        x = x.view(x.size(0), -1)       # [B, F * window]
        return x
        

class GRUEncoder(nn.Module):
    def __init__(self, config):
        super(GRUEncoder, self).__init__()
        self.bsize = config['bsize']
        self.word_emb_dim =  config['word_emb_dim']
        self.enc_hidden_dim = config['enc_hidden_dim']
        self.pool_type = config['pool_type']
        self.dpout_model = config['dpout_model']
        self.use_cuda = config['use_cuda']
        self.num_layer = config['num_layer']
        self.use_attention = config['use_attention']
        
        self.enc_lstm = nn.GRU(self.word_emb_dim, self.enc_hidden_dim, self.num_layer, bidirectional=True, dropout=self.dpout_model)
        
        if self.use_attention==1: 
            self.attention = nn.Linear(2*self.enc_hidden_dim, 1)
        elif self.use_attention==2:
            pass
        else:
            pass

        if self.use_cuda:
            self.init_lstm = Variable(torch.FloatTensor(self.num_layer*2, self.bsize, self.enc_hidden_dim).zero_()).cuda()
        else:
            self.init_lstm = Variable(torch.FloatTensor(self.num_layer*2, self.bsize, self.enc_hidden_dim).zero_())
            
        
    def forward(self, sent_tuple):
        # sent_len: [max_len, ..., min_len] (batch)
        # sent: Variable(seqlen x batch x worddim)
        
        sent, sent_len = sent_tuple
        bsize = sent.size(1)
        
        # init hidden state size 
        if bsize != self.init_lstm.size(1):
            self.init_lstm = Variable(torch.FloatTensor(self.num_layer*2, bsize, self.enc_hidden_dim).zero_())
            if self.use_cuda: self.init_lstm = self.init_lstm.cuda() 

        # Sort by length (keep idx)
        sent_len, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        idx_sort = Variable(torch.LongTensor(idx_sort))
        idx_unsort = Variable(torch.LongTensor(np.argsort(idx_sort)))  
        if self.use_cuda: 
            idx_sort = idx_sort.cuda()
            idx_unsort = idx_unsort.cuda()
        sent = sent.index_select(1, idx_sort)
        
        # Handling padding in Recurrent Networks
        sent_packed = pack_padded_sequence(sent, sent_len)
        sent_output, hidden = self.enc_lstm(sent_packed, self.init_lstm)
#         return sent_output, hidden ###### ###### ###### ###### ###### ###### ###### ###### ######
    
        # attention
        if self.use_attention == 0:
            encode = torch.cat(tuple(hidden), 1)
            encode = encode.index_select(0, idx_unsort)
            return encode
        else:
            padded_out_gru, lengths = pad_packed_sequence(sent_output, padding_value=int(0), batch_first=True)
            weight = torch.tanh(torch.squeeze(self.attention(padded_out_gru), 2)) # seq_len x batch_size

            weight = torch.softmax(weight, dim=1)
            weight = pack_padded_sequence(weight, lengths, batch_first=True)
            weight, lengths = pad_packed_sequence(weight, padding_value=0.0, batch_first=True)
            weight = torch.nn.functional.normalize(weight, p=1, dim=1)  # normalizd
            weight = weight.view(weight.size(0), 1, -1)  
            encode = torch.squeeze(weight.bmm(padded_out_gru), 1)
            
            encode = encode.index_select(0, idx_unsort)
            weight = weight.index_select(0, idx_unsort)
                       
            return encode



class NLINet(nn.Module):
    def __init__(self, config):
        super(NLINet, self).__init__()
        
        self.fc_dim = config['fc_dim']
        self.n_classes = config['n_classes']
        self.enc_hidden_dim = config['enc_hidden_dim']
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
        if self.encoder_type=="GRUEncoder":
            self.inputdim = 4*2*config['num_layer']*self.enc_hidden_dim
        elif self.encoder_type=="CNNEncoder":
            self.inputdim = 4*self.enc_hidden_dim*len(config['window_sizes']) if 'window_sizes' in config else self.enc_hidden_dim*3*4
        else:
            pass
        
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