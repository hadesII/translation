import random
import torch
import torch.nn as nn
class Encoder(nn.Module):

    def __init__(self,input_dim,emb_dim,hid_dim,n_layers,dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embed = nn.Embedding(input_dim,emb_dim)

        self.rnn = nn.LSTM(input_size=emb_dim,hidden_size=hid_dim,num_layers=n_layers,dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self,src):
        embed = self.dropout(self.embed(src))
        _,(hidden,cell) = self.rnn(embed)

        return hidden,cell


class Decoder(nn.Module):

    def __init__(self,output_dim,emb_dim,hid_dim,n_layers,dropout):
        super(Decoder, self).__init__()
        self.output_size = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embed = nn.Embedding(output_dim,emb_dim)
        self.rnn = nn.LSTM(input_size=emb_dim,hidden_size=hid_dim,num_layers=n_layers,dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hid_dim,output_dim)

    def forward(self,input,hidden,cell):

        input = input.unsqueeze(0)
        emb = self.dropout(self.embed(input))
        output,(hidden,cell) = self.rnn(emb,(hidden,cell))
        predict = self.fc(output.squeeze(0))
        return predict,hidden,cell

class Seq2Seq(nn.Module):

    def __init__(self,encoder,decoder,device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert self.encoder.hid_dim == self.decoder.hid_dim
        assert self.encoder.n_layers == self.decoder.n_layers

    def forward(self,src,trg,teacher_force=0.5):
        batch = trg.shape[1]
        sen_length = trg.shape[0]

        outputs = torch.zeros(sen_length,batch,self.decoder.output_size).to(self.device)

        hidden,cell = self.encoder(src)
        input = trg[0,:]

        for i in range(1,sen_length):
            output,hidden,cell = self.decoder(input,hidden,cell)
            outputs[i] = output
            top1 = output.argmax(1)
            input = trg[i] if random.random() < teacher_force else top1

        return outputs
