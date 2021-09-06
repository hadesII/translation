import random
import torch
import torch.nn as nn
class Encoder(nn.Module):

    def __init__(self,input_dim,emb_dim,hid_dim,dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.embed = nn.Embedding(input_dim,emb_dim)

        self.rnn = nn.GRU(input_size=emb_dim,hidden_size=hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self,src):
        embed = self.dropout(self.embed(src))
        _,hidden = self.rnn(embed)

        return hidden


class Decoder(nn.Module):

    def __init__(self,output_dim,emb_dim,hid_dim,dropout):
        super(Decoder, self).__init__()
        self.output_size = output_dim
        self.hid_dim = hid_dim
        self.embed = nn.Embedding(output_dim,emb_dim)
        self.rnn = nn.GRU(input_size=emb_dim+hid_dim,hidden_size=hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hid_dim*2+emb_dim,output_dim)

    def forward(self,input,hidden,context):

        input = input.unsqueeze(0)
        emb = self.dropout(self.embed(input))
        combine = torch.cat((emb,context),dim=-1)
        output,hidden = self.rnn(combine,hidden)
        output = torch.cat((output.squeeze(0),context.squeeze(0),emb.squeeze(0)),dim=-1)
        predict = self.fc(output)
        return predict,hidden

class Seq2Seq(nn.Module):

    def __init__(self,encoder,decoder,device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert self.encoder.hid_dim == self.decoder.hid_dim

    def forward(self,src,trg,teacher_force=0.5):
        batch = trg.shape[1]
        sen_length = trg.shape[0]

        outputs = torch.zeros(sen_length,batch,self.decoder.output_size).to(self.device)

        hidden= self.encoder(src)
        context = hidden
        input = trg[0,:]

        for i in range(1,sen_length):
            output,hidden = self.decoder(input,hidden,context)
            outputs[i] = output
            top1 = output.argmax(1)
            input = trg[i] if random.random() < teacher_force else top1

        return outputs
