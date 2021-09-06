import random
import torch
import torch.nn as nn
import torch.nn.functional as F
class Encoder(nn.Module):

    def __init__(self,input_dim,emb_dim,hid_dim,dec_hid,dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.embed = nn.Embedding(input_dim,emb_dim)

        self.rnn = nn.GRU(input_size=emb_dim,hidden_size=hid_dim,bidirectional=True)
        self.fc = nn.Linear(hid_dim*2,dec_hid)
        self.dropout = nn.Dropout(dropout)

    def forward(self,src):
        embed = self.dropout(self.embed(src))
        outputs,hidden = self.rnn(embed)
        hidden = torch.tanh(self.fc(torch.cat((hidden[0,:,:],hidden[1,:,:]),dim=1)))

        return outputs,hidden.unsqueeze(0)


class Attention(nn.Module):

    def __init__(self,enc_hid_dim,dec_hid_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(enc_hid_dim*2+dec_hid_dim,dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim,1,bias=False)

    def forward(self,hidden,enc_outputs):
        src_len = enc_outputs.shape[0]
        enc_outputs = enc_outputs.permute(1,0,2)
        hidden = hidden.unsqueeze(1).repeat(1,src_len,1)

        atten  = torch.tanh(self.attention(torch.cat((enc_outputs,hidden),dim=2)))
        attention = self.v(atten).squeeze(2)

        return F.softmax(attention,dim=1)

class Decoder(nn.Module):

    def __init__(self,output_dim,emb_dim,enc_hid_dim,dec_hid_dim,dropout):
        super(Decoder, self).__init__()
        self.output_size = output_dim
        self.hid_dim = dec_hid_dim
        self.embed = nn.Embedding(output_dim,emb_dim)
        self.rnn = nn.GRU(input_size=emb_dim+enc_hid_dim*2,hidden_size=dec_hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(enc_hid_dim*2+dec_hid_dim+emb_dim,output_dim)

    def forward(self,input,hidden,context):

        input = input.unsqueeze(0)
        emb = self.dropout(self.embed(input))
        output,hidden = self.rnn(torch.cat((emb,context),dim=-1),hidden)
        output,context,emb = output.squeeze(),context.squeeze(),emb.squeeze()
        output = self.fc(torch.cat((context,output,emb),dim=-1))
        return output,hidden



class Seq2Seq(nn.Module):

    def __init__(self,encoder,decoder,attention,device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.attention = attention
        self.device = device

    def forward(self,src,trg,teacher_force=0.5):
        batch = trg.shape[1]
        sen_length = trg.shape[0]

        predicts = torch.zeros(sen_length,batch,self.decoder.output_size).to(self.device)

        outputs,hidden = self.encoder(src)
        input = trg[0,:]

        for i in range(1,sen_length):
            a = self.attention(hidden.squeeze(0),outputs)
            a = a.unsqueeze(1)
            context = torch.bmm(a,outputs.permute(1,0,2))
            output,hidden = self.decoder(input,hidden,context.permute(1,0,2))
            predicts[i] = output
            top1 = output.argmax(1)
            input = trg[i] if random.random() < teacher_force else top1

        return predicts
