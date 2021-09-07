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

    def forward(self,src,srclen):
        embed = self.dropout(self.embed(src))
        packed_embed = nn.utils.rnn.pack_padded_sequence(embed,srclen.to(torch.device("cpu")))
        outputs,hidden = self.rnn(packed_embed)
        outputs ,_= nn.utils.rnn.pad_packed_sequence(outputs)
        hidden = torch.tanh(self.fc(torch.cat((hidden[0,:,:],hidden[1,:,:]),dim=1)))

        return outputs,hidden.unsqueeze(0)


class Attention(nn.Module):

    def __init__(self,enc_hid_dim,dec_hid_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(enc_hid_dim*2+dec_hid_dim,dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim,1,bias=False)

    def forward(self,hidden,enc_outputs,mask):
        src_len = enc_outputs.shape[0]
        enc_outputs = enc_outputs.permute(1,0,2)
        hidden = hidden.squeeze(0).unsqueeze(1).repeat(1,src_len,1)

        atten  = torch.tanh(self.attention(torch.cat((enc_outputs,hidden),dim=2)))
        attention = self.v(atten).squeeze(2)
        attention = attention.masked_fill_(mask==0,-1e10)
        return F.softmax(attention,dim=1)

class Decoder(nn.Module):

    def __init__(self,output_dim,attention,emb_dim,enc_hid_dim,dec_hid_dim,dropout):
        super(Decoder, self).__init__()
        self.output_size = output_dim
        self.hid_dim = dec_hid_dim
        self.embed = nn.Embedding(output_dim,emb_dim)
        self.attention = attention
        self.rnn = nn.GRU(input_size=emb_dim+enc_hid_dim*2,hidden_size=dec_hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(enc_hid_dim*2+dec_hid_dim+emb_dim,output_dim)

    def forward(self,input,hidden,enc_outputs,mask):

        input = input.unsqueeze(0)
        emb = self.dropout(self.embed(input))
        weight = self.attention(hidden,enc_outputs,mask)
        weight = weight.unsqueeze(1)
        enc_outputs = enc_outputs.permute(1,0,2)
        context = torch.bmm(weight,enc_outputs).permute(1,0,2)
        output,hidden = self.rnn(torch.cat((emb,context),dim=-1),hidden)
        output,context,emb = output.squeeze(),context.squeeze(),emb.squeeze()
        output = self.fc(torch.cat((context,output,emb),dim=-1))
        return output,hidden,weight



class Seq2Seq(nn.Module):

    def __init__(self,encoder,decoder,src_pad_id,device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.src_pad_id = src_pad_id

    def create_mask(self,src):
        return (src != self.src_pad_id).permute(1,0)
    def forward(self,src,src_len,trg,teacher_force=0.5):
        batch = trg.shape[1]
        sen_length = trg.shape[0]

        predicts = torch.zeros(sen_length,batch,self.decoder.output_size).to(self.device)

        outputs,hidden = self.encoder(src,src_len)
        input = trg[0,:]
        mask = self.create_mask(src)

        for i in range(1,sen_length):
            output,hidden,_= self.decoder(input,hidden,outputs,mask)
            predicts[i] = output
            top1 = output.argmax(1)
            input = trg[i] if random.random() < teacher_force else top1

        return predicts
from data import SRC, TGT

INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TGT.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch = 128
n_epochs = 10
CLIP = 1
dropout = 0.5

src_pad_id = SRC.vocab.stoi[SRC.pad_token]
atten = Attention(HID_DIM, HID_DIM)
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, HID_DIM, dropout)
dec = Decoder(OUTPUT_DIM, atten, DEC_EMB_DIM, HID_DIM, HID_DIM, dropout)

model = Seq2Seq(enc, dec, src_pad_id, device).to(device)
