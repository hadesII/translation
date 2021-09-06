import os
import time
import math
import torch
from trainer import count_parameters,epoch_time,train,evaluate,init_weights

from data import SRC,TGT,iterator
from model import Encoder,Decoder,Seq2Seq,Attention

if __name__ == '__main__':
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
    train_iter,valid_iter,_ = iterator(batch,device)



    enc = Encoder(INPUT_DIM,ENC_EMB_DIM,HID_DIM,HID_DIM,dropout)
    dec = Decoder(OUTPUT_DIM,DEC_EMB_DIM,HID_DIM,HID_DIM,dropout)
    atten = Attention(HID_DIM,HID_DIM)

    model = Seq2Seq(enc,dec,atten,device).to(device)
    model.apply(init_weights)
    if os.path.exists("tut3-model.pt"):
        model.load_state_dict(torch.load("tut3-model.pt"))
    count_parameters(model)

    pad_idx = TGT.vocab.stoi[TGT.pad_token]
    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(model.parameters())

    best_valid_loss = float("inf")
    for epoch in range(n_epochs):

        start_time = time.time()
        train_loss = train(model,train_iter,optimizer,criterion,CLIP)
        valid_loss = evaluate(model,valid_iter,criterion)

        end_time = time.time()

        epoch_min,epoch_sec = epoch_time(start_time,end_time)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(),"tut3-model.pt")

        print(f"Epoch:{epoch+1:02} | Time: {epoch_min}m {epoch_sec}s")
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
