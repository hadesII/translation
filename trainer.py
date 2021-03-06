import torch

def train(model,iterator,optimizer,criterion,clip):
    model.train()
    epoch_loss = 0

    for i,batch in enumerate(iterator):
        src, src_len= batch.src
        trg = batch.trg
        optimizer.zero_grad()
        output = model(src,src_len,trg)
        output_dim = output.shape[-1]
        output = output[1:].view(-1,output_dim)
        trg = trg[1:].view(-1)
        loss = criterion(output,trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),clip)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss/len(iterator)

def evaluate(model,iterator,criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src, srclen= batch.src
            trg = batch.trg

            output = model(src,srclen,trg,0)
            output_dim = output.shape[-1]
            output = output[1:].view(-1,output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(output,trg)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def init_weights(m):
    for name,param in m.named_parameters():
        if "weight" in name:
            torch.nn.init.normal_(param.data,mean=0,std=0.01)
        else:
            torch.nn.init.constant_(param.data,0)
def count_parameters(model):
    p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"The model has {p} trainable parameters")

def epoch_time(start_time,end_time):
    time = end_time - start_time
    min_time,sec_time = divmod(time,60)
    min_time,sec_time = int(min_time),int(sec_time)
    return min_time,sec_time