import torch
from torchtext.data.metrics import bleu_score
import matplotlib.pyplot as plt
from matplotlib import ticker
import spacy
from data import SRC,TGT,train_data,test_data
from model import model
def translate_sentence(sentence,src_feild,trg_feild,model,device,max_len=50):
    model.eval()

    if isinstance(sentence,str):
        nlp = spacy.load("de")
        sentence = [i.text.lower() for i in nlp(sentence) ]
    else:
        sentence = [i.lower() for i in sentence]

    sentence = [src_feild.init_token ] + sentence + [src_feild.eos_token]
    sentence = [src_feild.vocab.stoi[i] for i in sentence]
    senlen = torch.tensor([len(sentence)])
    sentence = torch.LongTensor(sentence).unsqueeze(1)

    with torch.no_grad():
        enc_outputs,hidden = model.encoder(sentence,senlen)

    outputs = [TGT.vocab.stoi[TGT.init_token]]
    attentions = torch.zeros(max_len,senlen)
    mask = model.create_mask(sentence)

    for i in range(max_len):
        output = torch.LongTensor([outputs[-1]]).to(device)
        with torch.no_grad():
            predict,hidden,attention = model.decoder(output,hidden,enc_outputs,mask)
        predict = predict.argmax(0).item()
        outputs.append(predict)
        attentions[i] = attention.squeeze(0)
        if predict == TGT.vocab.stoi[TGT.eos_token]:
            break
    outputs = [TGT.vocab.itos[i] for i in outputs]

    return outputs[1:],attentions[:len(outputs)-1]



def show_attention(sentence,translation,attention):
    fig= plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    attention = attention.cpu().detach().numpy()
    ax.matshow(attention,cmap="bone")
    ax.tick_params(labelsize=15)
    xtick = [""] + ["<sos>"] + [i for i in sentence] + ["<eos>"]
    ytick = [""] + [i for i in translation]
    ax.set_xticklabels(xtick,rotation=45)
    ax.set_yticklabels(ytick)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.show()
    plt.close()

def calculate_bleu(data,src_feild,trg_feild,model,device,maxlen=50):
    trgs = []
    pred_trgs = []
    for d in data:
        src = d.src
        trg = d.trg
        pred_trg,_ = translate_sentence(src,src_feild,trg_feild,model,device,maxlen)
        pred_trgs.append(pred_trg[:-1])
        trgs.append([trg])

    return bleu_score(pred_trgs,trgs)


if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.load_state_dict(torch.load("tut4-model.pt",map_location=device))
    sentence = train_data.examples[12].src
    tgt,atten = translate_sentence(sentence,SRC,TGT,model,device)
    show_attention(sentence,tgt,atten)
    bleu = calculate_bleu(test_data,SRC,TGT,model,device)
    print(f'BLEU score = {bleu*100:.2f}')