import spacy
from torchtext.legacy.data import  Field,BucketIterator
from torchtext.legacy.datasets import Multi30k


token_en = spacy.load("en_core_web_sm")
token_de = spacy.load("de_core_news_sm")

def tokenizer_de(text):
    return [tok.text for tok in token_de.tokenizer(text)]
def tokenizer_en(text):
    return [tok.text for tok in token_en.tokenizer(text)]

SRC = Field(tokenize=tokenizer_de,init_token="bos",eos_token="eos",lower=True,include_lengths=True)
TGT = Field(tokenize=tokenizer_en,init_token="bos",eos_token="eos",lower=True)

train_data,valid_data,test_data = Multi30k.splits(exts=('.de','.en'),fields=(SRC,TGT))

SRC.build_vocab(train_data,min_freq=2)
TGT.build_vocab(train_data,min_freq=2)





def iterator(batch,device):
    return BucketIterator.splits((train_data,valid_data,test_data),batch_size=batch,device=device,sort_within_batch=True,sort_key=lambda x:len(x.src))

def printexample():
    print(f"Number of training examples: {len(train_data.examples)}")
    print(f"Number of validation examples: {len(valid_data.examples)}")
    print(f"Number of testing examples: {len(test_data.examples)}")
    print(vars(train_data.examples[0]))

def showvocab():
    print(f"Unique tokens in source (de) vocabulary: {len(SRC.vocab)}")
    print(f"Unique tokens in target (en) vocabulary: {len(TGT.vocab)}")


if __name__ == '__main__':
    printexample()
    showvocab()