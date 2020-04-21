from transformers import AutoModel, AutoTokenizer
from enum import Enum
import pandas as pd
import nltk
import torch

class Level(Enum):
    token = 1
    word = 2
    sentence = 3

def vocabulary_and_embeddings(text, tokenizer, model, level, aggr_func):    
    nltk.download('punkt')

    if level in ['token', 'word']:
        encoded = tokenizer.encode(text,add_special_tokens=False) #don't use [CLS] and [SEP]
        input_ids = torch.tensor(encoded).unsqueeze(0)

        vocab = tokenizer.tokenize(text)
        embeddings = model(input_ids)[0][0]
        
        if level == 'token':
            return vocab, embeddings.tolist()
        elif level == 'word':
            words = nltk.word_tokenize(text)

            vocab_idx = 0
            map_idx = []
            selected_words = []
            for idx, word in enumerate(words):
                if word == vocab[vocab_idx]:
                    if len(word) > 1: #ignore punctuations
                        map_idx.append(idx)
                        selected_words.append(word)
                else:
                    word_idx = []
                    term = vocab[vocab_idx]      
                    while(word != term):
                        vocab_idx = vocab_idx+1
                        term+= vocab[vocab_idx].replace('#', "")
                        word_idx.append(vocab_idx)
                    map_idx.append(word_idx)
                    selected_words.append(word)
                
                vocab_idx = vocab_idx+1
            selected_embeddings = []
            for idx in map_idx:
                if type(idx) == int:
                    selected_embeddings.append(embeddings[idx].tolist())
                elif type(idx) == list:
                    word_embbedings = [embeddings[i] for i in idx]
                    mean = aggr_func(torch.stack(word_embbedings), 0)
                    selected_embeddings.append(mean.tolist())
            
            return  selected_words, selected_embeddings
    elif level == 'sentence':
        sentences = nltk.sent_tokenize(text)
        embeddings_list = []
        for sentence in sentences:
            encoded = tokenizer.encode(sentence,add_special_tokens=False) #don't use [CLS] and [SEP]
            input_ids = torch.tensor(encoded).unsqueeze(0)
            embeddings = model(input_ids)[0][0]

            embeddings_list.append(aggr_func(embeddings, 0).tolist())

        return sentences, embeddings_list

def extract_embeddings(text, model_name, level_name, **kargs):    
    assert Level[level_name]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    aggr_func = kargs['aggr_func'] if 'aggr_func' in kargs.keys() else torch.mean
   
    vocab, embeddings = vocabulary_and_embeddings(text, tokenizer, model, level_name, aggr_func)   

    df = pd.DataFrame(embeddings) 
    df.to_csv('embeddings.tsv', index = None, sep = '\t', header = None)

    series = pd.Series(vocab)
    series.to_csv('vocab.tsv', index = None, sep = '\n', header = None)
