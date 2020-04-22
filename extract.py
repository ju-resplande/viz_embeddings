"""Extractor for TensorFlow Embbeding Projector  

Embbeding are extracted from text using HuggingFace Transformers.
Vocabulary is saved as vocab.tsv.
Embbedings vectors are saved as embedding.tsv.
"""

from enum import Enum
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import nltk
import torch

class Level(Enum):
    """ Input text segmentation levels
    """

    token = 1
    word = 2
    sentence = 3

def form_token_embeddings(text, tokenizer, model):
    """Generate tokense embeddings vector

        Args:  
            text (str): input text  
            tokenizer (transformers.AutoTokenizer): HuggingFace Tokenizer  
            model (transformers.AutoModel): HuggingFace Transformer Model  

       Returns:  
            embeddings: tokens embeddings vector
    """
    encoded = tokenizer.encode(text, add_special_tokens=False)
    input_ids = torch.tensor(encoded).unsqueeze(0)

    return model(input_ids)[0][0]


def form_word_embeddings(text, words, tokenizer, model, aggr_func):
    """Generate tokense embeddings vector

        Args:  
            text (str): input text  
            words (str): word list  
            tokenizer (transformers.AutoTokenizer): HuggingFace Tokenizer  
            model (transformers.AutoModel): HuggingFace Transformer Model  
            aggr_func (func): Agreggation Function to form embedding  

       Returns:  
            embeddings: tokens embeddings vector
    """

    tokens, token_embeddings = vocabulary_and_embeddings(text, tokenizer, model, 'token', None)

    token_idx = 0
    map_idx = list()
    for idx, word in enumerate(words):
        token = tokens[token_idx]

        if word == tokens[token_idx]:
            map_idx.append(idx)
        else:
            word_idx = list()
            while token != word:
                token_idx = token_idx+1
                token += tokens[token_idx].replace('#', "")
                word_idx.append(token_idx)
            map_idx.append(word_idx)
        token_idx = token_idx+1

    word_embeddings = list()
    for idx in map_idx:
        token = token_embeddings[idx]
        word_embedding = token_embeddings[idx] if isinstance(idx) == int \
                         else aggr_func(torch.stack(token), 0)
        word_embeddings.append(word_embedding.tolist())
    return word_embeddings

def form_sentence_embeddings(sentences, tokenizer, model, aggr_func):
    """Generate sentence embeddings vector  

        Args:  
            sentences (str): sentences list  
            tokenizer (transformers.AutoTokenizer): HuggingFace Tokenizer  
            model (transformers.AutoModel): HuggingFace Transformer Model  
            aggr_func (func): Agreggation Function to form embedding  
            
       Returns:  
            embeddings: sentences embeddings vector
    """
    sentence_embeddings = list()
    for sentence in sentences:
        token_embeddings = form_token_embeddings(sentence, tokenizer, model)
        sentence_embeddings.append(aggr_func(token_embeddings, 0).tolist())

    return  sentence_embeddings

def vocabulary_and_embeddings(text, tokenizer, model, level, aggr_func):
    """Generate embeddings vector and vocabulary  

        Args:  
            text (str): input text  
            tokenizer (transformers.AutoTokenizer): HuggingFace Tokenizer  
            model (transformers.AutoModel): HuggingFace Transformer Model  
            level (str): Segmentation level  
            aggr_func (func): Agreggation Function  

       Returns:  
            vocabulary: embeddings vocabulary  
            embeddings: embeddings vector
    """

    nltk.download('punkt')

    if level == 'token':
        vocab = tokenizer.tokenize(text)
        embeddings = form_token_embeddings(text, tokenizer, model)
    elif level == 'word':
        vocab = nltk.word_tokenize(text)
        embeddings = form_word_embeddings(text, vocab, tokenizer, model, aggr_func)
    elif level == 'sentenc':
        vocab = nltk.sent_tokenize(text)
        embeddings = form_sentence_embeddings(vocab, tokenizer, model, aggr_func)


    return vocab, embeddings

def extract_embeddings(text, model_name, level_name, aggr_func=torch.mean):
    """Extract embeddings from text using Hugging Face model

        Embbedings are save as embeddings.tsv.
        Vocabulary is saved as vocabulary.tsv.

            
        Args:  
            text (str): input text  
            model_name (str): Hugging Face model name  
            level_name (str): Text segmentation level name
            aggr_func (func): Agreggation Function  

       Raises:  
            ValueError: f'There is no {level_name} segmentation.'
    """

    if level_name not in Level:
        raise ValueError(f'There is no {level_name} segmentation.')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    vocab, embeddings = vocabulary_and_embeddings(text, tokenizer, model, level_name, aggr_func)

    df = pd.DataFrame(embeddings)
    df.to_csv('embeddings.tsv', index=None, sep='\t', header=None)

    series = pd.Series(vocab)
    series.to_csv('vocab.tsv', index=None, sep='\n', header=None)
