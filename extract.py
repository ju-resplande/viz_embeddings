"""Embeddings extraction to files

"""

from enum import Enum
import math

from transformers import AutoModel, AutoTokenizer
import pandas as pd
import nltk
import torch

class Level(Enum):
    """ Input text segmentation levels.
    """

    token = 1
    word = 2
    sentence = 3

def form_token_vocab(tokenizer, text):
    """Separate text in tokens.  

        Args:  
            text (str): input text  
            tokenizer (transformers.AutoTokenizer): HuggingFace Tokenizer  

       Returns:  
          tokens (list) : tokens list
    """

    tokens = tokenizer.tokenize(text)
    return tokens

def form_word_vocab(text):
    """Separate text in words.  

        Args:  
            text (str): input text  

       Returns:  
          words (list) : words list
    """

    nltk.download('punkt')

    words = nltk.word_tokenize(text)
    return words

def form_sentence_vocab(text):
    """Separate text in sentences.  

        Args:  
            text (str): input text  

       Returns:  
          sentences (list) : sentences list
    """

    nltk.download('punkt')

    sentences = nltk.sent_tokenize(text)
    return sentences

def form_token_embeddings(text, tokenizer, model):
    """Generate tokens embeddings list.  

        Args:  
            text (str): input text  
            tokenizer (transformers.AutoTokenizer): HuggingFace Tokenizer  
            model (transformers.AutoModel): HuggingFace Transformer Model  

       Returns:  
            embeddings (torch.tensor): tokens embeddings tensor
    """
    encoded = tokenizer.encode(text, add_special_tokens=False)
    input_ids = torch.tensor(encoded).unsqueeze(0)
    embeddings = model(input_ids)[0][0]

    return embeddings


def form_word_embeddings(text, words, tokenizer, model, pooling):
    """Generate tokens embeddings list.  

        Words composed by more than one token, have tokens embeddings 
        combined in one embedding through pooling.

        Args:  
            text (str): input text  
            words (str): word list  
            tokenizer (transformers.AutoTokenizer): HuggingFace Tokenizer  
            model (transformers.AutoModel): HuggingFace Transformer Model  
            pooling (func): Agreggation Function to form embedding  

       Returns:  
            embeddings (torch.tensor): tokens embeddings tensor
    """

    tokens = form_token_vocab(tokenizer, text)
    tokens_tensor = form_token_embeddings(text, tokenizer, model)

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

    word_embeddings = None
    for mapping in map_idx:
        if isinstance(mapping, int):
            word_embedding = tokens_tensor[mapping].unsqueeze(0)
        elif isinstance(mapping, list):
            #[tokens_tensor[idx] for idx in mapping] for tensor#
            word_embedding = None
            for idx in mapping: 
                token_tensor = tokens_tensor[idx].unsqueeze(0)
                word_embedding = token_tensor if word_embedding == None \
                                 else torch.cat((word_embedding, token_tensor), 0)
            
            word_embedding = pooling(word_embedding, 0).unsqueeze(0)
        else:
            raise TypeError('mapping should be list or int')
        word_embeddings = word_embedding if word_embeddings == None \
                          else torch.cat((word_embeddings, word_embedding), 0) 
    return word_embeddings

def form_sentence_embeddings(sentences, tokenizer, model, pooling):
    """Generate sentence embeddings list.  

        Sentences composed by more than one token, have tokens embeddings 
        combined in one embedding through pooling.  

        Args:  
            sentences (str): sentences list  
            tokenizer (transformers.AutoTokenizer): HuggingFace Tokenizer  
            model (transformers.AutoModel): HuggingFace Transformer Model  
            pooling (func): function to aggregate tensor embeddings  
            
       Returns:  
            embeddings (torch.tensor): tokens embeddings tensor
    """
    sentence_embeddings = None
    for sentence in sentences:
        token_embeddings = form_token_embeddings(sentence, tokenizer, model)
        sentence_embedding = pooling(token_embeddings, 0).unsqueeze(0)
        sentence_embeddings =  sentence_embedding if sentence_embeddings == None \
                            else torch.cat((sentence_embeddings, sentence_embedding), 0) 


    return  sentence_embeddings

def vocabulary_and_embeddings(text, tokenizer, model, level, pooling):
    """Generate embeddings vector and vocabulary  

        Vocabulary is chopped according to level.
        If needed, aggregation function is used.

        Args:  
            text (str): input text  
            tokenizer (transformers.AutoTokenizer): HuggingFace Tokenizer  
            model (transformers.AutoModel): HuggingFace Transformer Model  
            level (str): Segmentation level  
            pooling (func): function to aggregate tensor embeddings  

       Returns:  
            vocab (list): embeddings vocabulary
            embeddings (list): embeddings
    """

    if level == 'token':
        vocab = form_token_vocab(tokenizer, text)
        embeddings = form_token_embeddings(text, tokenizer, model)
    elif level == 'word':
        vocab = form_word_vocab(text)
        embeddings = form_word_embeddings(text, vocab, tokenizer, model, pooling)
    elif level == 'sentence':
        vocab = form_sentence_vocab(text)
        embeddings = form_sentence_embeddings(vocab, tokenizer, model, pooling)

    return vocab, embeddings

def filter_vocabulary(vocab, embeddings, filter_func):
    """Filter vocabulary according to filter_func.  

        Args:  
            vocab(list): embeddings vocabulary  
            embeddings(list): embeddings vector  
            filter_func (func): Filter vocabulary level function  

       Returns:  
            vocab(list): filtered vocabulary  
            embeddings(list): filtered embeddings vocabulary  
    """
    vocab_idx = [idx for idx in range(len(vocab)) if filter_func(vocab[idx])]
    selected_vocab = [vocab[idx] for idx in vocab_idx]
    
    #embeddings = [embeddings[idx] for idx in vocab_idx]
    selected_embedding = None
    for idx in vocab_idx:
        selected = embeddings[idx].unsqueeze(0)

        selected_embedding = selected if selected_embedding == None \
                             else torch.cat((selected_embedding, selected), 0)

    return selected_vocab, selected_embedding

def unique_vocabulary(vocab, embeddings, pooling, do_lower):
    """Combine repeated terms into a unique term.  
    
      Depends on segmentation level on vocab.
      If vocab is chopped in words, words will be considered.
      Hence, it is probably useless in sentences segmentation.

        Args:  
            vocab(list): embeddings vocabulary  
            embeddings(list): embeddings vector  
            pooling (func): torch function to aggregate tensor embeddings  
             do_lower (bool): whether considere cased  

       Returns:  
            vocab(list): embeddings vocabulary with unique values  
            embeddings(list): vocabulary embeddings  
    """
    unique_vocab = list(set([term.lower() for term in vocab])) if do_lower\
                    else list(set(vocab)) 
    

    compare = lambda x, y: x.lower() == y.lower() if do_lower \
              else x == y

    unique_embeddings = None
    for unique_term in unique_vocab:

        unique_embedding = None
        for idx in range(len(vocab)):
            if compare(vocab[idx],unique_term):
                selected = embeddings[idx].unsqueeze(0)              

                unique_embedding = selected if unique_embedding == None \
                                   else  torch.cat((unique_embedding, selected), 0)

        unique_embedding = pooling(unique_embedding, 0).unsqueeze(0)

        unique_embeddings = unique_embedding if unique_embeddings == None \
                            else torch.cat((unique_embeddings, unique_embedding), 0)
    
    return unique_vocab, unique_embeddings

def extract_embeddings(text, model_name, level_name, embeddings_file, vocab_file, pooling=torch.mean, filter_func=lambda x: len(x) >= 3, unique = True, doc_stride=1, do_lower=False):
    """Extract embeddings from text using Hugging Face model.  

        Args:  
            text (str): input text  
            model_name (str): Hugging Face model name  
            level_name (str): text segmentation level name  
            embeddings_file (str): file to save embeddings  
            vocab_file (str): file to save vocabulary  
            pooling (func): torch function to aggregate tensor embeddings Default: torch.mean  
            filter_func (func): filter vocabulary level function. Default: lambda x: len(x) >= 3  
            unique (bool): if terms on level_name should be unique. Default: True  
            doc_stride(int): number of segments to generate on level_name. Default: 1  
            do_lower (bool): when unique = True, whether considere cased. Default: False


       Raises:  
            ValueError: f'There is no {level_name} segmentation.'  
    """

    if level_name not in Level._member_names_:
        raise ValueError(f'There is no {level_name} segmentation.')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
  
    words =  form_word_vocab(text)
    doc_length = math.ceil(len(words)/doc_stride)
    doc_lst = [words[k*doc_length: min((k+1)*doc_length, len(words))]\
                for k in range(doc_stride)]
    doc_lst = [" ".join(doc) for doc in doc_lst]
    vocab = list()
    embeddings = list()

    embeddings = None
    try:
        for doc in doc_lst:
            doc_vocab, doc_embeddings = vocabulary_and_embeddings(doc, 
                                                                tokenizer, 
                                                                model, 
                                                                level_name, 
                                                                pooling)
            vocab.extend(doc_vocab)
            embeddings = doc_embeddings if embeddings == None \
                        else torch.cat((embeddings, doc_embeddings), 0)
    except RuntimeError:
        message = f'Please use doc_stride higher than {doc_stride} for this input'
        print(message)
        return 

    if unique:
        vocab, embeddings = unique_vocabulary(vocab, 
                                              embeddings,
                                              pooling,
                                              do_lower)

    if filter_func:
        vocab, embeddings = filter_vocabulary(vocab, 
                                              embeddings,
                                              filter_func)


    embeddings = embeddings.tolist()

    df = pd.DataFrame(embeddings)
    series = pd.Series(vocab)
    assert df.shape[0] == series.size

    df.to_csv(embeddings_file, index=None, sep='\t', header=None)
    series.to_csv(vocab_file, index=None, sep='\n', header=None)
