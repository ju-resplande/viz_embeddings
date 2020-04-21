# viz_embeddings
Embeddings projections using TensorFlow Projector and HuggingFace. 

[Example](http://projector.tensorflow.org/?config=https://raw.githubusercontent.com/jubs12/viz_embeddings/master/examples/config.json)

## Requirements
- transformers
- nltk 

## Usage
1. Run extract_embbedings to produce embeddings.tsv and vocab.tsv

``` python
    from extract import extract_embeddings
    
    extract_embeddings(text, model_name, level)
```

- text: string
  Input text

- model_name: string
  HuggingFace model name
 
- level: string

  Represents the segmentation level

  Options: 'word', 'token', 'sentence'

-  aggr_func: torch function (default = torch.mean)

  Pytorch functions to agreggate tokens embedding vectors to form word or array. 
  
 2. Upload embeddings.tsv and vocab.tsv to [Embedding Projector](http://projector.tensorflow.org/)
 
     OR

     - Generate a config.json file as in examples/config.son
     
     - Access link http://projector.tensorflow.org/?config={your config_url}
