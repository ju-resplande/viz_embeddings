# viz_embeddings

Embeddings projections for TensorFlow Projector using HuggingFace Transformers. 

[Example 1](http://projector.tensorflow.org/?config=https://raw.githubusercontent.com/jubs12/viz_embeddings/master/examples/example%201/config.json), [Example 2](http://projector.tensorflow.org/?config=https://raw.githubusercontent.com/jubs12/viz_embeddings/master/examples/example%202/config.json)

## Requirements
The only requirements are: HuggingFace and NTLK.  

Using pip :

```bash
 pip install -r requirements.txt
```

## Usage
1. Run extract_embbedings to produce embeddings.tsv and vocab.tsv

    ``` python
    from extract import extract_embeddings
    
    extract_embeddings(text, model_name, level_name, aggr_func=torch.mean, filter_func=lambda x: len(x) >= 3, unique = True, doc_stride=1, do_lower=False)
    ```
    Args:
    
    text (str): input text
    
    model_name (str): Hugging Face model name
    
    level_name (str): text segmentation level name
    
    aggr_func (func): function to aggregate tensor embbedings
    
    filter_func (func): filter vocabulary level function
    
    unique (bool): if terms on level_name should be unique
    
    do_lower (bool): when unique = True, whether considere cased
  
 2. View or Publish 

     - Upload embeddings.tsv and vocab.tsv to [Embedding Projector](http://projector.tensorflow.org/)

       OR
     
     - Generate a config.json file as in examples/config.son

     
     - Access link http://projector.tensorflow.org/?config={your config_url}
