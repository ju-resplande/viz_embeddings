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
    
    extract_embeddings(text, model_name, level)
    ```

    - text: string
    
      Input text

    - model_name: string
    
      HuggingFace model name

    - level: string

      Represents the segmentation level. Options: 'word', 'token', 'sentence'


  Pytorch functions to agreggate tokens embedding vectors to form word or sentence. 
  
 2. View or Publish 

     - Upload embeddings.tsv and vocab.tsv to [Embedding Projector](http://projector.tensorflow.org/)

       OR
     
     - Generate a config.json file as in examples/config.son

     
     - Access link http://projector.tensorflow.org/?config={your config_url}
