"""
    Embeddings manipulations
"""

from os import path, remove
from copy import copy
import json

from extract  import extract_embeddings
from utils import copy_to_s3

data_folder = 'data'

config_template = {"tensorName": "",
                   "tensorShape": [1000, 50],
                   "tensorPath": "",
                   "metadataPath": "",
                   }


def embeddings_files(data_folder, name):
    """Save embeddings files according to a pattern.  

    Args:  
        data_folder (str): folder to storage embeddings  
        name (str): user embeddings name  
    """
    embeddings_file = f'{data_folder}/{name}_embeddings.tsv'
    vocab_file = f'{data_folder}/{name}_vocab.tsv'

    return embeddings_file, vocab_file


def add_embeddings(name, text, model_name, level_name, **kwargs):
    """Create embeddings and vocabulary list files.  

    Choose name for the embeddings.
    Files are created in data_folder and saved according to name.  

    Args:  
        name (str): user embeddings name  
        text (str): input text  
        model_name (str): Hugging Face model name  
        level_name (str): text segmentation level name  
        **kwargs: extract_embeddings keyword arguments
        
    Raises:  
       FileExistsError: f'File {f} exists.'
    """
    embeddings_file, vocab_file = \
        embeddings_files(data_folder, name)

    for f in [embeddings_file, vocab_file]:
        if path.exists(f):
            raise FileExistsError(f'File {f} exists.')

    extract_embeddings(text, model_name, level_name,
                       embeddings_file, vocab_file, **kwargs)


def remove_embeddings(name):
    """Remove embeddings in data_folder.  

    Args:  
        name (str): user embeddings name  
    """
    embeddings_file, vocab_file = \
        embeddings_files(data_folder, name)

    for f in [embeddings_file, vocab_file]:
        remove(f)


def embeddings_to_s3(bucket_name, name):
    """Copy embeddings files to AWS S3 bucket.  

    Files are copied from data_folder and according to name.  

    Args:  
        bucket_name (str): AWS S3 bucket name  
        name (str): user embeddings name  
    """
    embeddings_file, vocab_file = \
        embeddings_files(data_folder, name)

    copy_to_s3(embeddings_file, bucket_name)
    copy_to_s3(vocab_file, bucket_name)


def publish_embeddings(names, bucket_name, bucket_region, config_filename):
    """Publish selected embeddings in Tensorflow Embedding Projector.  

    Prints publishing link.  
    Selected embeddings are posted on AWS S3 bucket and their links are 
    displayed in configuration file, which is used to create publishing 
    link.

    Args:  
        names (list): selected names  
        bucket_name (str): AWS S3 bucket name  
        bucket_region(str): AWS S3 bucket region  
        config_filename (str): config filename  
    """
    configs = {"embeddings": list()}
    bucket_link = 'https://{bucket_name}.s3.{bucket_region}.amazonaws.com/'

    for name in names:
        embeddings_to_s3(bucket_name, name)
        tensorPath = f'{bucket_link}/{name}_embeddings.tsv'
        metadataPath = f'{bucket_link}/{name}_vocab.tsv'

        config = config_template.copy()
        config['tensorName'] = name
        config['tensorPath'] = tensorPath
        config['metadataPath'] = metadataPath

        configs['embeddings'].append(config)

    with open('config_filename', 'w') as f:
        json.dump(configs)

    copy_to_s3(config_filename, bucket_name)
    config_url = '{bucket_link}/{config_filename}'
    url = f'http://projector.tensorflow.org/?config={config_url}'

    print(url)
