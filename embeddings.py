"""
    Embbedings manipulations
"""

from os import path, remove
from subprocess import run
from copy import copy
import json

from extract  import extract_embeddings

data_folder = 'data'

config_template = {"tensorName": "",
                   "tensorShape": [1000, 50],
                   "tensorPath": "",
                   "metadataPath": "",
                   }


def vocab_embedding_files(data_folder, name):
    """Saves vocabulary and embeddings files according to a pattern.  

    Args:  
        data_folder (str): folder to storage embeddings  
        name (str): user embbedings name  
    """
    embeddings_file = f'{data_folder}/{name}_embeddings.tsv'
    vocab_file = f'{data_folder}/{name}_vocab.tsv'

    return embeddings_file, vocab_file


def add_embeddings(name, text, model_name, level_name, **kwargs):
    """Create embeddings and vocabulary list files

    Choose name for the embbedings.  
    Files are created in data_folder and saved according to name.  

    Args:  
        name (str): user embbedings name  
       text (str): input text  
            model_name (str): Hugging Face model name  
            level_name (str): text segmentation level name  
        **kwargs: extract_embeddings keyword arguments
    """
    embeddings_file, vocab_file = \
        vocab_embedding_files(data_folder, name)

    for f in [embeddings_file, vocab_file]:
        if path.exists(f):
            raise FileExistsError(f'File {f} exists.')

    extract_embeddings(text, model_name, level_name,
                       embeddings_file, vocab_file, **kwargs)


def remove_embeddings(name):
    """Remove embbedings for data_folder.

    Args:  
        name (str): user embbedings name  
    """
    embeddings_file, vocab_file = \
        vocab_embedding_files(data_folder, name)

    for f in [embeddings_file, vocab_file]:
        remove(f)


def copy_to_s3(filename, bucket_s3):
    """Copy file to AWS S3 bu bucket  

    Args:  
        filename (str): file name to copy to S3  
        bucket_s3 (str): AWS S3 bucket url  
    """
    cmds = ['aws', 's3', 'cp', filename, f's3://{bucket_s3}/']
    run(cmds)


def vocab_embeddings_to_s3(bucket_s3, name):
    """Copy embeddings and vocabulaty files to AWS S3 bucket  

    Files are copied from data_folder and according to name.  

    Args:  
        bucket_s3 (str): AWS S3 bucket url  
        name (str): user embbedings name  
    """
    embeddings_file, vocab_file = \
        vocab_embedding_files(data_folder, name)

    copy_to_s3(embbedings_file, bucket_s3)
    copy_to_s3(vocab_file, bucket_s3)


def set_config_file(bucket_s3, names, config_filename):
    """Set configuration file for publish in Tensorflow Embedding Projector.  

    Selected embbedings are posted on AWS S3 bucket and their links are 
    displayed in configuration file, which is used to create publising 
    link.

    Args:  
        bucket_s3 (str): AWS S3 bucket url  
        name (lst): selected names  
        config_filename (str): config filename  
    """
    configs = {"embeddings": list()}

    for name in names:
        vocab_embeddings_to_s3(bucket_s3, name)
        tensorPath = f's3://{bucket_s3}/{name}_embeddings.tsv'
        metadataPath = f's3://{bucket_s3}/{name}_vocab.tsv'

        config = config_template.copy()
        config['tensorName'] = name
        config['tensorPath'] = tensorPath
        config['metadataPath'] = metadataPath
        configs['embeddings'].append(config)

    with open('config_filename', 'w') as f:
        json.dump(configs)

    copy_to_s3(config_filename, bucket_s3)
    config_url = 's3://{bucket_s3}/{config_filename}'
    url = f'http://projector.tensorflow.org/?config={config_url}'

    print(url)
