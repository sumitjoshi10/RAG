from box import ConfigBox

import chromadb

from src.rag.utility.utils_helper import resolve_config


import yaml

def read_yaml(path_to_yaml: str) -> ConfigBox:
    """Reads a yaml file and returns a ConfigBox object.
    Args:
        path_to_yaml (str): path like input
    Errors:
        exception: Empty file or any other exception while reading yaml file
    Returns:
        ConfigBox: A ConfigBox object containing the loaded YAML data
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            content = resolve_config(content)
            return ConfigBox(content)
    except Exception as e:
        raise e  
    
def get_chroma_client(persistant_director:str = None):
    '''
    It will
    '''
    if persistant_director is None:
        # Create IN-MEMORY Chroma client
        return chromadb.Client() 
    else:
        # Creat Persistant Client
        return chromadb.PersistentClient(path=persistant_director)