from src.rag.constants.constants import *
from src.rag.utility.utils import read_yaml
from src.rag.components.load_data import load_txt_document
from src.rag.components.chunking import chunk_documents


config = read_yaml(CONFIG_FILE_PATH)
params = read_yaml(PARAMS_FILE_PATH)


def rag_pipeline(querry:str ="The querry to be answered"):
    """
    A simple RAG pipeline that takes a querry as input and returns the answer.
    Args:
        querry (str): The querry to be answered.
    """
    ## Step 1: Load Document
    document = load_txt_document(file_path=config.text_document.document_path,
                                 encoding=config.text_document.document_encoding)
    
    ## Step 2: Chunk Document
    print(params.chunking.separators)
    chunked_documents = chunk_documents(document=document,
                                      seperator=params.chunking.separators,
                                      chunk_size=params.chunking.chunk_size,
                                      chunk_overlap=params.chunking.chunk_overlap)
    
    return chunked_documents