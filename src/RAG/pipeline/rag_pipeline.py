from src.rag.constants.constants import *
from src.rag.utility.utils import read_yaml
from src.rag.components.load_data import load_txt_document
from src.rag.components.chunking import chunk_documents
from src.rag.components.embedding import create_embeddings
from src.rag.components.query_embedding import generate_query_embedding
from src.rag.components.semantic_search import semantic_search


config = read_yaml(CONFIG_FILE_PATH)
params = read_yaml(PARAMS_FILE_PATH)


def rag_pipeline(query:str ="The querry to be answered"):
    """
    A simple RAG pipeline that takes a querry as input and returns the answer.
    Args:
        querry (str): The querry to be answered.
    """
    ## Step 1: Load Document
    document = load_txt_document(file_path=config.text_document.document_path,
                                 encoding=config.text_document.document_encoding)
    
    ## Step 2: Chunk Document
    chunked_documents = chunk_documents(document=document,
                                        seperator=params.chunking.separators,
                                        chunk_size=params.chunking.chunk_size,
                                        chunk_overlap=params.chunking.chunk_overlap)
    
    ## Step 3: Create Embeddings
    embeddings = create_embeddings(chunks=chunked_documents,
                                   model_name=params.embedding.model_name)
    
    ## Step 4: Semantic Search
    query_embedding = generate_query_embedding(query, 
                                               embedding_model=params.embedding.model_name)
    
    result = semantic_search(chunks=chunked_documents,
                    chunk_embeddings=embeddings,
                    query_embedding=query_embedding,
                    params=params.semantic_search)
    
    
    return result