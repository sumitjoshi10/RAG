import numpy as np
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

def generate_query_embedding(query: str, embedding_model):
    """ 
    Generates embedding for the input query string using the provided embedding model.
    """
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model)
    return np.array(embedding_model.embed_query(query)).reshape(1, -1)