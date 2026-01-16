from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


def get_relevant_qa(collection_name: str, embedding_model: str, chroma_client, params):
    
    # Embedding Function for Chroma DB
    ef = HuggingFaceEmbeddings(model_name=embedding_model)
    
    # Creating the vector Database using the Chroma
    vectordb = Chroma(
        collection_name=collection_name,
        embedding_function=ef,
        client=chroma_client     
    )
    
    
    retriever = vectordb.as_retriever(
        search_type=params.search_type, 
        search_kwargs={"score_threshold": params.similarity_score_limit,
                       "k" : params.top_k}
    )

    return retriever
    
    
    
    