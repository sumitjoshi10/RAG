from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


def get_relevant_qa(query:str, collection_name: str, embedding_model: str, chroma_client, k: int):
    
    # Embedding Function for Chroma DB
    ef = HuggingFaceEmbeddings(model_name=embedding_model)
    
    # Creating the vector Database using the Chroma
    vectordb = Chroma(
        collection_name=collection_name,
        embedding_function=ef,
        client=chroma_client     
    )
    
    ### Option 1 Can directly return doing the similarity search
    return vectordb.similarity_search_with_relevance_scores(query, k=k)
    
    retriever = vectordb.as_retriever()
    
    
    
    