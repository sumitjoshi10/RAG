import pandas as pd

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings



def faq_data_ingestion(path: str, collection_name: str, embedding_model: str, chroma_client):
    
    # Embedding Function for Chroma DB
    ef = HuggingFaceEmbeddings(model_name=embedding_model)
    
    
    # Load FAQ CSV
    df = pd.read_csv(path)
    
    # Convert each row into a LangChain "Document"
    documents = []
    for idx, row in df.iterrows():
        doc = Document(
            page_content=row["question"],   # content to embed
            metadata={"answer": row["answer"],"category": row["category"], "link": row["link"], "id": f"id_{idx}"},
        )
        documents.append(doc)
    
    # Create the Chroma  (collection) and insert data
    Chroma.from_documents(
        documents=documents,
        embedding=ef,
        collection_name=collection_name,
        client=chroma_client                # <-- use in-memory client
    )
    

    