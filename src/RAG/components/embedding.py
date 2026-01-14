from langchain_huggingface.embeddings import HuggingFaceEmbeddings

def create_embeddings(chunks, model_name):
    """
    Converts text chunks into embeddings
    """
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)

    texts = [chunk.page_content for chunk in chunks]
    embeddings = embedding_model.embed_documents(texts)

    return embeddings