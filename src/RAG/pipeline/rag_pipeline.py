from src.rag.constants.constants import *
from src.rag.utility.utils import read_yaml, get_chroma_client

from src.rag.components.faq_data_ingestion import faq_data_ingestion
from src.rag.components.get_relevent_qa import get_relevant_qa




config = read_yaml(CONFIG_FILE_PATH)
params = read_yaml(PARAMS_FILE_PATH)

### Initial Setup to load the chroma Client and Collection Name
persistant_director = config.chroma_db.persistant_memory

chroma_client = get_chroma_client(
    # persistant_director    # Uncomment if you want persistant directory
    )

collection_name = params.chroma_db.collection_name


def rag_pipeline():
    existing_collections = chroma_client.list_collections()
    print(existing_collections)
    
    if collection_name not in [c.name for c in existing_collections]:
        print("Ingesting FAQ data into Chromadb...")
    
        faq_data_ingestion(path = config.csv_document.document_path,
                        collection_name = collection_name,
                        embedding_model= params.embedding.model_name,
                        chroma_client = chroma_client
                    )
        
        print(f"FAQ Data successfully ingested into Chroma collection: {collection_name}")
        
    else:
        print(f"Collection {collection_name} already exists")
    
    retreiver = get_relevant_qa(collection_name= collection_name,
                             embedding_model=params.embedding.model_name,
                             chroma_client = chroma_client,
                             params= params.semantic_search)
    

    # answers = semantic_search(answers=answers, params=params.semantic_search)
    
    return retreiver
    
# def rag_pipeline(query:str ="The querry to be answered"):
#     """
#     A simple RAG pipeline that takes a querry as input and returns the answer.
#     Args:
#         querry (str): The querry to be answered.
#     """
#     ## Step 1: Load Document
#     document = load_txt_document(file_path=config.text_document.document_path,
#                                  encoding=config.text_document.document_encoding)
    
#     ## Step 2: Chunk Document
#     chunked_documents = chunk_documents(document=document,
#                                         seperator=params.chunking.separators,
#                                         chunk_size=params.chunking.chunk_size,
#                                         chunk_overlap=params.chunking.chunk_overlap)
    
#     ## Step 3: Create Embeddings
#     embeddings = create_embeddings(chunks=chunked_documents,
#                                    model_name=params.embedding.model_name)
    
#     ## Step 4: Semantic Search
#     query_embedding = generate_query_embedding(query, 
#                                                embedding_model=params.embedding.model_name)
    
#     result = semantic_search(chunks=chunked_documents,
#                     chunk_embeddings=embeddings,
#                     query_embedding=query_embedding,
#                     params=params.semantic_search)
    
    
#     return result