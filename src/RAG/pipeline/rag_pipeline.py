from src.rag.constants.constants import *
from src.rag.utility.utils import read_yaml, get_chroma_client

from src.rag.components.faq_data_ingestion import faq_data_ingestion
from src.rag.components.get_relevent_qa import get_relevant_qa
from src.rag.components.similarity_score import similarity_score

## Reading the Config Files
config = read_yaml(CONFIG_FILE_PATH)
params = read_yaml(PARAMS_FILE_PATH)

### Initial Setup to load the chroma Client and Collection Name
persistant_director = config.chroma_db.persistant_memory

chroma_client = get_chroma_client(
    persistant_director    # Uncomment if you want In Memory Chroma DB
    )

collection_name = params.chroma_db.collection_name


def rag_pipeline(query:str):
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
    
    answers = get_relevant_qa(query=query,
                             collection_name= collection_name,
                             embedding_model=params.embedding.model_name,
                             chroma_client = chroma_client,
                             k= params.semantic_search.top_k)
    

    answers = similarity_score(answers=answers, params=params.semantic_search)
    
    return answers
