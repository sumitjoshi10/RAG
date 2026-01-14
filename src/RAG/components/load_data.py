from langchain_community.document_loaders import TextLoader

def load_txt_document(file_path: str, encoding: str = "utf-8"):
    """
    Loads a text file and returns LangChain Document objects
    """
    
    loader = TextLoader(file_path, encoding=encoding)
    documents = loader.load()
    return documents