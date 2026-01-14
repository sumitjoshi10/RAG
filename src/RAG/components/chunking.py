from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(document, seperator = ["\n\n", "\n", ".", " "], chunk_size=1000, chunk_overlap=0):
    """
    Chunks the given document into smaller pieces.
    """
    
    text_splitter = RecursiveCharacterTextSplitter(
        separators=seperator,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunked_documents = text_splitter.split_documents(document)
    return chunked_documents