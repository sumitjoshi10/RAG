from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(document, seperator = ["\n\n", "\n", ".", " "], chunk_size=1000, chunk_overlap=0):
    """
    Chunks the given document into smaller pieces.
    
    Args:
        document: The document to be chunked.
        chunk_size: The size of each chunk.
        chunk_overlap: The overlap between chunks.
        
    Returns:
        A list of chunked documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        separators=seperator,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunked_documents = text_splitter.split_documents(document)
    return chunked_documents