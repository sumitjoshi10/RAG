from langchain_core.documents import Document

def similarity_score(answers, params):
    """
    Perform semantic search to retrieve top-k relevant chunks based on cosine similarity.
    """
    
    # Formate Result
    results = []
    for answer in answers:
        
        if float(answer[1]) < float(params.similarity_score_limit):
            doc = Document(
                page_content= "No matching Content Found",   # content to embed
                metadata={"answer": "Not Available" ,"category": "Not Available", "link": "Not Available",},
                )
            results.append((doc,0.0))
            break
        else:   
           results.append(answer)

    return results