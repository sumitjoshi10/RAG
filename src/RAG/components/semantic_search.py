from sklearn.metrics.pairwise import cosine_similarity


def semantic_search(chunks, chunk_embeddings, query_embedding, params):
    """
    Perform semantic search to retrieve top-k relevant chunks based on cosine similarity.
    """

    # Compute cosine similarity
    similarity_scores = cosine_similarity(query_embedding, chunk_embeddings)[0]  # shape → (num_chunks,)

    # Get top-k indices
    top_indices = similarity_scores.argsort()[::-1][:params.top_k]
    
    
    # Formate Result
    results = []
    if float(similarity_scores[top_indices][0]) < float(params.similarity_score_limit):
        results.append({
            "content":"No any Matching Context Found for the question",
            "metadata": "Not Available",
            "similarity_score": float(similarity_scores[top_indices][0])
        })
    else:   
        for idx in top_indices:
            results.append({
                "content": chunks[idx].page_content,
                "metadata": chunks[idx].metadata,
                "similarity_score": float(similarity_scores[idx])
            })

    return results