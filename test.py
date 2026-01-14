from src.rag.pipeline.rag_pipeline import rag_pipeline


answer = rag_pipeline()
print("Number of Embeddings:", len(answer))
for embedding in answer:
    print(embedding)
    print("="*50)
    print("\n")

