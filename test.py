from src.rag.pipeline.rag_pipeline import rag_pipeline


answer = rag_pipeline()
print("Number of Chunks:", len(answer))
for chunk in answer:
    print(chunk)
    print("="*50)
    print("\n")

