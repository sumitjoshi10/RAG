from src.rag.pipeline.rag_pipeline import rag_pipeline


while True:
    question = input("Please type the question or type q to exit: ")
    if question.lower() =='q':
        break
    else:
        answer = rag_pipeline(query=question)[0]
        # print("Number of Embeddings:", len(answer))
        # print("Embeddings Shape:", answer.shape)
        print(f'Question: {question}')
        print(f'Content: {answer["content"]}')
        print(f'Metadata: {answer["metadata"]}')
        print(f'Similarity Score: {answer['similarity_score']}')
        print("="*50)
        print("\n")

