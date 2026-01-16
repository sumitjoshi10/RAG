from src.rag.pipeline.rag_pipeline import rag_pipeline

print("Please ask the question Realted to Daraz FAQ.")
print("If you want to quit please press 'q'")
while True:
    question = input(">>> ")
    if question.lower() =='q':
        break
    else:
        answers = rag_pipeline(query=question)
        print(f'Question: {question}')
        for answer in answers:
            # print(answer)
            print(f'Matching Content: {answer[0].page_content}' )
            print(f'Answer: {answer[0].metadata["answer"]}')
            print(f'Metadata: {answer[0].metadata["link"]}')
            print(f'Category: {answer[0].metadata["category"]}')
            print(f'Similarity Score: {answer[1]}')
            print("="*50)
        print("\n")

