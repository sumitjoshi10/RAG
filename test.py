from src.rag.pipeline.rag_pipeline import rag_pipeline

retreiver = rag_pipeline()

print("Please ask the question Realted to Daraz FAQ.")
print("If you want to quit please press 'q'")
    
while True:
    question = input(">>> ")
    if question.lower() =='q':
        break
    else:
        answers = retreiver.invoke(question)
        print(f'Question: {question}')
        for answer in answers:
            # print(answer)
            print(f'Matching Content: {answer.page_content}' )
            print(f'Answer: {answer.metadata["answer"]}')
            print(f'Metadata: {answer.metadata["link"]}')
            print(f'Category: {answer.metadata["category"]}')
            print("="*50)
        print("\n")

