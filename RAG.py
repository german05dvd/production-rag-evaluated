''' Phase 4 to build the RAG Model
Join the other parts on one script'''

from indexer import load_model

''' In case you have to build the chroma db for first time uncomment this'''
# from indexer import indexing, init_client, create_collection
# path = 'documentos/tiny-shakespeare.txt'
# size, overlap = 2000, 200
# cliente = init_client()
# coleccion = create_collection(cliente)
# indexing(path, load_model(), coleccion, size, overlap)

from retriever import retrieve, extract
from generator import build_prompt, generate
model = load_model()
collection = extract()

def rag(question: str, k: int = 3) -> dict:
    # 1. Context
    documents = retrieve(question, model, collection, k)
    context = documents["documents"][0]  

    # 2. Build prompt
    prompt = build_prompt(question, context)
    
    # 3. Generate answer
    answer = generate(prompt)
    
    return {
        "question": question,
        "context": context,
        "answer": answer
    }

if __name__ == "__main__":
    pregunta = "What does Hamlet say about death?"
    resultado = rag(pregunta)
    print(f'\nModel: llama-3.2-1b-instruct \nQuestion: {pregunta} \nContext:')
    for i in resultado['context']:
        print(f'{'-'*50} \n{i} \n{'-'*50}')
    print(f'Answer: {resultado['answer']}')