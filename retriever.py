''' Phase 2 to build the RAG Model
It's related to the decoder part'''
from sentence_transformers import SentenceTransformer
import chromadb

def extract(persist: str='chroma', nombre: str='collection'):
    client = chromadb.PersistentClient(path=persist)
    return client.get_collection(name=nombre)

def retrieve(query, model, collection, k=3):
    embedding = model.encode([query])  # Array (1, 384)
    results = collection.query(
        query_embeddings=embedding.tolist(),
        n_results=k
    )
    return results

from indexer import load_model
if __name__ == "__main__":
    model = load_model()
    collection = extract()
    pregunta = "What does Hamlet say about death?"
    print(f"Results from retriever process. \nQuestion: {pregunta} \n\nAnswers: \n{"-"*50}")
    respuesta = retrieve(pregunta, model, collection, k=3)
    for i, (documento, distancia) in enumerate(zip(respuesta["documents"][0], respuesta["distances"][0])):
        print(f"Vector {i+1} (distance {distancia:.4f}):")
        print(f"{documento} \n{"-"*50}")
