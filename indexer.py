''' Phase 1 to build the RAG Model
It's related to the encoder part'''

from sentence_transformers import SentenceTransformer
def load_model(name = "all-MiniLM-L6-v2") -> SentenceTransformer:
    model = SentenceTransformer(name)
    return model

import chromadb
from chromadb import PersistentClient
from chromadb.api.models.Collection import Collection
def init_client(persist: str = "chroma") -> PersistentClient:
    client = chromadb.PersistentClient(path=persist)
    return client
def create_collection(client: PersistentClient, nombre: str = "collection") -> Collection:
    data = client.get_or_create_collection(name = nombre)
    return data

from chunker import get
import os
os.system('clear')
def indexing(path: str, model: SentenceTransformer, collection: Collection, size: int, overlap: int):
    with open(path, 'r') as file:
        text = file.read()
    chunks = get(text, size, overlap)
    print(f'Len chunks: {len(chunks)}')
    embeddings = model.encode(chunks)

    batch, total = 5000, len(chunks)
    ids = [f"id{i}" for i in range(len(chunks))]
    for start in range(0, total, batch):
        end = min(start + batch, total)
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end].tolist(),
            documents=chunks[start:end]
        )
        print(f"Lote {start//batch + 1}: chunks {start}-{end-1} added")

if __name__ == "__main__":
    path = 'documentos/tiny-shakespeare.txt'
    size, overlap = 2000, 200
    cliente = init_client()
    coleccion = create_collection(cliente)
    indexing(path, load_model(), coleccion, size, overlap)