''' Phase 6 to build the RAG Model
Close with an API REST to fill the RAG pipeline '''

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

import RAG as module

app = FastAPI(title="Shakespeare RAG API", description="API for querying the Shakespeare corpus using local RAG")
class QueryRequest(BaseModel):
    question: str
    k: Optional[int] = 3

class QueryResponse(BaseModel):
    question: str
    answer: str
    context: List[str]

@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    """ Receive a question, retrieve the most relevant fragments
        and generate an answer using the local LLM."""
    try:
        result = module.rag(req.question, k=req.k)
        return QueryResponse(
            question=result["question"],
            answer=result["answer"],
            context=result["context"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Problem detected on RAG: \n{str(e)}")

import os
import shutil
from fastapi import UploadFile, File, Form
from indexer import indexing

@app.post("/index")
async def index_endpoint(
    file: UploadFile = File(...),
    size: int = Form(1000),
    overlap: int = Form(100)
):
    if not file.filename.endswith('.txt'):
        raise HTTPException(status_code=400, detail="Only .txt files")

    temporal = f"temp_{file.filename}"
    try:
        with open(temporal, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        from RAG import model, collection

        indexing(
            path=temporal,
            model=model,
            collection=collection,
            size=size,
            overlap=overlap
        )
        return {"message": f"File '{file.filename}' indexed."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Problem detected on the indexing: {str(e)}")
    finally:
        if os.path.exists(temporal):
            os.remove(temporal)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)