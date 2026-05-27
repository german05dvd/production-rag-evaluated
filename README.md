# Production RAG Evaluated

> Retrieval-Augmented Generation system built from scratch with Python.
> Local embeddings, local LLM, local vector store. No OpenAI, no cloud.

## What I built

A complete RAG pipeline that answers questions about a corpus using retrieved context:

- **Chunking**: Recursive splitting with hierarchical separators and overlap
- **Embeddings**: `all-MiniLM-L6-v2` (384 dims) via SentenceTransformers
- **Vector store**: ChromaDB persistente
- **Retrieval**: Dense semantic search with cosine similarity
- **Generation**: Local LLM via LM Studio (llama-3.2-1b-instruct)
- **Evaluation**: 10-question dataset with retrieval and generation metrics
- **API**: FastAPI endpoint for querying and indexing new documents

## Dataset

**tiny-shakespeare.txt**: Complete works of Shakespeare (~1 million characters).

Place in `documentos/` directory.

## Requirements

```bash
pip install -r requirements.txt
```

Dependencies: `sentence-transformers`, `chromadb`, `fastapi`, `uvicorn`, `requests`, `numpy`, `pydantic`

## Repository structure

```
production-rag-evaluated/
├── chunker.py           # Recursive text splitting with overlap
├── indexer.py           # Load model, init Chroma, index documents
├── retriever.py         # Connect to collection, retrieve top-k chunks
├── generator.py         # Build prompt, call local LLM via LM Studio
├── RAG.py               # Orchestrator: query → retrieve → generate
├── metrics.py           # Evaluation on 10-question dataset
├── app.py               # FastAPI: /query and /index endpoints
├── requirements.txt
├── .gitignore
├── chroma/              # Persistent vector DB (generated)
├── documentos/
│   └── tiny-shakespeare.txt
└── README.md
```

## How to run

### 1. Start LM Studio

- Download [LM Studio](https://lmstudio.ai/)
- Load `llama-3.2-1b-instruct` (or any local model)
- Start server on `http://127.0.0.1:1234`

### 2. Index corpus

```bash
python indexer.py
```

Configuration (edit `indexer.py`):
| Parameter | Default | Description |
|-----------|---------|-------------|
| `size` | 2000 | Target chunk size (characters) |
| `overlap` | 200 | Overlap between consecutive chunks |

Outputs:
- `chroma/` folder with indexed embeddings
- ~2,000-3,000 chunks (varies with parameters)

### 3. Query

```bash
python RAG.py
```

Or use the API:
```bash
python app.py
# POST to http://localhost:8000/query
# Body: {"question": "What does Hamlet say about death?", "k": 3}
```

### 4. Evaluate

```bash
python metrics.py
```

Runs 10 questions through the pipeline and reports metrics.

## Architecture

```
Document (txt)
  |
chunker.py: recursive split + overlap
  |
indexer.py: embed (MiniLM) → store (Chroma)
  |
Query → retriever.py: dense search (cosine similarity)
  |
generator.py: build prompt → call LLM (LM Studio)
  |
Answer
```

## Key implementation decisions

- **Recursive chunking**: Splits by `\n\n`, `\n`, `. `, ` `, then by character. Respects target size better than fixed chunking.
- **Local everything**: No API keys, no internet required after model download. Designed for offline/restricted environments.
- **Batch insertion**: Chroma has max batch size (5461). Indexed in batches of 5000.
- **English prompts**: Model (llama-3.2-1b) performs better with English prompts than Spanish, despite Spanish queries.
- **Low temperature (0.2)**: Reduces hallucination when context is present.

## Limitations

- **Small LLM**: 1B parameters. Struggles with extraction even when context is correct. Often answers "I don't know" or hallucinates.
- **No metadata filtering**: Cannot restrict search to specific plays (e.g., only Hamlet).
- **Character-level chunking**: Not semantic. May split mid-sentence or mid-argument.
- **No re-ranking**: Retrieved chunks ordered by embedding similarity only. No cross-encoder refinement.
- **Single corpus**: Trained/indexed on Shakespeare only. No generalization.
- **LM Studio dependency**: Requires local server running. No fallback if server is down.

## What I learned

- **Chunking matters**: Initial attempt produced 14,492 micro-chunks. Fixing the splitter reduced to ~2,000 usable chunks with better context preservation.
- **Prompt language**: A Spanish prompt with English corpus produced mixed-language, confused responses. Switching to English prompts fixed this.
- **Evaluation is hard**: Keyword matching and substring search are crude metrics. A response can be correct without containing the exact expected phrase.
- **Small models are brittle**: 1B parameters can retrieve but not reliably extract. The retriever is solid; the generator is the bottleneck.
- **Local RAG is viable**: Full pipeline runs on CPU i3 without cloud dependencies. Slower than GPU, but functional.

## Stack

Python 3 · SentenceTransformers · ChromaDB · FastAPI · LM Studio

## Results

### Retrieval metrics (k=10)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Recall@10 (keywords) | 90% | At least one keyword found in retrieved chunks |
| Recall@10 (exact answer) | 60% | Exact expected phrase present in retrieved chunks |

**Analysis**: The retriever (dense search with MiniLM) finds relevant context most of the time. The 60% exact-match rate reflects that Shakespeare's text rarely contains modern paraphrases of the expected answers.

### Generation metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Exact generation (substring) | 20% | Generated answer contains exact expected phrase |
| Flexible generation (keywords) | 0% | Generated answer contains all keywords (too strict) |

**Analysis**: The 1B model extracts precise information poorly. It answers correctly in simple cases ("Yes", "Caius Marcius") but fails on questions requiring inference or exact quotation from context.

### Example outputs

| Question | Expected | Generated | Verdict |
|----------|----------|-----------|---------|
| Who is the "chief enemy to the people"? | Caius Marcius | "Caius Marcius." | ✅ Correct |
| Does Volumnia prefer her son die nobly? | Yes | "Yes." | ✅ Correct |
| What new name for Caius Marcius? | CORIOLANUS! | "I don't know." | ❌ Failed extraction |
| Who does Richard III suborn? | Tyrrel | "Buckingham." | ❌ Hallucination |
| How many years in Winter's Tale gap? | Sixteen | "six winters" | ⚠️ Close but wrong |

### Conclusion

The RAG pipeline demonstrates solid retrieval (90% keyword recall) but is bottlenecked by the 1B parameter generator. For production use, a larger local model (3-7B parameters) or API-based LLM would significantly improve answer accuracy.

## Contact

German David Rojas Lam · Havana, Cuba · github.com/german05dvd
