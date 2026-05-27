''' Phase 3 to build the RAG Model
It's related to the generation part'''
import requests

# I use llama-3.2-1b-instruct running in local on LM Studio

def build_prompt(query: str, context: list[str]) -> str:
    text = "\n\n".join(context)
    return f"""You are an expert in classic literature, specifically Shakespeare's works.
    Answer the QUESTION using ONLY the provided CONTEXT.
    If the answer is not in the context, say that you don't know.

    CONTEXT:
    <context>
    {text}
    </context>

    QUESTION: {query}

    ANSWER:
    """

def generate(prompt: str, model: str = "llama-3.2-1b-instruct") -> str:
    url = "http://127.0.0.1:1234/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,  # Low Temperature for precission
        "max_tokens": 512,
        "stream": False
    }
    
    response = requests.post(url, json=payload)
    response.raise_for_status()
    
    return response.json()["choices"][0]["message"]["content"]