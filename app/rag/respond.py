import os
from functools import lru_cache

from openai import OpenAI

@lru_cache(maxsize=1)
def _client_and_model():
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    if provider == "ollama":
        base = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434").rstrip("/")
        client = OpenAI(base_url=f"{base}/v1", api_key="ollama")
        model = os.getenv("LLM_MODEL", "gemma3:1b")
    else:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        model  = os.getenv("CHAT_MODEL", "gpt-4o-mini")

    return client, model

def respond(prompt: str) -> str:
    client, model = _client_and_model()
    timeout_s = float(os.getenv("LLM_TIMEOUT", "30"))
    c = client.with_options(timeout=timeout_s)

    try:
        resp = c.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=int(os.getenv("MAX_TOKENS", "256"))
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"Model error: {type(e).__name__}: {e}"