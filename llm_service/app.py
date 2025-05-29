from fastapi import FastAPI
from pydantic import BaseModel
import os
import re

try:
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
    from mistral_inference.transformer import Transformer
    from mistral_inference.generate import generate
    from mistral_common.protocol.instruct.messages import UserMessage
    from mistral_common.protocol.instruct.request import ChatCompletionRequest

    MISTRAL_AVAILABLE = True
except Exception:
    # If the mistral libraries are not installed we fall back to a simple
    # Jaccard similarity implementation for demo purposes.
    MISTRAL_AVAILABLE = False

app = FastAPI()

class ScoreRequest(BaseModel):
    query: str
    text: str

class ScoreResponse(BaseModel):
    score: float

token_re = re.compile(r"\w+")

MODEL_PATH = os.environ.get("MISTRAL_PATH", "")
tokenizer = None
model = None

if MISTRAL_AVAILABLE and MODEL_PATH and os.path.exists(MODEL_PATH):
    try:
        tokenizer = MistralTokenizer.from_file(f"{MODEL_PATH}/tokenizer.model.v3")
        model = Transformer.from_folder(MODEL_PATH)
    except Exception as e:
        print("Failed to load Mistral model:", e)
        tokenizer = None
        model = None

def jaccard_score(a: str, b: str) -> float:
    set_a = set(token_re.findall(a.lower()))
    set_b = set(token_re.findall(b.lower()))
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def mistral_score(query: str, text: str) -> float:
    """Return a relevance score using a local Mistral model if available."""
    if tokenizer is None or model is None:
        return jaccard_score(query, text)

    prompt = (
        "Rate how relevant the following product text is to the query on a "
        "scale from 0 to 1. Only return the numeric score."\
        f"\nQuery: {query}\nProduct: {text}"
    )
    req = ChatCompletionRequest(messages=[UserMessage(content=prompt)])
    tokens = tokenizer.encode_chat_completion(req).tokens
    out_tokens, _ = generate(
        [tokens],
        model,
        max_tokens=8,
        temperature=0.0,
        eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
    )
    result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0]).strip()
    try:
        return float(result.split()[0])
    except Exception:
        return 0.0

@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    return ScoreResponse(score=mistral_score(req.query, req.text))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)
