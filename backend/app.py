from fastapi import FastAPI, Query
from pydantic import BaseModel
import os
import csv
import re
import math
from collections import Counter
import json
import urllib.request

app = FastAPI()

DATA_PATH = os.environ.get("DATA_PATH", "/data/products.csv")
LLM_URL = os.environ.get("LLM_URL", "http://localhost:8100")


def load_products(path: str):
    products = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            products.append({
                "product_id": int(row["product_id"]),
                "name": row["name"],
                "description": row["description"],
                "price": float(row.get("price", 0)),
            })
    return products


products = load_products(DATA_PATH)

token_re = re.compile(r"\w+")

def tokenize(text: str):
    return token_re.findall(text.lower())

doc_freq: Counter[str] = Counter()
for p in products:
    terms = set(tokenize(f"{p['name']} {p['description']}"))
    for t in terms:
        doc_freq[t] += 1

N_DOCS = len(products)
idf = {t: math.log(N_DOCS / (1 + df)) for t, df in doc_freq.items()}


def vectorize_text(text: str):
    tokens = tokenize(text)
    tf = Counter(tokens)
    if not tokens:
        return {}
    return {t: (tf[t] / len(tokens)) * idf.get(t, 0.0) for t in tf}

for p in products:
    p["vector"] = vectorize_text(f"{p['name']} {p['description']}")

class SearchResult(BaseModel):
    product_id: int
    name: str
    description: str
    relevance: float


def simple_search(query: str):
    results = []
    q_lower = query.lower()
    for p in products:
        if q_lower in p["name"].lower() or q_lower in p["description"].lower():
            results.append(
                {
                    "product_id": p["product_id"],
                    "name": p["name"],
                    "description": p["description"],
                    "relevance": 1.0,
                }
            )
    return results


def cosine_similarity(v1: dict, v2: dict) -> float:
    all_keys = set(v1) | set(v2)
    num = sum(v1.get(k, 0.0) * v2.get(k, 0.0) for k in all_keys)
    denom1 = math.sqrt(sum(v * v for v in v1.values()))
    denom2 = math.sqrt(sum(v * v for v in v2.values()))
    if denom1 == 0 or denom2 == 0:
        return 0.0
    return num / (denom1 * denom2)


def vector_search(query: str):
    query_vec = vectorize_text(query)
    results = []
    for p in products:
        similarity = cosine_similarity(query_vec, p["vector"])
        if similarity > 0:
            results.append(
                {
                    "product_id": p["product_id"],
                    "name": p["name"],
                    "description": p["description"],
                    "relevance": float(similarity),
                }
            )
    results.sort(key=lambda x: x["relevance"], reverse=True)
    return results


@app.get("/search", response_model=list[SearchResult])
def search(query: str = Query(...)):
    return vector_search(query)


@app.get("/vector_search", response_model=list[SearchResult])
def vector_search_endpoint(query: str = Query(...)):
    return vector_search(query)


class EvaluationRequest(BaseModel):
    query: str
    results: list[int]  # list of product ids in ranked order


class EvaluationMetrics(BaseModel):
    precision: float
    recall: float
    ndcg: float


class EvaluationResponse(BaseModel):
    scores: list[float]
    metrics: EvaluationMetrics


product_map = {p["product_id"]: p for p in products}


def call_llm(query: str, text: str) -> float:
    payload = json.dumps({"query": query, "text": text}).encode("utf-8")
    req = urllib.request.Request(
        f"{LLM_URL}/score", data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req) as resp:
            obj = json.load(resp)
            return float(obj.get("score", 0.0))
    except Exception:
        return 0.0


def evaluate_with_llm(query: str, results: list[int]):
    scores = []
    for pid in results:
        prod = product_map.get(pid)
        if not prod:
            scores.append(0.0)
            continue
        text = f"{prod['name']} {prod['description']}"
        scores.append(call_llm(query, text))
    return scores


def compute_dcg(vals: list[float]) -> float:
    return sum(v / math.log2(i + 2) for i, v in enumerate(vals))


def compute_ndcg(vals: list[float]) -> float:
    ideal = sorted(vals, reverse=True)
    dcg = compute_dcg(vals)
    idcg = compute_dcg(ideal)
    return dcg / idcg if idcg > 0 else 0.0


def compute_metrics(query: str, ids: list[int], scores: list[float]):
    """Compute ranking metrics using provided LLM scores.

    The caller supplies the LLM scores corresponding to ``ids`` so this
    function does not perform additional LLM calls. Items with a score
    greater than ``0.5`` are treated as relevant.
    """

    if len(ids) != len(scores):
        raise ValueError("ids and scores must be the same length")

    retrieved_relevant = [1 if s > 0.5 else 0 for s in scores]
    precision = sum(retrieved_relevant) / len(retrieved_relevant) if ids else 0.0
    # Without relevance information for the entire catalog we approximate recall
    # using only the retrieved documents.
    recall = precision
    ndcg = compute_ndcg(retrieved_relevant)
    return EvaluationMetrics(precision=precision, recall=recall, ndcg=ndcg)


@app.post("/evaluate", response_model=EvaluationResponse)
def evaluate(req: EvaluationRequest):
    scores = evaluate_with_llm(req.query, req.results)
    metrics = compute_metrics(req.query, req.results, scores)
    return EvaluationResponse(scores=scores, metrics=metrics)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
