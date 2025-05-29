from fastapi import FastAPI, Query
from pydantic import BaseModel
import pandas as pd
import os
import random

app = FastAPI()

DATA_PATH = os.environ.get("DATA_PATH", "/data/products.csv")
products_df = pd.read_csv(DATA_PATH)

class SearchResult(BaseModel):
    product_id: int
    name: str
    description: str
    relevance: float


def simple_search(query: str):
    results = []
    q_lower = query.lower()
    for _, row in products_df.iterrows():
        if q_lower in row["name"].lower() or q_lower in row["description"].lower():
            results.append(
                {
                    "product_id": int(row["product_id"]),
                    "name": row["name"],
                    "description": row["description"],
                    "relevance": 1.0,
                }
            )
    return results


def vectorize(text: str):
    random.seed(hash(text) % 2 ** 32)
    return [random.random() for _ in range(10)]


def vector_search(query: str):
    query_vec = vectorize(query)
    results = []
    for _, row in products_df.iterrows():
        product_vec = vectorize(f"{row['name']} {row['description']}")
        similarity = sum(q * p for q, p in zip(query_vec, product_vec))
        results.append(
            {
                "product_id": int(row["product_id"]),
                "name": row["name"],
                "description": row["description"],
                "relevance": float(similarity),
            }
        )
    results.sort(key=lambda x: x["relevance"], reverse=True)
    return results


@app.get("/search", response_model=list[SearchResult])
def search(query: str = Query(...)):
    return simple_search(query)


@app.get("/vector_search", response_model=list[SearchResult])
def vector_search_endpoint(query: str = Query(...)):
    return vector_search(query)


class EvaluationRequest(BaseModel):
    query: str
    results: list[int]  # list of product ids in ranked order


class EvaluationResponse(BaseModel):
    scores: list[float]


def evaluate_with_llm(query: str, results: list[int]):
    # Placeholder for LLM integration
    # TODO: connect to actual LLM and generate scores
    return [1.0 for _ in results]


@app.post("/evaluate", response_model=EvaluationResponse)
def evaluate(req: EvaluationRequest):
    scores = evaluate_with_llm(req.query, req.results)
    return EvaluationResponse(scores=scores)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
