# LLM Judge Demo

This repository contains a demo showing how to search a toy e-commerce
dataset and evaluate the ranking quality with a lightweight local LLM
service. Everything is packaged with Docker Compose.

## Services

- **backend**: FastAPI application that provides search endpoints and
  computes ranking metrics by calling the LLM service.
- **frontend**: Static HTML/JS that calls the backend.
- **llm**: FastAPI service exposing a `/score` endpoint. It loads a local
  Mistral 7B model when available (set `MISTRAL_PATH` to the folder
  containing the weights) and falls back to a simple Jaccard similarity
  if the model cannot be loaded.

## Usage

Build and start the services:

```bash
docker-compose up --build
```

Then open <http://localhost:8080> in your browser. Enter a search query
and the results will be shown. The evaluation endpoint is available at
`POST http://localhost:8000/evaluate` and returns per-item LLM scores as
well as overall precision, recall and NDCG metrics.

## Development Notes

- `data/products.csv` holds a few example products.
- The backend implements a simple TF‑IDF search. The `llm` service can
  run a real Mistral model for evaluation if you mount the pre-downloaded
  weights into the container.

## Testing

Install the development dependencies and run the test suite with pytest:

```bash
pip install -r requirements-dev.txt
pytest
```
