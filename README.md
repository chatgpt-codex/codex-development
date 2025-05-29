# LLM Judge Demo

This repository contains a very small demo showing how to search a toy
e-commerce dataset and evaluate the ranking with an LLM. Everything is
packaged with Docker Compose.

## Services

- **backend**: FastAPI application that provides search endpoints and a
  placeholder LLM evaluation endpoint.
- **frontend**: Static HTML/JS that calls the backend.

## Usage

Build and start the services:

```bash
docker-compose up --build
```

Then open <http://localhost:8080> in your browser. Enter a search query
and the results will be shown. The evaluation endpoint is available at
`POST http://localhost:8000/evaluate` and currently returns placeholder
scores.

## Development Notes

- `data/products.csv` holds a few example products.
- The backend uses naive text search and a trivial vector similarity
  placeholder. Replace `evaluate_with_llm` in `backend/app.py` with a
  real LLM call to score ranking quality.
