# Turkish RAG Retrieval Benchmark & Pipeline

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![Sentence Transformers](https://img.shields.io/badge/Embeddings-Sentence_Transformers-FFB000)](https://www.sbert.net)

An end-to-end RAG (Retrieval-Augmented Generation) preprocessing and retrieval pipeline for Turkish text. It covers document chunking, embedding-based similarity search and reranking, and includes a benchmark comparing **25+ embedding/retrieval models** on Turkish retrieval performance.

## Highlights

- **Chunking pipeline** — splits PDF, DOCX, TXT, CSV and Excel files into meaningful chunks (with and without OCR)
- **Production-ready FastAPI service** — a chunking API with Docker, nginx and rate limiting
- **Broad model comparison** — dense embeddings (bge-m3, e5, LaBSE, Qwen3, etc.), Turkish-specific models, and classic methods (BM25, TF-IDF, Jaccard)
- **Reranking & sentence similarity** — ColBERT-based reranking and feature extraction

## Project Structure

```
1-Preprocessing/
├── chunking_with_ocr/        # OCR-based chunking for scanned documents
└── chunking_without_ocr/
    ├── chunking.py           # Core chunking logic
    └── 2-API/                # FastAPI-based chunking service (Docker + nginx)
2-Merge Chunks/
└── merge.py                  # Merging of chunks
3-Similar Chunks/
├── Sentence Similarty Model/ # ColBERT-based similarity
├── Reranking/                # Reranking of results
├── Finetune Feature extraction/
└── Text Based and Feature extraction/
```

## Pipeline Steps

1. **Preprocessing** — Text extraction from raw documents (OCR when needed), cleaning and chunking
2. **Merge** — Organizing/merging the chunks
3. **Similar Chunks** — Finding the nearest chunks via embeddings, improving results with reranking

## Quick Start (Chunking API)

```bash
cd "1-Preprocessing/chunking_without_ocr/2-API"
pip install -r requirements.txt
uvicorn main:app --reload
```

- API: `http://localhost:8000`
- Swagger UI: `http://localhost:8000/docs`

With Docker:

```bash
docker-compose up -d
```

## Model Accuracy Comparison

Top-1 / Top-5 / Top-10 accuracy scores of different models on a Turkish retrieval test set. The best results come from **bge_m3** and **bge_m3_turkish** (Top-1 ≈ 0.76), while classic BM25/TF-IDF methods provide a reasonable baseline.

| Model                     | Top-1 | Top-5 | Top-10 |
|---------------------------|-------|-------|--------|
| multilingual_e5_base      | 0.698 | 0.889 | 0.937  |
| bge_m3                    | 0.762 | 0.937 | 0.984  |
| snowflake                 | 0.714 | 0.873 | 0.873  |
| e5_large                  | 0.508 | 0.635 | 0.683  |
| LaBSE                     | 0.492 | 0.730 | 0.810  |
| instructor                | 0.381 | 0.444 | 0.476  |
| roberta                   | 0.190 | 0.317 | 0.365  |
| jina                      | 0.270 | 0.429 | 0.476  |
| turkish_bert              | 0.143 | 0.270 | 0.333  |
| turkish_e5_large          | 0.730 | 0.889 | 0.921  |
| bge_m3_turkish            | 0.762 | 0.889 | 0.921  |
| base-allnli-stsb          | 0.016 | 0.063 | 0.143  |
| mean-nli-stsb-tr          | 0.365 | 0.524 | 0.619  |
| Qwen3-Embedding           | 0.571 | 0.714 | 0.746  |
| msbayindir/turkish-legal  | 0.238 | 0.302 | 0.365  |
| fkuyumcu/turkish          | 0.000 | 0.016 | 0.032  |
| MiniLM                    | 0.508 | 0.651 | 0.667  |
| eneSadi                   | 0.714 | 0.794 | 0.841  |
| multilingual_e5_large     | 0.651 | 0.778 | 0.873  |
| bm25_word                 | 0.444 | 0.556 | 0.587  |
| bm25_bert                 | 0.540 | 0.762 | 0.778  |
| tfidf_word                | 0.460 | 0.587 | 0.635  |
| tfidf_bert                | 0.413 | 0.683 | 0.746  |
| jacc_word                 | 0.127 | 0.270 | 0.365  |
| jacc_bert                 | 0.317 | 0.556 | 0.619  |

> Top-K: the rate at which the correct chunk appears within the first K results. Values are specific to the test set used.

## Tech Stack

- **Python**, **FastAPI**, **Docker**, **nginx**
- **Sentence Transformers** / dense embedding models
- **BM25**, **TF-IDF**, **ColBERT** (reranking)
- Document processing: PDF / DOCX / CSV / Excel

## Note

This work was built to experimentally investigate which retrieval method performs best in Turkish RAG systems. Scores depend on the dataset used and may vary with different data.
