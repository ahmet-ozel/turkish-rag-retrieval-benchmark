# Turkish RAG Retrieval Benchmark & Pipeline

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![Sentence Transformers](https://img.shields.io/badge/Embeddings-Sentence_Transformers-FFB000)](https://www.sbert.net)

Türkçe metinler üzerinde uçtan uca bir RAG (Retrieval-Augmented Generation) ön işleme ve retrieval hattı. Doküman parçalama (chunking), embedding tabanlı benzerlik araması ve yeniden sıralama (reranking) adımlarını içerir; ayrıca **25+ embedding/retrieval modelini** Türkçe retrieval başarımına göre karşılaştıran bir benchmark sunar.

## 🎯 Öne Çıkanlar

- **Chunking pipeline** — PDF, DOCX, TXT, CSV, Excel dosyalarını anlamlı parçalara böler (OCR'lı ve OCR'sız iki yol)
- **Üretime hazır FastAPI servisi** — Docker, nginx ve rate limiting ile chunking API'si
- **Geniş model karşılaştırması** — dense embedding'ler (bge-m3, e5, LaBSE, Qwen3 vb.), Türkçe modeller ve klasik yöntemler (BM25, TF-IDF, Jaccard)
- **Reranking & sentence similarity** — ColBERT tabanlı yeniden sıralama ve özellik çıkarımı

## 📁 Proje Yapısı

```
1-Preprocessing/
├── chunking_with_ocr/        # Taranmış belgeler için OCR'lı chunking
└── chunking_without_ocr/
    ├── chunking.py           # Temel chunking mantığı
    └── 2-API/                # FastAPI tabanlı chunking servisi (Docker + nginx)
2-Merge Chunks/
└── merge.py                  # Parçaların birleştirilmesi
3-Similar Chunks/
├── Sentence Similarty Model/ # ColBERT tabanlı benzerlik
├── Reranking/                # Sonuçların yeniden sıralanması
├── Finetune Feature extraction/
└── Text Based and Feature extraction/
```

## ⚙️ Hattın Adımları

1. **Ön İşleme (Preprocessing)** — Ham dokümanlardan metin çıkarımı (gerekirse OCR), temizleme ve chunking
2. **Birleştirme (Merge)** — Chunk'ların düzenlenmesi/birleştirilmesi
3. **Benzerlik (Similar Chunks)** — Embedding ile en yakın chunk'ların bulunması, reranking ile iyileştirme

## 🚀 Hızlı Başlangıç (Chunking API)

```bash
cd "1-Preprocessing/chunking_without_ocr/2-API"
pip install -r requirements.txt
uvicorn main:app --reload
```

- API: `http://localhost:8000`
- Swagger UI: `http://localhost:8000/docs`

Docker ile:

```bash
docker-compose up -d
```

## 📊 Model Doğruluk Karşılaştırmaları

Türkçe bir retrieval test kümesinde farklı modellerin Top-1 / Top-5 / Top-10 doğruluk skorları. En iyi sonuçları **bge_m3** ve **bge_m3_turkish** (Top-1 ≈ 0.76) verirken, klasik BM25/TF-IDF yöntemleri makul bir temel çizgi (baseline) oluşturuyor.

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

> Top-K: doğru chunk'ın ilk K sonuç içinde bulunma oranı. Değerler kullanılan test kümesine özeldir.

## 🧰 Teknoloji Yığını

- **Python**, **FastAPI**, **Docker**, **nginx**
- **Sentence Transformers** / dense embedding modelleri
- **BM25**, **TF-IDF**, **ColBERT** (reranking)
- Doküman işleme: PDF / DOCX / CSV / Excel

## 📝 Not

Bu çalışma, Türkçe RAG sistemlerinde hangi retrieval yönteminin daha iyi sonuç verdiğini deneysel olarak araştırmak için hazırlanmıştır. Skorlar kullanılan veri kümesine bağlıdır ve farklı verilerde değişebilir.
