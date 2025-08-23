#!/usr/bin/env python
# all_methods_benchmark_full.py
# =============================================================
# WikiRAG‑TR,  TQuAD‑TR  ve  MSMARCO‑TR (örneklenmiş)
#    • Yoğun (dense) modeller  +  BM25  +  TF‑IDF  +  Jaccard
#    • Her yöntem için Top‑1 / Top‑5 / Top‑10 doğruluk
# =============================================================

RUN_WIKIRAG  = False     # 1000  soru – 5000 chunk
RUN_TQUAD    = False     # 1190  soru – ~600 paragraf
RUN_MSMARCO  = True      # 1000* soru – 100k passage  (örnek)

MSM_PASSAGE_SAMPLE = 10_000   # None yapın = tüm passage'lar
MSM_QUERY_SAMPLE   = 100      # None yapın = tüm query'ler

import warnings, numpy as np, matplotlib.pyplot as plt, tqdm, faiss, nltk, gc, torch, sys, random
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, logging as hf_logging
from huggingface_hub import snapshot_download

# ---------- Ortam ---------- #
nltk.download("punkt")
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"► Çalışma aygıtı: {device}")

# ---------- Yoğun modeller ---------- #
dense_models = {
    "multilingual_e5_base"   : "intfloat/multilingual-e5-base",
    "bge_m3"                 : "BAAI/bge-m3",
    "snowflake"              : "Snowflake/snowflake-arctic-embed-l-v2.0",
    "e5_large"               : "intfloat/e5-large-v2",
    "LaBSE"                  : "sentence-transformers/LaBSE",
    "instructor"             : "hkunlp/instructor-large",
    "roberta"                : "sentence-transformers/all-roberta-large-v1",
    "jina"                   : "Thaweewat/jina-embedding-v3-m2v-1024",
    "turkish_bert"           : "dbmdz/bert-base-turkish-cased",
    "turkish_e5_large"       : "ytu-ce-cosmos/turkish-e5-large",
    "bge_m3_turkish"         : "seroe/bge-m3-turkish-triplet-matryoshka",
    "base-allnli-stsb"       : "emrecan/turkish-bert-base-allnli-stsb",
    "mean-nli-stsb-tr"       : "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr",
    "Qwen3-Embedding"        : "Qwen/Qwen3-Embedding-0.6B",
    "msbayindir/turkish-legal": "msbayindir/turkish-legal-bert-base-uncased-stsb-v1-sts",
    "fkuyumcu/turkish"       : "fkuyumcu/turkish-wiki-rag-embeddings-v2",
    "jinaai"                 : "jinaai/jina-embeddings-v3",
    "MiniLM"                 : "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "eneSadi"                : "eneSadi/turkuaz-embeddings",
    "Alibaba"                : "Alibaba-NLP/gte-multilingual-base",
    "multilingual_e5_large"  : "intfloat/multilingual-e5-large-instruct",
}

print("\n► Modeller indiriliyor / doğrulanıyor …")
ok_models = {}
for tag, repo in dense_models.items():
    try:
        ok_models[tag] = snapshot_download(
            repo,
            resume_download=True,
            allow_patterns=["*.safetensors", "*tokenizer.json", "*config.json", "*.json"],
        )
        print(f"  ✓ {tag}")
    except Exception as e:
        print(f"  ✗ {tag} atlandı – ({e.__class__.__name__})")
if not ok_models:
    sys.exit("Hiçbir yoğun model indirilemedi, çıkılıyor.")

# ---------- Veri hazırlayıcı ---------- #
def prepare_dataset(name, target_q=1000):
    if name == "MSMARCO-TR":
        print("► MSMARCO‑TR seti örnekleniyor …")

        # Train split'ini kullan
        print("  • Train split'i yükleniyor...")
        qrels = load_dataset("parsak/msmarco-tr", "qrels", split="train")
        queries = load_dataset("parsak/msmarco-tr", "queries", split="train")

        # İlgili query ID'lerini topla
        print("  • Qrels işleniyor...")
        pos = {}
        for r in qrels:
            if r["rank"] > 0:
                pos.setdefault(r["qid"], []).append(r["pid"])

        print(f"  • {len(pos)} query'nin pozitif passage'ı var")

        # Query'leri filtrele
        print("  • Query'ler filtreleniyor...")
        valid_queries = []
        queries_shuffled = queries.shuffle(seed=42)

        for q in queries_shuffled:
            if q["qid"] in pos:
                valid_queries.append(q)
                # MSM_QUERY_SAMPLE None ise tüm query'leri al
                if MSM_QUERY_SAMPLE is not None and len(valid_queries) == MSM_QUERY_SAMPLE:
                    break

        print(f"  • {len(valid_queries)} geçerli query bulundu")

        # Gerçekten işlenecek query sayısını belirle
        final_query_count = len(valid_queries) if MSM_QUERY_SAMPLE is None else min(MSM_QUERY_SAMPLE,
                                                                                    len(valid_queries))
        used_queries = valid_queries[:final_query_count]

        # İhtiyaç duyulan passage ID'lerini topla
        needed_pids = set()
        for q in used_queries:
            needed_pids.update(pos[q["qid"]])
        print(f"  • {len(needed_pids)} pozitif passage gerekli")

        # Passage'ları yükle - İYİLEŞTİRİLDİ!
        print("  • Passage'lar yükleniyor...")
        corpus = load_dataset("parsak/msmarco-tr", "passages", split="train")

        selected_passages = []
        pid2idx = {}
        positive_count = 0

        # ÖNCE TÜM POZİTİF PASSAGE'LARI BUL VE EKLE
        print("  • Pozitif passage'lar aranıyor...")
        corpus_list = list(corpus)  # Tüm corpus'u listeye çevir
        print(f"  • Toplam {len(corpus_list):,} passage taranacak")

        for i, p in enumerate(corpus_list):
            if p["pid"] in needed_pids:
                pid2idx[p["pid"]] = len(selected_passages)
                selected_passages.append(p["text"])
                positive_count += 1
                if positive_count % 50 == 0:
                    print(
                        f"    ↳ {positive_count}/{len(needed_pids)} pozitif bulundu ({i + 1:,}/{len(corpus_list):,} tarandı)")

        print(f"  • {positive_count}/{len(needed_pids)} pozitif passage bulundu")

        # SONRA RASTGELE PASSAGE'LAR EKLE (eğer MSM_PASSAGE_SAMPLE belirtilmişse)
        if MSM_PASSAGE_SAMPLE is not None:
            remaining_needed = MSM_PASSAGE_SAMPLE - len(selected_passages)
            if remaining_needed > 0:
                print(f"  • {remaining_needed} rastgele passage ekleniyor...")
                # Rastgele sıralama için shuffle kullan
                random.seed(42)
                random_indices = list(range(len(corpus_list)))
                random.shuffle(random_indices)

                added_random = 0
                for idx in random_indices:
                    p = corpus_list[idx]
                    if p["pid"] not in pid2idx:  # Daha önce eklenmemiş
                        pid2idx[p["pid"]] = len(selected_passages)
                        selected_passages.append(p["text"])
                        added_random += 1
                        if added_random >= remaining_needed:
                            break
                        if added_random % 1000 == 0:
                            print(f"    ↳ {added_random}/{remaining_needed} rastgele eklendi")
        else:
            # MSM_PASSAGE_SAMPLE None ise, tüm corpus'u ekle
            print("  • Tüm passage'lar ekleniyor...")
            for i, p in enumerate(corpus_list):
                if p["pid"] not in pid2idx:  # Daha önce eklenmemiş
                    pid2idx[p["pid"]] = len(selected_passages)
                    selected_passages.append(p["text"])
                    if (i + 1) % 50000 == 0:
                        print(f"    ↳ {len(selected_passages):,} passage eklendi")

        print(f"  • Toplam: {positive_count} pozitif + {len(selected_passages) - positive_count:,} diğer passage")

        # Final Q, C, Y oluştur
        Q, Y = [], []
        missing_count = 0

        for q in used_queries:
            if q["qid"] in pos:
                first_positive_pid = pos[q["qid"]][0]
                if first_positive_pid in pid2idx:
                    Q.append(q["text"])
                    Y.append(pid2idx[first_positive_pid])
                else:
                    missing_count += 1

        if missing_count > 0:
            print(f"  ⚠ {missing_count} query'nin pozitif passage'ı bulunamadı")

        print(f"MSMARCO‑TR örnek: {len(Q)} soru, {len(selected_passages):,} passage.")

        # DEBUG BİLGİLERİ
        print(f"► DEBUG: İlk 3 Y değeri: {Y[:3] if len(Y) >= 3 else Y}")
        print(f"► DEBUG: Y min/max: {min(Y) if Y else 'YOK'} / {max(Y) if Y else 'YOK'}")
        print(f"► DEBUG: İlk soru: '{Q[0][:100]}...' " if Q else "► DEBUG: Soru yok!")

        return Q, selected_passages, Y

# ---------- Yardımcılar ---------- #
def eval_dense(model, C, Q, Y, K=10):
    c_vec = model.encode(C, convert_to_tensor=False, show_progress_bar=False)
    q_vec = model.encode(Q, convert_to_tensor=False, show_progress_bar=False)
    idx = faiss.IndexFlatL2(c_vec.shape[1]); idx.add(np.asarray(c_vec, np.float32))
    _, I = idx.search(np.asarray(q_vec, np.float32), K)
    return dict(top1=np.mean([Y[i] in I[i][:1]  for i in range(len(Q))]),
                top5=np.mean([Y[i] in I[i][:5]  for i in range(len(Q))]),
                top10=np.mean([Y[i] in I[i][:10] for i in range(len(Q))]))
jac = lambda a,b: len(a&b)/len(a|b) if a|b else 0

# ---------- Koşul listesi ---------- #
datasets_to_run = []
if RUN_WIKIRAG: datasets_to_run.append("WikiRAG-TR")
if RUN_TQUAD:   datasets_to_run.append("TQuAD")
if RUN_MSMARCO: datasets_to_run.append("MSMARCO-TR")

# ---------- Ana döngü ---------- #
for ds_name in datasets_to_run:
    print(f"\n================ {ds_name} ================")
    Q, C, Y = prepare_dataset(ds_name)
    K, metrics = 10, {}

    # --- Yoğun modeller
    for tag, path in ok_models.items():
        print(f"► {tag}: embedding …")
        model=None
        try:
            model = SentenceTransformer(path, device=device)
            metrics[tag] = eval_dense(model, C, Q, Y, K)
        except Exception as e:
            print(f"    ↳ {tag} atlandı ({e.__class__.__name__})")
        finally:
            if model is not None:
                try: model.to("cpu")
                except: pass
            del model; torch.cuda.empty_cache(); gc.collect()

    # --- BM25
    print("► BM25 …")
    tok_word=[word_tokenize(c.lower()) for c in C]
    bm25_w = BM25Okapi(tok_word)
    tok    = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
    wp     = lambda t: tok.tokenize(t.lower())
    bm25_b = BM25Okapi([wp(c) for c in C])
    for tag,bm25,prep in [("bm25_word",bm25_w,word_tokenize),
                          ("bm25_bert",bm25_b,wp)]:
        I=[np.argsort(bm25.get_scores(prep(q.lower())))[::-1][:K] for q in Q]
        metrics[tag]=dict(top1=np.mean([Y[i] in I[i][:1] for i in range(len(Q))]),
                          top5=np.mean([Y[i] in I[i][:5] for i in range(len(Q))]),
                          top10=np.mean([Y[i] in I[i][:10]for i in range(len(Q))]))

    # --- TF‑IDF
    print("► TF‑IDF …")
    vec_w=TfidfVectorizer().fit(C); X_w=vec_w.transform(C)
    C_wp=[" ".join(wp(c)) for c in C]
    vec_b=TfidfVectorizer().fit(C_wp); X_b=vec_b.transform(C_wp)
    for tag,V,qv in [("tfidf_word",X_w,lambda s:vec_w.transform([s])),
                     ("tfidf_bert",X_b,lambda s:vec_b.transform([" ".join(wp(s))]))]:
        I=[np.argsort((V@qv(q).T).toarray().ravel())[::-1][:K] for q in Q]
        metrics[tag]=dict(top1=np.mean([Y[i] in I[i][:1] for i in range(len(Q))]),
                          top5=np.mean([Y[i] in I[i][:5] for i in range(len(Q))]),
                          top10=np.mean([Y[i] in I[i][:10]for i in range(len(Q))]))

    # --- Jaccard
    print("► Jaccard …")
    set_w=[set(t) for t in tok_word]
    set_b=[set(wp(c)) for c in C]
    for tag,sets,prep in [("jacc_word",set_w,lambda s:set(word_tokenize(s.lower()))),
                          ("jacc_bert",set_b,lambda s:set(wp(s)))]:
        I=[]
        for q in tqdm.tqdm(Q,desc=tag,ncols=80):
            sims=[jac(prep(q),cs) for cs in sets]
            I.append(np.argsort(sims)[::-1][:K])
        metrics[tag]=dict(top1=np.mean([Y[i] in I[i][:1] for i in range(len(Q))]),
                          top5=np.mean([Y[i] in I[i][:5] for i in range(len(Q))]),
                          top10=np.mean([Y[i] in I[i][:10]for i in range(len(Q))]))

    # --- Sonuç tablosu
    print("\n=== DOĞRULUKLAR ===")
    for m,s in metrics.items():
        print(f"{m:22s}  Top‑1:{s['top1']:.3f} | Top‑5:{s['top5']:.3f} | Top‑10:{s['top10']:.3f}")

    # --- Grafik
    lbl=list(metrics.keys()); x=np.arange(len(lbl)); w=.25
    plt.figure(figsize=(1.2*len(lbl),6))
    plt.bar(x-w,[metrics[m]["top1"] for m in lbl],w,label="Top‑1")
    plt.bar(x,  [metrics[m]["top5"] for m in lbl],w,label="Top‑5")
    plt.bar(x+w,[metrics[m]["top10"]for m in lbl],w,label="Top‑10")
    plt.xticks(x,lbl,rotation=45,ha="right",fontsize=8)
    plt.ylabel("Accuracy"); plt.ylim(0,1)
    plt.title(f"{ds_name} – Yöntem Karşılaştırması (Top‑10’a kadar)")
    plt.legend(); plt.tight_layout(); plt.show()

print("\n✓ Tüm işlemler tamamlandı.")
