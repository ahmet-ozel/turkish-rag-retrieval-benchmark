# fine_tune_wikirag_cpufaiss_ep10.py
# =============================================================
# WikiRAG‑TR : batch 32, 10 epoch,  yalnız CPU‑FAISS değerlendirme
# =============================================================
import os, sys, time, random, gc, warnings
import numpy as np, torch, faiss
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from transformers import logging as hf_logging

# ---------------- Ortam ----------------
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
BATCH    = 32          # 32 = 12 GB RTX 3080 için güvenli; CPU'da da makul
EPOCHS   = 10
LR       = 5e-5
SAVE_DIR = "ft_models"
LOG_DIR  = "logs"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR,  exist_ok=True)

# -------------- Unicode‑safe Tee --------------
class Tee:
    def __init__(self, fname):
        self.file = open(fname, "a", encoding="utf-8", buffering=1)
        self.stdout = sys.stdout
    def write(self, txt):
        self.file.write(txt)
        try: self.stdout.write(txt)
        except UnicodeEncodeError:
            self.stdout.write(txt.encode("ascii", "ignore").decode())
    def flush(self):
        self.file.flush(); self.stdout.flush()

sys.stdout = Tee(os.path.join(LOG_DIR, "train_log.txt"))
print(f"\n=== Run started {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
print(f"Device: {DEVICE}, Batch: {BATCH}, Epochs: {EPOCHS}\n")

# -------------- Modeller --------------
dense_models = {
    "multilingual_e5_base" : "intfloat/multilingual-e5-base",
    "multilingual_e5_large": "intfloat/multilingual-e5-large-instruct",
    "bge_m3"               : "BAAI/bge-m3",
    "turkish_e5_large"     : "ytu-ce-cosmos/turkish-e5-large",
    "snowflake"             : "Snowflake/snowflake-arctic-embed-l-v2.0",
}

# -------------- Veri --------------
print("Loading WikiRAG‑TR ...")
ds = load_dataset("Metin/WikiRAG-TR", split="train")
train_triplets, eval_Q, eval_C, eval_Y = [], [], [], []

for ex in ds:
    chunks = ex["context"].split("\n")
    if len(chunks) not in (5, 6): continue
    idx = ex["correct_intro_idx"]
    pos = chunks[idx]
    neg = random.choice([c for i,c in enumerate(chunks) if i!=idx])
    train_triplets.append(InputExample(texts=[ex["question"], pos, neg]))

    if len(eval_Q) < 1000:
        keep = chunks[:5] if len(chunks)==6 and idx>=5 else chunks
        if len(keep) < 5: keep.append(pos)
        start = len(eval_C)
        eval_C.extend(keep)
        eval_Q.append(ex["question"])
        eval_Y.append(start + keep.index(pos))

    if len(train_triplets) >= 10_000 and len(eval_Q) == 1000:
        break

print(f"Triplets : {len(train_triplets):,}")
print(f"Eval Q/C : {len(eval_Q)} / {len(eval_C)}\n")

# -------------- CPU‑FAISS değerlendirme --------------
def evaluate_cpu(model, k=10):
    c_vec = model.encode(eval_C, batch_size=128, convert_to_numpy=True, show_progress_bar=False)
    q_vec = model.encode(eval_Q, batch_size=128, convert_to_numpy=True, show_progress_bar=False)
    faiss.normalize_L2(c_vec);  faiss.normalize_L2(q_vec)

    index = faiss.IndexFlatIP(c_vec.shape[1])   # cosine = IP(normalized)
    index.add(c_vec)
    _, I = index.search(q_vec, k)

    top = lambda kk: np.mean([eval_Y[i] in I[i][:kk] for i in range(len(eval_Q))])
    return round(float(top(1)),3), round(float(top(5)),3), round(float(top(10)),3)

# -------------- Eğitim döngüsü --------------
orig_scores, final_scores = {}, {}

for tag, repo in dense_models.items():
    print(f"\n===== {tag} =====")
    model  = SentenceTransformer(repo, device=DEVICE)
    loader = DataLoader(train_triplets, shuffle=True, batch_size=BATCH, drop_last=True)
    lossfn = losses.MultipleNegativesRankingLoss(model)

    t1,t5,t10 = evaluate_cpu(model)
    orig_scores[tag] = {"top1":t1,"top5":t5,"top10":t10}
    print(f"Epoch 0  Top‑1:{t1}  Top‑5:{t5}  Top‑10:{t10}")

    warmup = int(0.1*len(loader))
    for ep in range(1, EPOCHS+1):
        model.fit(
            train_objectives=[(loader, lossfn)],
            epochs=1,
            warmup_steps=warmup,
            optimizer_params={"lr": LR},
            use_amp=(DEVICE=="cuda"),
            show_progress_bar=True
        )
        t1,t5,t10 = evaluate_cpu(model)
        print(f"Epoch {ep:<2} Top‑1:{t1}  Top‑5:{t5}  Top‑10:{t10}")

    final_scores[tag] = {"top1":t1,"top5":t5,"top10":t10}
    path = os.path.join(SAVE_DIR, tag)
    model.save(path);  print(f"Saved  {path}")

    model.to('cpu'); del model
    torch.cuda.empty_cache(); gc.collect()

# -------------- Ortalama raporu --------------
def avg(dic, key): return round(np.mean([v[key] for v in dic.values()]),3)

print("\n=== AVERAGE RESULTS ===")
print(f"Original   Top‑1:{avg(orig_scores,'top1')}  Top‑5:{avg(orig_scores,'top5')}  Top‑10:{avg(orig_scores,'top10')}")
print(f"Fine‑tuned Top‑1:{avg(final_scores,'top1')}  Top‑5:{avg(final_scores,'top5')}  Top‑10:{avg(final_scores,'top10')}")

print(f"\n=== Run finished {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
