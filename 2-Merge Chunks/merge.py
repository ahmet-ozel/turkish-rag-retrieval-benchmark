# -*- coding: utf-8 -*-
"""
JSON/JSONL klasörünü tarar, kayıtlardaki `text` alanından embedding üretir
ve .npy olarak kaydeder. (İsteğe bağlı metadata .jsonl)

 Yapılandırma değişkenleri bu dosyanın en üstündedir.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer


# =========================
# YAPILANDIRMA (burayı düzenleyin)
# =========================
INPUT_DIR      = r"C:\Users\ahmet\Desktop\RAG - cursor\2-Merge Chunks\deneme"             # JSON/JSONL dosyalarının bulunduğu klasör
OUTPUT_NPY     = r"./embeddings/e5_embeddings.npy"  # Embedding matrisinin kaydedileceği .npy
OUTPUT_META    = r"./embeddings/e5_metadata.jsonl"  # Metadata .jsonl (satır-satır hizalı). Boş bırakılırsa yazılmaz.

MODEL_NAME     = "intfloat/e5-large"           # Değiştirilebilir model adı (HF id)
BATCH_SIZE     = 64                             # Encode batch size
TEXT_KEY       = "text"                         # JSON içindeki metin alanı

NORMALIZE      = True                           # L2 normalize embeddings
USE_MEMMAP     = True                           # .npy yazarken memmap (büyük veri için önerilir)
SHOW_PROGRESS  = True                           # TQDM ilerleme çubuğu

# Prefix politikası:
USE_AUTO_PREFIX = True                          # True: modele göre otomatik prefix; False: otomatik kapalı
CUSTOM_PREFIX   = ""                             # Boş değilse bu değer kullanılır ve otomatik tercihleri ezer
# E5 ailesinde belge embedding'i için önerilen prefix: "passage: "
# Sorgu embedding'i için: "query: "


# =========================
# Yardımcılar
# =========================
def detect_default_prefix(model_name: str) -> str:
    """Model adına göre makul varsayılan prefix seçer (E5 -> 'passage: ')."""
    name = model_name.lower()
    if "e5" in name:
        return "passage: "
    return ""

def resolve_prefix(model_name: str) -> str:
    """CUSTOM_PREFIX önceliklidir; o boşsa isteğe bağlı otomatik prefix."""
    if CUSTOM_PREFIX:
        return CUSTOM_PREFIX
    if USE_AUTO_PREFIX:
        return detect_default_prefix(model_name)
    return ""

def load_json_file(fp: Path) -> List[Dict[str, Any]]:
    """
    Tek bir JSON/JSONL dosyasını yükler ve liste döndürür.
    - .jsonl: her satır bir JSON obje
    - .json : dizi veya obje (objede 'chunks' varsa onu alır)
    """
    if fp.suffix.lower() == ".jsonl":
        out = []
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    out.append(obj)
                except json.JSONDecodeError:
                    pass
        return out

    with fp.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "chunks" in data and isinstance(data["chunks"], list):
            return data["chunks"]
        return [data]
    raise ValueError(f"Beklenmedik JSON formatı: {fp}")

def gather_texts(input_dir: Path, text_key: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Klasördeki tüm .json/.jsonl dosyalarını (alt klasörler dahil) tarar.
    `text_key` alanını alır; metin listesi ve metadata listesi döndürür.
    """
    files = sorted(list(input_dir.rglob("*.json")) + list(input_dir.rglob("*.jsonl")))
    if not files:
        raise FileNotFoundError(f"{input_dir} altında .json/.jsonl bulunamadı.")

    texts: List[str] = []
    metas: List[Dict[str, Any]] = []

    for fp in files:
        try:
            records = load_json_file(fp)
        except Exception as e:
            print(f"[!] {fp} okunamadı: {e}")
            continue

        for i, rec in enumerate(records):
            if not isinstance(rec, dict):
                continue
            txt = rec.get(text_key, "")
            if not isinstance(txt, str):
                continue
            clean = txt.strip()
            if not clean:
                continue

            texts.append(clean)
            meta = {
                "source_file": str(fp),
                "source_index": i,
                "length": len(clean),
            }
            # İlgili olabilecek alanları koruyalım
            for k in ("chunk_id", "filename"):
                if k in rec:
                    meta[k] = rec[k]
            metas.append(meta)

    if not texts:
        raise ValueError(f"Hiç metin bulunamadı. text_key='{text_key}' doğru mu?")
    return texts, metas

def encode_texts(
    texts: List[str],
    model_name: str,
    batch_size: int = 64,
    normalize: bool = False,
    prefix: str = "",
    show_progress: bool = True,
) -> np.ndarray:
    """
    SentenceTransformer ile embedding üretir.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)

    if prefix:
        inputs = [prefix + t for t in texts]
    else:
        inputs = texts

    embeddings = model.encode(
        inputs,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=False,  # gerekirse aşağıda L2 normalize ederiz
    )

    if normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms

    return embeddings

def save_embeddings_npy(embeddings: np.ndarray, output_path: Path, use_memmap: bool = False):
    """
    Embedding matrisini .npy olarak yazar.
    use_memmap=True ise başlıklı .npy dosyasını memory-mapped olarak oluşturur.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if use_memmap:
        mm = open_memmap(
            filename=str(output_path),
            mode="w+",
            dtype=embeddings.dtype,
            shape=embeddings.shape,
        )
        mm[:] = embeddings[:]
        del mm  # flush
    else:
        np.save(str(output_path), embeddings)

def save_metadata_jsonl(metas: List[Dict[str, Any]], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")


# =========================
# Ana akış
# =========================
def main():
    input_dir  = Path(INPUT_DIR)
    output_npy = Path(OUTPUT_NPY)
    output_meta = Path(OUTPUT_META) if OUTPUT_META else None

    prefix = resolve_prefix(MODEL_NAME)
    print(f"[i] Model:   {MODEL_NAME}")
    print(f"[i] Prefix:  {repr(prefix)}")
    print(f"[i] Input:   {input_dir}")
    print(f"[i] Output:  {output_npy}")
    if output_meta:
        print(f"[i] Meta:    {output_meta}")

    print("[i] JSON taranıyor...")
    texts, metas = gather_texts(input_dir, text_key=TEXT_KEY)
    print(f"[i] Toplam metin: {len(texts)}")

    print("[i] Embedding hesaplanıyor...")
    emb = encode_texts(
        texts=texts,
        model_name=MODEL_NAME,
        batch_size=BATCH_SIZE,
        normalize=NORMALIZE,
        prefix=prefix,
        show_progress=SHOW_PROGRESS,
    )
    print(f"[i] Embedding şekli: {emb.shape}")

    print(f"[i] Kaydediliyor -> {output_npy}")
    save_embeddings_npy(emb, output_npy, use_memmap=USE_MEMMAP)

    if output_meta:
        print(f"[i] Metadata yazılıyor -> {output_meta}")
        save_metadata_jsonl(metas, output_meta)

    print("[] Tamamlandı.")


if __name__ == "__main__":
    main()
