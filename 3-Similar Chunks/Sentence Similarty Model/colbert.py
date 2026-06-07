# streamlit_app.py
# -----------------------------------------------------------
# Gereksinimler:
# pip install streamlit torch colbert-ai requests
# (vLLM sunucusu:  google/gemma-3-12b-it  --port 8008  --serve)
# Çalıştırmak için:
# streamlit run streamlit_app.py
# -----------------------------------------------------------

import json
import requests
import torch
import streamlit as st
from colbert.modeling.checkpoint import Checkpoint
from colbert.infra import ColBERTConfig

# ---------- Yardımcı Fonksiyonlar -------------------------------------------

@st.cache_resource(show_spinner=False)
def load_colbert_model():
    cfg = ColBERTConfig(root="experiments")
    ckpt = Checkpoint("ytu-ce-cosmos/turkish-colbert", colbert_config=cfg)
    ckpt.eval()
    return ckpt

def load_chunks_from_json(file_obj):
    """JSON -> [chunk_text] listesi"""
    data = json.load(file_obj)
    return [item["chunk_text"] for item in data]

def maxsim(query_vecs, doc_vecs):
    """ColBERT MaxSim benzerlik skoru (tek query için)."""
    scores = []
    q = query_vecs[0]                       # (q_len, d_model)
    for d in doc_vecs:                      # (d_len, d_model)
        sim = torch.matmul(q, d.T)          # (q_len, d_len)
        score = sim.max(1).values.sum().item()
        scores.append(score)
    return scores

def retrieve_top_k(query, docs, ckpt, k=15):
    """En benzer k chunk'ı (metin + skor) döndürür."""
    # Türkçe'de büyük I/ı sorununu normalize edelim
    query_norm = query.replace("I", "ı").lower()

    with torch.no_grad():
        q_vec = ckpt.queryFromText([query_norm]).float()   # (1, q_len, dim)
        d_vecs = ckpt.docFromText(docs).float()            # (n, d_len, dim)

    scores = maxsim(q_vec, d_vecs)
    top_idx = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:k]
    return [(docs[i], scores[i]) for i in top_idx]

def call_vllm_model(prompt, api_url="http://localhost:8008/v1/chat/completions",
                    model_name="google/gemma-3-12b-it", temperature=0.2):
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "temperature": temperature,
        "messages": [
            {"role": "system",
             "content": ("Aşağıda bağlam (context) olarak verilen metin parçalarına dayalı olarak, "
                         "soruyu açık, kısa ve doğru biçimde Türkçe yanıtla.")},
            {"role": "user", "content": prompt}
        ]
    }
    resp = requests.post(api_url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

def build_prompt(query, contexts):
    ctx_text = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    prompt = (
        f"Context:\n{ctx_text}\n\n"
        f"Soru: {query}\n\n"
        "Yalnızca context içindeki bilgilere dayanarak cevap ver."
    )
    return prompt

# ---------- Streamlit UI -----------------------------------------------------

st.title("  ColBERT + vLLM Soru‑Cevap Demo (TR)")

st.sidebar.header(" Adımlar")
st.sidebar.markdown(
    "1. JSON dosyanızı yükleyin\n"
    "2. Sorgu yazın\n"
    "3. **Cevabı Al** butonuna basın"
)

uploaded_json = st.file_uploader("JSON dosyanızı seçin", type=["json"])
query_input = st.text_input("Sorgu", placeholder="Bilim insanları hakkında sorunuz...")

top_k = st.slider("Kaç bağlam (chunk) kullanılsın?", 5, 20, 15)
if st.button(" Cevabı Al"):

    if not uploaded_json:
        st.error("Lütfen önce bir JSON dosyası yükleyin.")
        st.stop()
    if not query_input.strip():
        st.error("Lütfen bir sorgu girin.")
        st.stop()

    with st.spinner(" Dokümanlar taranıyor..."):
        ckpt = load_colbert_model()
        docs = load_chunks_from_json(uploaded_json)
        top_chunks_scores = retrieve_top_k(query_input, docs, ckpt, k=top_k)
        top_chunks = [text for text, _ in top_chunks_scores]

    st.success(f"En benzer {top_k} parça bulundu ")

    with st.expander(" Görüntülenen bağlam parçaları"):
        for i, (chunk, score) in enumerate(top_chunks_scores, 1):
            st.markdown(f"**{i}. ({score:.2f})**  {chunk}")

    with st.spinner(" vLLM modelinden yanıt alınıyor..."):
        prompt = build_prompt(query_input, top_chunks)
        try:
            answer = call_vllm_model(prompt)
        except Exception as e:
            st.error(f"Model isteği başarısız: {e}")
            st.stop()

    st.subheader(" Yanıt")
    st.write(answer)
