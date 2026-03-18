import os
import json
import zipfile
import warnings

import streamlit as st
import torch
import numpy as np
import faiss
import gdown

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

# =========================
# CONFIG
# =========================
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
EMB_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

MODEL_DIR = "model_files"
os.makedirs(MODEL_DIR, exist_ok=True)


LORA_ZIP_URL = "https://drive.google.com/uc?id=1QD_-HgPjIn5UAP9fHc3s1FakKNn4v7Fo"
FAISS_URL = "https://drive.google.com/uc?id=196G_UnNofhxpf3ak6VnSU47ockt11he4"
CORPUS_URL = "https://drive.google.com/uc?id=1Fr2rLCjUBHFezKrmc6aXYPrjTvtYYtfY"
RECORDS_URL = "https://drive.google.com/uc?id=1T6BvMxfDt_ID-ZWowcGejnovADxQitey"

ADAPTER_PATH = os.path.join(MODEL_DIR, "lora-adapter")
FAISS_PATH = os.path.join(MODEL_DIR, "faiss.index")
CORPUS_PATH = os.path.join(MODEL_DIR, "rag_corpus.json")
RECORDS_PATH = os.path.join(MODEL_DIR, "geeta_records.json")
LORA_ZIP_PATH = os.path.join(MODEL_DIR, "lora-adapter.zip")

SYSTEM_GU = (
    "તમે ગીતા AI છો — ભગવદ ગીતાના જ્ઞાન પર આધારિત ગુજરાતી સહાયક. "
    "ભગવાન કૃષ્ણના ઉપદેશ અનુસાર, અધ્યાય અને શ્લોક નંબર સાથે "
    "ગુજરાતીમાં ઉત્તર આપો."
)

# =========================
# DEVICE
# =========================
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# =========================
# DOWNLOAD ASSETS
# =========================
def ensure_files():
    if not os.path.exists(FAISS_PATH):
        gdown.download(FAISS_URL, FAISS_PATH, quiet=False)

    if not os.path.exists(CORPUS_PATH):
        gdown.download(CORPUS_URL, CORPUS_PATH, quiet=False)

    if not os.path.exists(RECORDS_PATH):
        gdown.download(RECORDS_URL, RECORDS_PATH, quiet=False)

    # download zip only if extracted adapter folder not present
    if not os.path.exists(ADAPTER_PATH):
        gdown.download(LORA_ZIP_URL, LORA_ZIP_PATH, quiet=False)
        with zipfile.ZipFile(LORA_ZIP_PATH, "r") as zf:
            zf.extractall(ADAPTER_PATH)


# =========================
# LOAD EVERYTHING ONCE
# =========================
@st.cache_resource
def load_all():
    ensure_files()

    embedder = SentenceTransformer(EMB_MODEL)

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    )

    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model = model.to(device)
    model.eval()

    faiss_index = faiss.read_index(FAISS_PATH)

    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        rag_corpus = json.load(f)

    with open(RECORDS_PATH, "r", encoding="utf-8") as f:
        geeta_data = json.load(f)

    return embedder, tokenizer, model, faiss_index, rag_corpus, geeta_data


embedder, tokenizer, model, faiss_index, rag_corpus, geeta_data = load_all()


# =========================
# RETRIEVE
# =========================
def retrieve(query, top_k=3):
    q = embedder.encode([query], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(q)
    D, I = faiss_index.search(q, top_k)

    out = []
    for d, i in zip(D[0], I[0]):
        if i < len(geeta_data):
            r = geeta_data[i].copy()
            r["score"] = float(d)
            out.append(r)
    return out


# =========================
# CHAT
# =========================
def geeta_chat(question, top_k=3, max_new_tokens=180):
    refs = retrieve(question, top_k=top_k)

    ctx = ""
    for r in refs:
        ctx += (
            f"\n[અધ્યાય {r['chapter_number']}, શ્લોક {r['shloka_number']}]\n"
            f"અર્થ: {r['meaning']}\n"
            f"સંદેશ: {r['context_for_ml']}\n"
        )

    # simpler prompt for more stable generation
    prompt = f"""{SYSTEM_GU}

પ્રશ્ન: {question}

સંબંધિત ગીતા શ્લોકો:
{ctx}

ગુજરાતીમાં સ્પષ્ટ, સરળ અને અર્થપૂર્ણ જવાબ આપો.
અધ્યાય અને શ્લોક નંબર જરૂર લખો.
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1200,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[0][inputs["input_ids"].shape[-1]:]
    answer = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    ).strip()

    return answer, refs


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="🕉️ Geeta AI", layout="centered")

st.title("🕉️ Geeta AI")
st.subheader("ભગવદ ગીતાના જ્ઞાન પર આધારિત ગુજરાતી સહાયક")

question = st.text_area(
    "તમારો પ્રશ્ન લખો:",
    height=120,
    placeholder="ઉદાહરણ: હું બહુ નિરાશ છું, શું કરવું?"
)

if st.button("જવાબ મેળવો"):
    if not question.strip():
        st.warning("કૃપા કરીને પ્રશ્ન લખો.")
    else:
        with st.spinner("ગીતા AI વિચાર કરી રહ્યું છે..."):
            answer, refs = geeta_chat(question)

        st.markdown("### 🕉️ ઉત્તર")
        st.write(answer if answer else "માફ કરશો, યોગ્ય ઉત્તર જનરેટ થયો નથી. કૃપા કરીને ફરી પ્રયત્ન કરો.")

        st.markdown("### 📖 સંબંધિત શ્લોકો")
        for r in refs:
            st.markdown(
                f"""
**અધ્યાય {r['chapter_number']} — શ્લોક {r['shloka_number']}**  
**સ્કોર:** {r['score']:.3f}  
**અર્થ:** {r['meaning']}  
**સંદેશ:** {r['context_for_ml']}
"""
            )
