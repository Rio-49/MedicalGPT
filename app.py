# -*- coding: utf-8 -*-
"""
MedicalGPT — Books-only RAG (Exact Quotes + Remedies + Grounded Answer + Reranker)

What’s included
- Exact quotes only with 1–2 sentences per bullet
- Line-range citations (p.X Lstart-Lend, Heading) from our own chunker
- OCR cleanup before chunking (hyphen joins, common word-joins like infront→in front; ligatures)
- Heading-aware chunking
- Suggested remedies (books-only) with diversity-aware ranking and supporting quotes
- Grounded LLM answer that uses ONLY the retrieved quotes with inline citations
- Optional cross-encoder reranker (uses if available; otherwise falls back)
- NEW: Patient info (name, age, unique patient_id, brief) + ongoing chat transcript
- NEW: Auto-save transcript JSON locally and to Google Drive as <patient_id>.json
"""

import os
import io
import re
import csv
import json
import mimetypes
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Set

import streamlit as st
import numpy as np

# PDF report
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib import enums

# --- Google Drive dependencies (optional; enable via requirements) ---
try:
    from google.oauth2.service_account import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    _HAVE_GDRIVE = True
except Exception:
    _HAVE_GDRIVE = False

# Optional .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# PDF extraction
from PyPDF2 import PdfReader
try:
    import fitz  # PyMuPDF
    _HAVE_FITZ = True
except Exception:
    _HAVE_FITZ = False

# Embeddings + Vector store
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# LLMs (for grounded answer)
from langchain_openai import ChatOpenAI           # OpenRouter-compatible
from langchain_ollama import ChatOllama           # local fallback
from langchain_core.messages import HumanMessage, SystemMessage

# Optional reranker (auto-uses if import works)
_CE = None
try:
    from sentence_transformers import CrossEncoder
    _CE = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
except Exception:
    _CE = None

# -----------------------
# PAGE CONFIG & STYLES
# -----------------------
st.set_page_config(page_title="MedicalGPT — Books-only RAG", page_icon="🩺", layout="wide")
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.0rem; }
      textarea { min-height: 140px; }
      .stTextArea, .stButton { width: 100%; }
      .stButton>button { width: 100%; height: 48px; font-weight: 600; }
      code, pre { white-space: pre-wrap !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------
# PATHS / DEFAULTS
# -----------------------
APP_DIR = os.getcwd()
DATA_DIR = os.path.join(APP_DIR, "data")
INDEX_DIR = os.path.join(DATA_DIR, "indexes")
CFG_DIR = os.path.join(DATA_DIR, "cfg")
CASE_DIR = os.path.join(DATA_DIR, "cases")
REPORT_DIR = os.path.join(DATA_DIR, "reports")
for _d in (DATA_DIR, INDEX_DIR, CFG_DIR, CASE_DIR, REPORT_DIR):
    os.makedirs(_d, exist_ok=True)
CSV_PATH = os.path.join(CASE_DIR, "cases.csv")

DEFAULT_LIBRARY = os.getenv("LIBRARY_NAME", "main_library")
DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
DEFAULT_LOCAL_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct")

# OpenRouter (if available) for grounded answer
OR_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()  # sk-or-…
OR_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1").strip()
OR_MODEL = os.getenv("OPENAI_MODEL", "google/gemini-2.5-flash-preview-09-2025").strip()
OR_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "https://u-datatool.u-stocks.in").strip()
OR_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "MedicalGPT RAG").strip()

# -----------------------
# OCR CLEANUP (soft)
# -----------------------
_CLEAN_JOINS = {
    r"\binfront\b": "in front",
    r"\bto-day\b": "today",
    r"\bto-morrow\b": "tomorrow",
    r"\bco-operat": "cooperat",
}
LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-"}

def normalize_ligatures(text: str) -> str:
    for k, v in LIGATURES.items():
        text = text.replace(k, v)
    return text

def soft_ocr_cleanup(text: str) -> str:
    # join hyphenated linebreaks: exam-\nple → example
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    # collapse single newlines within paragraphs
    text = re.sub(r"([^\n])\n(?!\n)", r"\1 ", text)
    # shrink 3+ newlines → 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # common OCR word-joins
    for pat, repl in _CLEAN_JOINS.items():
        text = re.sub(pat, repl, text, flags=re.IGNORECASE)
    text = normalize_ligatures(text)
    # trim whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -----------------------
# PAGE/LINE STRUCTURES
# -----------------------
@dataclass
class Page:
    number: int
    lines: List[str]

@dataclass
class ChunkInfo:
    page: int
    heading: str
    start_line: int
    end_line: int
    text: str

HEADING_HINTS = [
    "CARE OF HAIR",
    "DYEING", "COLORING", "COLOURING",
    "PROTECTION OF HAIR",
    "BRUSHING", "COMBING",
    "HERBS", "FOODS", "HOME REMEDIES",
]

def is_heading(line: str) -> bool:
    caps = sum(1 for c in line if c.isupper())
    letters = sum(1 for c in line if c.isalpha())
    caps_ratio = (caps / letters) if letters else 0
    has_hint = any(h in line.upper() for h in HEADING_HINTS)
    return (caps_ratio > 0.8 and len(line) <= 90) or has_hint

# -----------------------
# PDF → PAGES (sentences as lines)
# -----------------------
def extract_pages_from_pdf(file_like: io.BytesIO) -> List[Page]:
    text_pages: List[str] = []
    try:
        file_like.seek(0)
        reader = PdfReader(file_like)
        for i in range(len(reader.pages)):
            try:
                t = reader.pages[i].extract_text() or ""
            except Exception:
                t = ""
            if not t and _HAVE_FITZ:
                try:
                    file_like.seek(0)
                    with fitz.open(stream=file_like.read(), filetype="pdf") as dz:
                        if i < dz.page_count:
                            t = dz[i].get_text("text") or ""
                except Exception:
                    t = ""
            text_pages.append(t)
    except Exception:
        if _HAVE_FITZ:
            try:
                file_like.seek(0)
                with fitz.open(stream=file_like.read(), filetype="pdf") as dz:
                    for i in range(dz.page_count):
                        try:
                            t = dz[i].get_text("text") or ""
                        except Exception:
                            t = ""
                        text_pages.append(t)
            except Exception:
                pass

    pages: List[Page] = []
    for idx, raw in enumerate(text_pages, start=1):
        cleaned = soft_ocr_cleanup(raw or "")
        if not cleaned:
            continue
        # sentence-as-line, preserves ability to cite Lx-Ly
        sent_like = re.split(r"(?<=[.!?])\s+", cleaned)
        sent_like = [s.strip() for s in sent_like if s.strip()]
        if sent_like:
            pages.append(Page(number=idx, lines=sent_like))
    return pages

# -----------------------
# HEADING-AWARE CHUNKING
# -----------------------
def chunk_by_headings(pages: List[Page], max_chars: int = 1200) -> List[ChunkInfo]:
    chunks: List[ChunkInfo] = []

    def flush(buf: List[Tuple[int, str]], heading: str, page_no: int):
        if not buf:
            return
        acc = ""
        start = buf[0][0]
        last = start
        for ln, txt in buf:
            if acc and len(acc) + 1 + len(txt) > max_chars:
                chunks.append(ChunkInfo(page=page_no, heading=heading, start_line=start, end_line=last, text=acc.strip()))
                acc = ""
                start = ln
            acc += (" " if acc else "") + txt
            last = ln
        if acc:
            chunks.append(ChunkInfo(page=page_no, heading=heading, start_line=start, end_line=last, text=acc.strip()))

    for p in pages:
        heading = "(General)"
        buf: List[Tuple[int, str]] = []
        for line_idx, line in enumerate(p.lines, start=1):
            if is_heading(line):
                flush(buf, heading, p.number)
                buf = []
                heading = re.sub(r"\s+", " ", line.strip())[:120]
                continue
            buf.append((line_idx, line))
        flush(buf, heading, p.number)
    return chunks

# -----------------------
# BUILD INDEX (FAISS)
# -----------------------
def build_embedder():
    return HuggingFaceEmbeddings(model_name=DEFAULT_EMBED_MODEL)

def chunks_to_documents(chunks: List[ChunkInfo], source_name: str) -> List[Document]:
    docs: List[Document] = []
    for ch in chunks:
        meta = {
            "source": source_name,
            "page": ch.page,
            "heading": ch.heading,
            "line_start": ch.start_line,
            "line_end": ch.end_line,
        }
        docs.append(Document(page_content=ch.text, metadata=meta))
    return docs

def build_faiss_from_uploads(files: List[io.BytesIO]):
    all_docs: List[Document] = []
    total_pages = 0

    for f in files:
        try:
            f.seek(0)
            reader = PdfReader(f)
            total_pages += len(reader.pages)
        except Exception:
            pass
    progress = st.sidebar.progress(0, text="Scanning & chunking…")
    done_pages = 0

    for upl in files:
        name = getattr(upl, "name", "uploaded.pdf")
        try:
            upl.seek(0)
        except Exception:
            pass
        pages = extract_pages_from_pdf(upl)
        done_pages += len(pages)
        if total_pages:
            progress.progress(min(done_pages / total_pages, 1.0), text=f"Scanning & chunking… {done_pages}/{total_pages}")
        chunks = chunk_by_headings(pages)
        all_docs.extend(chunks_to_documents(chunks, source_name=name))

    progress.empty()

    if not all_docs:
        return None, 0

    emb = build_embedder()
    db = FAISS.from_documents(all_docs, emb)
    return db, len(all_docs)

# -----------------------
# RETRIEVAL, RERANK & ANSWER SHAPING
# -----------------------
def retrieve_mmr(db: FAISS, question: str, k=40, fetch_k=120, diversity=0.4) -> List[Document]:
    try:
        return db.max_marginal_relevance_search(question, k=k, fetch_k=fetch_k, lambda_mult=diversity)
    except Exception:
        return db.similarity_search(question, k=k)

def rerank_with_cross_encoder(question: str, docs: List[Document], top_k: int = 24) -> List[Document]:
    if not _CE or not docs:
        return docs[:top_k]
    pairs = [(question, d.page_content) for d in docs]
    try:
        scores = _CE.predict(pairs)
        idxs = np.argsort(-np.array(scores))[:top_k]
        return [docs[i] for i in idxs]
    except Exception:
        return docs[:top_k]

def extract_exact_quotes_from_docs(hit_docs: List[Document], query: str, max_quotes: int = 12) -> List[Dict[str, Any]]:
    # sentence-level selection inside each chunk; line indices approximated by start_line + rel_idx
    q_terms = set(re.findall(r"\b\w+\b", query.lower()))
    quotes: List[Dict[str, Any]] = []

    for d in hit_docs:
        text = d.page_content or ""
        sents = re.split(r"(?<=[.!?])\s+", text)
        for rel_idx, sent in enumerate(sents):
            sent = sent.strip()
            if not sent:
                continue
            s_terms = set(re.findall(r"\b\w+\b", sent.lower()))
            score = len(q_terms & s_terms) / max(1, len(q_terms))
            # 1–2 sentence bullet: keep single sentence for precision; join next if very short
            snippet = sent
            if len(snippet) < 60 and rel_idx + 1 < len(sents):
                nxt = sents[rel_idx + 1].strip()
                if nxt:
                    snippet = (snippet + " " + nxt).strip()
            # trim to ~320 chars
            if len(snippet) > 320:
                snippet = snippet[:320].rsplit(" ", 1)[0] + "…"
            meta = d.metadata or {}
            start_line = int(meta.get("line_start", 1)) + rel_idx
            end_line = min(int(meta.get("line_end", start_line)), start_line)
            quotes.append({
                "text": snippet,
                "source": meta.get("source", "source.pdf"),
                "page": int(meta.get("page", 1)),
                "heading": meta.get("heading", "(General)"),
                "line_start": start_line,
                "line_end": end_line,
                "score": score + 1e-6,
            })

    # Rank: score desc, then early lines first
    quotes.sort(key=lambda x: (-x["score"], x["line_start"]))

    # De-dup by first 140 chars
    out: List[Dict[str, Any]] = []
    seen = set()
    for q in quotes:
        key = (q["source"], q["page"], q["heading"], q["text"][:140])
        if key in seen:
            continue
        seen.add(key)
        out.append(q)
        if len(out) >= max_quotes:
            break
    return out

def format_evidence_markdown(quotes: List[Dict[str, Any]]) -> str:
    lines = []
    for q in quotes:
        lines.append(
            f"- \"{q['text']}\"  \n  <span style='color:#888'>p.{q['page']} L{q['line_start']}-{q['line_end']}, {q['heading']}</span>"
        )
    return "\n".join(lines)

# -----------------------
# REMEDY EXTRACTION (diversity-aware)
# -----------------------
_REMEDY_STOPWORDS = {
    "the","and","with","from","into","that","this","case","cases","medicalgpt","extractive","evidence",
    "question","answer","plan","notes","symptoms","remedies","heading","chapter","general","patient",
    "history","right","left","pain","acute","chronic",
}
_REMEDY_PATTERN = re.compile(r"\b([A-Z][A-Za-z]{2,}(?:\s+(?:[A-Za-z]{2,}|[A-Z][A-Za-z]{1,}))*)")

def _normalise_remedy_name(raw: str) -> Optional[str]:
    cleaned = re.sub(r"[^\w\s-]", " ", raw)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return None
    words = [w.strip("-") for w in cleaned.split() if w.strip("-")]
    if not words:
        return None
    lower_words = [w.lower() for w in words]
    if all(w in _REMEDY_STOPWORDS for w in lower_words):
        return None
    if not any(len(w) > 3 for w in words):
        return None
    words[0] = words[0][:1].upper() + words[0][1:]
    return " ".join(words)

def _extract_candidates_from_text(text: str) -> List[str]:
    if not text:
        return []
    found: List[str] = []
    for match in _REMEDY_PATTERN.findall(text):
        name = _normalise_remedy_name(match)
        if not name:
            continue
        found.append(name)
    return found

def _heading_candidates(heading: str) -> List[str]:
    if not heading:
        return []
    names: List[str] = []
    for part in re.split(r"[;:/\-\u2013\u2014]", heading):
        name = _normalise_remedy_name(part)
        if name:
            names.append(name)
    return names

def _text_tokens(text: str) -> Set[str]:
    return set(re.findall(r"[a-z]{3,}", (text or "").lower()))

def extract_remedies(
    question: str,
    hit_docs: List[Document],
    quotes: List[Dict[str, Any]],
    top_n: int = 5,
    diversity_lambda: float = 0.6,
) -> List[Dict[str, Any]]:
    """Infer likely remedies by scoring headings and quotes across retrieved documents."""
    question_terms = _text_tokens(question)
    candidate_scores: Dict[str, Dict[str, Any]] = {}
    candidate_tokens: Dict[str, Set[str]] = {}
    supports: defaultdict[str, List[Dict[str, Any]]] = defaultdict(list)

    for doc in hit_docs:
        meta = doc.metadata or {}
        heading = meta.get("heading", "")
        doc_candidates = set(_heading_candidates(heading))
        doc_candidates.update(_extract_candidates_from_text(doc.page_content or ""))
        if not doc_candidates:
            continue
        for name in doc_candidates:
            key = name.lower()
            tokens = _text_tokens(name)
            if not tokens:
                continue
            candidate_tokens[key] = tokens
            payload = candidate_scores.setdefault(key, {"name": name, "score": 0.0})
            overlap = len(tokens & question_terms)
            payload["score"] += 1.0 + 0.3 * overlap

    for quote in quotes:
        quote_candidates = set(_extract_candidates_from_text(quote.get("text", "")))
        quote_candidates.update(_heading_candidates(quote.get("heading", "")))
        if not quote_candidates:
            continue
        for name in quote_candidates:
            key = name.lower()
            tokens = candidate_tokens.setdefault(key, _text_tokens(name))
            if not tokens:
                continue
            payload = candidate_scores.setdefault(key, {"name": name, "score": 0.0})
            payload["score"] += 0.5
            if len(supports[key]) < 4:
                supports[key].append(quote)

    if not candidate_scores:
        return []

    ordered = sorted(candidate_scores.items(), key=lambda item: item[1]["score"], reverse=True)
    selected: List[str] = []
    selected_token_sets: List[Set[str]] = []
    diversity_threshold = max(0.2, 1.0 - max(0.0, min(diversity_lambda, 1.0)))

    for key, payload in ordered:
        tokens = candidate_tokens.get(key)
        if not tokens:
            continue
        if any((len(tokens & prev_tokens) / max(len(tokens), 1)) > diversity_threshold for prev_tokens in selected_token_sets):
            continue
        selected.append(key)
        selected_token_sets.append(tokens)
        if len(selected) >= max(1, top_n):
            break

    results: List[Dict[str, Any]] = []
    for key in selected:
        remedy = candidate_scores[key]["name"]
        tokens = candidate_tokens.get(key, set())
        if not supports[key]:
            for quote in quotes:
                quote_tokens = _text_tokens(quote.get("text", ""))
                if tokens & quote_tokens:
                    supports[key].append(quote)
                if len(supports[key]) >= 3:
                    break
        results.append(
            {
                "remedy": remedy,
                "score": round(float(candidate_scores[key]["score"]), 3),
                "support": supports[key][:3],
            }
        )
    return results

# -----------------------
# GOOGLE DRIVE (Service Account)
# -----------------------
def _drive_client():
    if not _HAVE_GDRIVE:
        return None
    sa_file = os.getenv("GDRIVE_SERVICE_ACCOUNT_FILE")
    if not sa_file or not os.path.exists(sa_file):
        return None
    scopes = ["https://www.googleapis.com/auth/drive.file"]
    try:
        creds = Credentials.from_service_account_file(sa_file, scopes=scopes)
        return build("drive", "v3", credentials=creds, cache_discovery=False)
    except Exception:
        return None

def upload_to_drive(local_path: str, folder_id: Optional[str] = None) -> Optional[str]:
    """Upload a file to Google Drive. Returns a shareable web link or None."""
    svc = _drive_client()
    if svc is None:
        return None
    fname = os.path.basename(local_path)
    mime, _ = mimetypes.guess_type(local_path)
    media = MediaFileUpload(local_path, mimetype=mime or "application/octet-stream", resumable=True)
    meta = {"name": fname}
    folder_id = folder_id or os.getenv("GDRIVE_FOLDER_ID")
    if folder_id:
        meta["parents"] = [folder_id]
    try:
        created = svc.files().create(body=meta, media_body=media, fields="id, webViewLink").execute()
        # Make link shareable (anyone with link can view)
        try:
            svc.permissions().create(fileId=created["id"], body={"role": "reader", "type": "anyone"}).execute()
            meta2 = svc.files().get(fileId=created["id"], fields="webViewLink").execute()
            return meta2.get("webViewLink")
        except Exception:
            return created.get("webViewLink")
    except Exception:
        return None

# -----------------------
# GROUNDED GENERATOR (quotes-only)
# -----------------------
SYSTEM_STRICT = (
    "You are a books-only assistant. Use ONLY the provided quotes. "
    "Every sentence must be supportable by a quote. "
    "Include inline citations as (p.X Lstart-Lend, Heading). Keep it concise."
)

def pick_llm() -> Optional[Any]:
    # Prefer OpenRouter (if configured), else Ollama, else None
    if OR_API_KEY and OR_BASE_URL and OR_MODEL:
        try:
            return ChatOpenAI(
                model=OR_MODEL,
                temperature=0.2,
                base_url=OR_BASE_URL,
                api_key=OR_API_KEY,
                default_headers={"HTTP-Referer": OR_SITE_URL, "X-Title": OR_APP_NAME},
            )
        except Exception as e:
            st.sidebar.error(f"OpenRouter init failed: {e}")
    try:
        return ChatOllama(model=DEFAULT_LOCAL_MODEL, temperature=0.2)
    except Exception:
        return None

def generate_grounded_answer(quotes: List[Dict[str, Any]], question: str, llm: Optional[Any]) -> str:
    if not llm or not quotes:
        return ""
    bullets = []
    for q in quotes[:12]:
        citation = f"(p.{q['page']} L{q['line_start']}-{q['line_end']}, {q['heading']})"
        bullets.append(f"- \"{q['text']}\" {citation}")
    user = (
        "Answer the user's question using ONLY these quotes. "
        "Cite support inline after the matching sentence.\n\n" + "\n".join(bullets) +
        f"\n\nQuestion: {question}\n\nWrite a short, neutral answer."
    )
    try:
        resp = llm.invoke([SystemMessage(content=SYSTEM_STRICT), HumanMessage(content=user)])
        return getattr(resp, "content", str(resp))
    except Exception as e:
        return f"[answer error: {e}]"

# -----------------------
# CASE MANAGEMENT (storage + PDF helpers)
# -----------------------
def _safe_patient_id(pid: str) -> str:
    pid = (pid or "").strip()
    if not pid:
        return f"PID-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    pid = re.sub(r"[^A-Za-z0-9._-]+", "_", pid)
    return pid[:64]

def _ensure_csv_header() -> None:
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp","case_id","patient","age","question","top_remedies","answer_present","files"])

def make_case_record_with_patient(
    question: str,
    quotes: List[Dict[str, Any]],
    remedies: List[Dict[str, Any]],
    answer: str,
    patient_info: Dict[str, Any],
    chat_log: List[Dict[str, Any]],
) -> Dict[str, Any]:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pid = _safe_patient_id(patient_info.get("patient_id",""))
    pname = (patient_info.get("patient_name") or "").strip() or "(anonymous)"
    age = (patient_info.get("age") or "").strip()
    brief = (patient_info.get("brief") or "").strip()
    return {
        "timestamp": ts,
        "case_id": pid,
        "patient": pname,
        "age": age,
        "brief": brief,
        "question": question,
        "evidence": quotes,
        "remedies": remedies,
        "answer": answer,
        "chat": chat_log or [],
        "files": [],
    }

def save_case_json(record: Dict[str, Any]) -> str:
    fname = f"{record['case_id']}.json"
    path = os.path.join(CASE_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return path

def append_case_csv(record: Dict[str, Any]) -> str:
    _ensure_csv_header()
    top = ", ".join(r.get("remedy", "?") for r in record.get("remedies", [])[:5])
    files = ", ".join(record.get("files", []))
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                record.get("timestamp", ""),
                record.get("case_id", ""),
                record.get("patient", ""),
                record.get("age", ""),
                record.get("question", ""),
                top,
                bool(record.get("answer")),
                files,
            ]
        )
    return CSV_PATH

def build_pdf_report(record: Dict[str, Any]) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
    )
    styles = getSampleStyleSheet()
    styles["Heading1"].alignment = enums.TA_LEFT
    elems = []

    title = f"MedicalGPT Case — {record['case_id']}"
    subtitle = f"Patient: {record.get('patient','(anonymous)')} | Age: {record.get('age','-')} | {record['timestamp']}"
    elems += [Paragraph(title, styles["Heading1"]), Spacer(1, 6), Paragraph(subtitle, styles["Normal"]), Spacer(1, 12)]

    if record.get("brief"):
        elems += [Paragraph("Brief of the problem", styles["Heading2"]),
                  Paragraph(record["brief"], styles["Normal"]), Spacer(1,8)]

    elems += [Paragraph("Question", styles["Heading2"]), Paragraph(record.get("question", ""), styles["Normal"]), Spacer(1, 8)]

    elems += [Paragraph("Suggested remedies (from books)", styles["Heading2"])]
    rem_bullets = []
    for remedy in record.get("remedies", []):
        rem_bullets.append(ListItem(Paragraph(f"<b>{remedy['remedy']}</b> (score {remedy['score']})", styles["Normal"])))
    if rem_bullets:
        elems.append(ListFlowable(rem_bullets, bulletType="1"))
    elems.append(Spacer(1, 8))

    elems += [Paragraph("Extractive evidence", styles["Heading2"])]
    ev_bullets = []
    for quote in record.get("evidence", [])[:12]:
        cite = f"{quote['source']}, p.{quote['page']} L{quote['line_start']}-{quote['line_end']}, {quote['heading']}"
        ev_bullets.append(ListItem(Paragraph(f"\"{quote['text']}\" — {cite}", styles["Normal"])))
    if ev_bullets:
        elems.append(ListFlowable(ev_bullets, bulletType="bullet"))
    elems.append(Spacer(1, 8))

    if record.get("answer"):
        elems += [
            Paragraph("Answer (from books only)", styles["Heading2"]),
            Paragraph(record["answer"].replace("\n", "<br/>"), styles["Normal"]),
        ]

    doc.build(elems)
    return buf.getvalue()

# -----------------------
# TRANSCRIPT HELPERS (NEW)
# -----------------------
def build_transcript_record(patient_info: Dict[str, Any], chat_log: List[Dict[str, Any]]) -> Dict[str, Any]:
    pid = _safe_patient_id(patient_info.get("patient_id",""))
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "patient_id": pid,
        "patient_name": patient_info.get("patient_name",""),
        "age": patient_info.get("age",""),
        "brief": patient_info.get("brief",""),
        "chat": chat_log or [],  # [{"ts","role","text"}, ...]
        "type": "transcript",
    }

def save_transcript_json(record: Dict[str, Any]) -> str:
    fname = f"{record['patient_id']}.json"
    path = os.path.join(CASE_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return path

# -----------------------
# SIDEBAR
# -----------------------
st.sidebar.header("Library")
library = st.sidebar.text_input("Library name", value=DEFAULT_LIBRARY)

st.sidebar.subheader("Upload & Build Index")
uploads = st.sidebar.file_uploader("Upload authorized PDF books", type=["pdf"], accept_multiple_files=True)
btn_build = st.sidebar.button("🔎 Build / Rebuild Index", use_container_width=True, disabled=not bool(uploads))

# --- NEW: Drive saving toggle
st.sidebar.subheader("Drive saving")
auto_save_drive = st.sidebar.checkbox(
    "Auto-save chat JSON to Drive",
    value=True,
    help="On every Ask, save/update a JSON transcript to Google Drive as <patient_id>.json"
)

if "db" not in st.session_state:
    st.session_state.db = None
if "ready" not in st.session_state:
    st.session_state.ready = False

# --- NEW: patient info + chat transcript state ---
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []  # list of {"ts","role","text"}
if "patient_info" not in st.session_state:
    st.session_state.patient_info = {"patient_name":"", "age":"", "patient_id":"", "brief":""}

if btn_build and uploads:
    with st.spinner("Building heading-aware index…"):
        db, n_docs = build_faiss_from_uploads(uploads)
        if db is None:
            st.sidebar.error("No extractable text found.")
        else:
            st.session_state.db = db
            st.session_state.ready = True
            st.sidebar.success(f"Index ready with {n_docs} chunks.")

# -----------------------
# MAIN UI
# -----------------------
st.title("🩺 MedicalGPT ")
st.caption("Books-only medical assistant with extractive evidence & suggested remedies")

# --- NEW: Patient info section ---
st.subheader("Patient info")
cpi1, cpi2, cpi3 = st.columns([1.6, 0.6, 1.0])
with cpi1:
    st.session_state.patient_info["patient_name"] = st.text_input("Patient name", value=st.session_state.patient_info.get("patient_name",""))
with cpi2:
    st.session_state.patient_info["age"] = st.text_input("Age", value=st.session_state.patient_info.get("age",""), placeholder="e.g., 42")
with cpi3:
    st.session_state.patient_info["patient_id"] = st.text_input("Patient ID (unique)", value=st.session_state.patient_info.get("patient_id",""),
        help="Used as the file name in Drive: <patient_id>.json")

st.session_state.patient_info["brief"] = st.text_area(
    "Brief of the problem", value=st.session_state.patient_info.get("brief",""),
    placeholder="Short clinical summary in your own words…"
)

query = st.text_area("Ask from your books", placeholder="e.g., Remedies for renal colic?")
ask = st.button("Ask", type="primary", use_container_width=True)

if ask:
    if not st.session_state.ready or st.session_state.db is None:
        st.error("Please build the index from the left sidebar first.")
    elif not query.strip():
        st.warning("Type a question.")
    else:
        with st.spinner("Retrieving relevant passages…"):
            prelim = retrieve_mmr(st.session_state.db, query, k=40, fetch_k=120, diversity=0.4)
            hits = rerank_with_cross_encoder(query, prelim, top_k=24)

        # --- Extractive evidence (kept EXACTLY as-is) ---
        quotes = extract_exact_quotes_from_docs(hits, query, max_quotes=12)
        if not quotes:
            st.info("No direct quotes found for this query. Try rephrasing.")
        else:
            st.subheader("Extractive evidence (with citations)")
            st.markdown(format_evidence_markdown(quotes), unsafe_allow_html=True)

            # --- Suggested remedies (diversity-aware) ---
            st.subheader("Suggested remedies (from your books)")
            remedies = extract_remedies(query, hits, quotes, top_n=5, diversity_lambda=0.6)
            if not remedies:
                st.write("No clear remedy mentions surfaced in top passages.")
            else:
                for idx, r in enumerate(remedies, start=1):
                    st.markdown(f"**{idx}. {r['remedy']}** — score {r['score']}")
                    for s in r["support"]:
                        st.markdown(f"&nbsp;&nbsp;• \"{s['text']}\" — {s['source']}, p.{s['page']} L{s['line_start']}-{s['line_end']}, {s['heading']}")

            # --- Grounded LLM answer (quotes-only, inline citations) ---
            llm = pick_llm()
            with st.spinner("Composing grounded answer (from quotes only)…"):
                answer = generate_grounded_answer(quotes, query, llm)
            if answer:
                st.subheader("Answer (from books only)")
                st.write(answer)

            # --- NEW: Append to transcript and auto-save ---
            _now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.chat_log.append({"ts": _now, "role": "user", "text": query})
            if answer:
                st.session_state.chat_log.append({"ts": _now, "role": "assistant", "text": answer})

            # Build a transcript JSON snapshot and (optionally) push to Drive
            _transcript_record = build_transcript_record(
                patient_info=st.session_state.patient_info,
                chat_log=st.session_state.chat_log
            )
            transcript_local_path = save_transcript_json(_transcript_record)
            st.caption(f"Transcript snapshot saved locally: `{os.path.basename(transcript_local_path)}`")

            if auto_save_drive:
                try:
                    folder_id = os.getenv("GDRIVE_FOLDER_ID", "")
                    drive_link = upload_to_drive(transcript_local_path, folder_id)
                    if drive_link:
                        st.info(f"Transcript auto-saved to Drive: {drive_link}")
                    else:
                        st.warning("Auto-save to Drive skipped (service account/file/folder not configured).")
                except Exception as e:
                    st.warning(f"Auto-save to Drive error: {e}")

            # --- Save / Export Case (uses patient_id as case_id) ---
            st.markdown("---")
            st.subheader("Save / Export Case")

            # Reuse patient info already entered above (kept editable here)
            pi = st.session_state.patient_info.copy()
            cols = st.columns(3)
            with cols[0]:
                pi["patient_name"] = st.text_input("Patient name (optional)", value=pi.get("patient_name",""))
            with cols[1]:
                pi["patient_id"] = st.text_input("Patient ID (optional, unique)", value=pi.get("patient_id",""))
            with cols[2]:
                pi["age"] = st.text_input("Age (optional)", value=pi.get("age",""))

            brief_extra = st.text_input("Notes (optional)", value="")
            if brief_extra:
                if pi.get("brief"):
                    pi["brief"] = (pi["brief"] + "\n" + brief_extra).strip()
                else:
                    pi["brief"] = brief_extra

            case_record = make_case_record_with_patient(
                question=query,
                quotes=quotes,
                remedies=remedies,
                answer=answer,
                patient_info=pi,
                chat_log=st.session_state.chat_log
            )

            # Build bytes for immediate download buttons
            case_json_bytes = json.dumps(case_record, ensure_ascii=False, indent=2).encode("utf-8")
            case_pdf_bytes = build_pdf_report(case_record)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.download_button(
                    "⬇️ Download Case (JSON)",
                    data=case_json_bytes,
                    file_name=f"{case_record['case_id']}.json",
                    mime="application/json",
                    use_container_width=True,
                )
            with c2:
                st.download_button(
                    "⬇️ Download Report (PDF)",
                    data=case_pdf_bytes,
                    file_name=f"{case_record['case_id']}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            with c3:
                if st.button("💾 Save Locally (JSON + CSV + PDF)", use_container_width=True):
                    saved = []
                    # Save JSON
                    json_path = save_case_json(case_record)
                    saved.append(json_path)
                    # Save PDF
                    pdf_path = os.path.join(REPORT_DIR, f"{case_record['case_id']}.pdf")
                    with open(pdf_path, "wb") as f:
                        f.write(case_pdf_bytes)
                    saved.append(pdf_path)
                    # Index in CSV
                    case_record["files"] = saved
                    csv_path = append_case_csv(case_record)
                    st.success(f"Saved: {json_path}\nSaved: {pdf_path}\nIndexed in: {csv_path}")

                    # Also upload to Google Drive if configured
                    drive_json = drive_pdf = None
                    try:
                        folder_id = os.getenv("GDRIVE_FOLDER_ID", "")
                        drive_json = upload_to_drive(json_path, folder_id)
                        drive_pdf  = upload_to_drive(pdf_path,  folder_id)
                    except Exception as e:
                        st.warning(f"Drive upload skipped: {e}")

                    if drive_json or drive_pdf:
                        st.info(
                            "Uploaded to Google Drive:\n"
                            f"- JSON: {drive_json or '(no link)'}\n"
                            f"- PDF: {drive_pdf or '(no link)'}"
                        )

# Developer expander
with st.expander("Developer • Inspect first hits"):
    if 'db' in st.session_state and st.session_state.db is not None and 'prelim' in locals():
        raw = [d.metadata for d in (prelim[:5] if prelim else [])]
        st.json(raw)
