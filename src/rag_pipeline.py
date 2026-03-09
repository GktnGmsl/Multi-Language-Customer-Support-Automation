"""
ShopVista RAG Pipeline
=======================
Retrieval-Augmented Generation pipeline for multi-language customer support.

Flow: Question → Language Detection → Embedding + Retrieval → Prompt → LLM → Answer
LLM    : Google Gemini 2.5 Flash
Vector : ChromaDB (from Task 3)
"""

import logging
import os
import warnings
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from langdetect import detect

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
VECTORDB_DIR = BASE_DIR / "data" / "vectordb"

load_dotenv(BASE_DIR / ".env")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
os.environ.setdefault("HF_TOKEN", os.getenv("HF_API_KEY", ""))
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", message=".*UNEXPECTED.*")
logging.getLogger("transformers").setLevel(logging.ERROR)

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "customer_support_chunks"
LLM_MODEL = "gemini-2.5-flash"
TOP_K = 3

# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
Sen ShopVista müşteri destek asistanısın. Aşağıdaki kurallara kesinlikle uy:

1. SADECE sana verilen <context> bölümündeki bilgilere dayanarak cevap ver. \
Context dışında bilgi uydurma, tahmin yürütme veya genel bilgi ekleme.
2. Eğer sorunun cevabı verilen context'te yoksa, şu şekilde yanıt ver:
   - Türkçe soru için: "Bu konuda bilgim yok, müşteri hizmetlerimizi arayabilirsiniz: 0850 123 45 67"
   - İngilizce soru için: "I don't have information on this topic. Please contact our customer service: +90 850 123 45 67"
3. Cevabının dilini, sorunun diliyle eşleştir. Türkçe soruya Türkçe, İngilizce soruya İngilizce cevap ver.
4. Cevabın sonunda hangi kaynak dokümanları kullandığını belirt:
   - Türkçe: "📄 Kaynak: [dosya adları]"
   - İngilizce: "📄 Source: [file names]"
5. Cevaplarını kısa, net ve müşteri dostu tut. Gereksiz tekrar yapma.
"""

# ---------------------------------------------------------------------------
# Language Detection
# ---------------------------------------------------------------------------

LANG_MAP = {"tr": "tr", "en": "en"}


def detect_language(text: str) -> str:
    """Detect the language of the input text. Returns 'tr' or 'en'."""
    try:
        detected = detect(text)
        return LANG_MAP.get(detected, "en")
    except Exception:
        return "en"


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def get_collection() -> chromadb.Collection:
    """Load the existing ChromaDB collection."""
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=str(VECTORDB_DIR))
    return client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )


def retrieve(collection: chromadb.Collection, query: str, top_k: int = TOP_K) -> dict:
    """Retrieve top-k relevant chunks from ChromaDB."""
    result = collection.query(query_texts=[query], n_results=top_k)
    return {
        "ids": result["ids"][0],
        "documents": result["documents"][0],
        "metadatas": result["metadatas"][0],
        "distances": result["distances"][0],
    }


# ---------------------------------------------------------------------------
# Prompt Building
# ---------------------------------------------------------------------------

def build_prompt(query: str, retrieved: dict) -> str:
    """Build the user prompt with context from retrieved chunks."""
    context_parts = []
    for i, (doc, meta) in enumerate(
        zip(retrieved["documents"], retrieved["metadatas"]), start=1
    ):
        context_parts.append(
            f"[Chunk {i} | source: {meta['source']} | language: {meta['language']}]\n{doc}"
        )
    context_block = "\n\n---\n\n".join(context_parts)

    return f"<context>\n{context_block}\n</context>\n\nKullanıcı sorusu: {query}"


# ---------------------------------------------------------------------------
# LLM Call
# ---------------------------------------------------------------------------

def call_llm(system_prompt: str, user_prompt: str) -> str:
    """Send prompt to Gemini and return the response text."""
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=user_prompt,
        config={
            "system_instruction": system_prompt,
            "temperature": 0.3,
            "max_output_tokens": 1024,
        },
    )
    return response.text


# ---------------------------------------------------------------------------
# RAG Pipeline
# ---------------------------------------------------------------------------

def ask(query: str, collection: chromadb.Collection | None = None) -> dict:
    """
    Full RAG pipeline: detect language → retrieve → prompt → LLM → answer.

    Returns a dict with: answer, language, sources, retrieved_chunks.
    """
    # 1. Language detection
    language = detect_language(query)

    # 2. Retrieve relevant chunks
    if collection is None:
        collection = get_collection()
    retrieved = retrieve(collection, query)

    # 3. Build prompt
    user_prompt = build_prompt(query, retrieved)

    # 4. Call LLM
    answer = call_llm(SYSTEM_PROMPT, user_prompt)

    # 5. Collect source info
    sources = []
    for meta, dist in zip(retrieved["metadatas"], retrieved["distances"]):
        sources.append({
            "source": meta["source"],
            "language": meta["language"],
            "chunk_id": meta["chunk_id"],
            "distance": round(dist, 4),
        })

    return {
        "query": query,
        "detected_language": language,
        "answer": answer,
        "sources": sources,
    }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def print_result(result: dict) -> None:
    """Pretty-print a RAG result."""
    print(f"\n{'='*70}")
    print(f"  SORU / QUESTION: {result['query']}")
    print(f"  Detected Language: {result['detected_language']}")
    print(f"{'='*70}")
    print(f"\n{result['answer']}")
    print(f"\n  Retrieved Sources:")
    for i, src in enumerate(result["sources"], 1):
        print(f"    #{i} {src['source']} (chunk {src['chunk_id']}, "
              f"lang={src['language']}, dist={src['distance']})")


def main():
    print("=" * 70)
    print("  ShopVista — RAG Pipeline Demo")
    print("=" * 70)

    collection = get_collection()

    test_questions = [
        "İade süresi kaç gündür?",
        "What is your shipping policy for international orders?",
        "Garanti süresi ne kadar?",
        "How can I track my order?",
        "Hangi ödeme yöntemlerini kabul ediyorsunuz?",
    ]

    for question in test_questions:
        result = ask(question, collection)
        print_result(result)

    print(f"\n{'='*70}")
    print("  Demo complete.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
