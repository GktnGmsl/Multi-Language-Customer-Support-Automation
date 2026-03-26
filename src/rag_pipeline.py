"""
ShopVista RAG Pipeline
=======================
Retrieval-Augmented Generation pipeline for multi-language customer support.

Flow: Question → Language Detection → Embedding + Retrieval → Prompt → LLM → Answer
LLM    : Google Gemini 2.5 Flash Lite
Vector : ChromaDB (from Task 3)
"""

import logging
import os
import warnings
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors
from langdetect import detect

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from api_key_rotator import API_KEY_ROTATOR
from conversation_manager import ConversationManager

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
LLM_MODEL = "gemini-2.5-flash-lite"
TOP_K = 3

conv_manager = ConversationManager()

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

def build_prompt(query: str, retrieved: dict, history: list = None) -> str:
    """Build the user prompt with context from retrieved chunks and conversation history."""
    context_parts = []
    for i, (doc, meta) in enumerate(
        zip(retrieved["documents"], retrieved["metadatas"]), start=1
    ):
        context_parts.append(
            f"[Chunk {i} | source: {meta['source']} | language: {meta['language']}]\n{doc}"
        )
    context_block = "\n\n---\n\n".join(context_parts)
    
    history_block = ""
    if history:
        history_parts = []
        for msg in history:
            role = "Kullanıcı" if msg["role"] == "user" else "Asistan"
            history_parts.append(f"{role}: {msg['content']}")
        history_block = "<conversation_history>\n" + "\n".join(history_parts) + "\n</conversation_history>\n\n"

    return f"{history_block}<context>\n{context_block}\n</context>\n\nKullanıcı sorusu: {query}"


# ---------------------------------------------------------------------------
# LLM Call
# ---------------------------------------------------------------------------

def call_llm(system_prompt: str, user_prompt: str) -> str:
    """Send prompt to Gemini and return the response text (with rotation)."""
    max_retries = 15
    base_delay = 2
    for attempt in range(max_retries):
        current_key = next(API_KEY_ROTATOR)
        try:
            client = genai.Client(api_key=current_key)
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
        except errors.APIError as e:
            delay = base_delay * (1.5 ** (attempt // 3))
            logging.error(f"API Error with key (ending in ...{current_key[-4:]}): {e.message}. Retrying in {delay}s...")
            time.sleep(delay)
        except Exception as e:
            delay = base_delay * (1.5 ** (attempt // 3))
            logging.error(f"Unexpected error: {e}. Retrying in {delay}s...")
            time.sleep(delay)
            
    return "Özür dilerim, sistemlerimizde geçici bir yoğunluk var. / System is currently busy."


# ---------------------------------------------------------------------------
# RAG Pipeline
# ---------------------------------------------------------------------------

def ask(query: str, collection: chromadb.Collection | None = None, session_id: str = "default") -> dict:
    """
    Full RAG pipeline: detect language → retrieve → prompt → LLM → answer.

    Returns a dict with: answer, language, sources, retrieved_chunks.
    """
    # 1. Language detection
    language = detect_language(query)

    # 2. Add query to conversation history (temporary, we'll store final later)
    # Actually wait, we should pass history to prompt, then record Q and A.
    history = conv_manager.get_history(session_id)

    # 3. Retrieve relevant chunks
    if collection is None:
        collection = get_collection()
    
    # Optional: Improve retrieval by appending last history context to query
    search_query = query
    if history and len(history) > 0:
        # Just a naive heuristic: append the previous user question to contextualize query for retrieval
        last_qs = [m["content"] for m in history if m["role"] == "user"]
        if last_qs:
            search_query = last_qs[-1] + " " + query
            
    retrieved = retrieve(collection, search_query)

    # 4. Build prompt
    user_prompt = build_prompt(query, retrieved, history)

    # 5. Call LLM
    answer = call_llm(SYSTEM_PROMPT, user_prompt)
    
    # 6. Save to history
    conv_manager.add_message(session_id, "user", query)
    conv_manager.add_message(session_id, "assistant", answer)

    # 7. Collect source info
    sources = []
    retrieved_texts = retrieved["documents"]
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
        "contexts": retrieved_texts,
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
