"""
ShopVista Embedding & Vector Database Pipeline
===============================================
Loads production chunks, embeds them using a multilingual sentence-transformer,
stores in ChromaDB, and runs cross-lingual retrieval tests.

Embedding model : sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
Vector database : ChromaDB (persistent, local)
"""

import json
import logging
import os
import time
import warnings
from pathlib import Path

from dotenv import load_dotenv

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
CHUNKS_PATH = BASE_DIR / "data" / "chunks" / "chunks.jsonl"
VECTORDB_DIR = BASE_DIR / "data" / "vectordb"

load_dotenv(BASE_DIR / ".env")
os.environ.setdefault("HF_TOKEN", os.getenv("HF_API_KEY", ""))
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", message=".*UNEXPECTED.*")
logging.getLogger("transformers").setLevel(logging.ERROR)

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "customer_support_chunks"

# ---------------------------------------------------------------------------
# Chunk Loading
# ---------------------------------------------------------------------------

def load_chunks(path: Path = CHUNKS_PATH) -> list[dict]:
    """Load production chunks from JSONL file."""
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    print(f"  Loaded {len(chunks)} chunks from {path.name}")
    return chunks


# ---------------------------------------------------------------------------
# ChromaDB Setup & Indexing
# ---------------------------------------------------------------------------

def get_embedding_function():
    """Create SentenceTransformer embedding function for ChromaDB."""
    return SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME)


def create_collection(embedding_fn) -> chromadb.Collection:
    """Create or get a persistent ChromaDB collection."""
    client = chromadb.PersistentClient(path=str(VECTORDB_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def index_chunks(collection: chromadb.Collection, chunks: list[dict]) -> None:
    """Upsert all chunks into the ChromaDB collection."""
    ids = []
    documents = []
    metadatas = []

    for chunk in chunks:
        chunk_id = f"{chunk['source']}_{chunk['chunk_id']}"
        ids.append(chunk_id)
        documents.append(chunk["text"])
        metadatas.append({
            "source": chunk["source"],
            "language": chunk["language"],
            "chunk_id": chunk["chunk_id"],
            "chunk_method": chunk["chunk_method"],
            "chunk_size_setting": chunk["chunk_size_setting"],
            "token_count": chunk["token_count"],
        })

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    print(f"  Upserted {len(ids)} chunks into collection '{COLLECTION_NAME}'")


# ---------------------------------------------------------------------------
# Retrieval Test
# ---------------------------------------------------------------------------

def retrieval_test(collection: chromadb.Collection) -> None:
    """Run cross-lingual retrieval tests and compare results."""
    queries = {
        "TR": "İade süresi kaç gündür?",
        "EN": "What is the return period?",
    }

    results_by_lang: dict[str, list[str]] = {}

    for lang, query in queries.items():
        print(f"\n{'='*70}")
        print(f"  QUERY ({lang}): \"{query}\"")
        print(f"{'='*70}")

        result = collection.query(query_texts=[query], n_results=5)

        retrieved_ids = result["ids"][0]
        distances = result["distances"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]

        results_by_lang[lang] = retrieved_ids

        for rank, (doc_id, dist, doc, meta) in enumerate(
            zip(retrieved_ids, distances, documents, metadatas), start=1
        ):
            preview = doc[:200].replace("\n", " ")
            print(f"\n  #{rank}  ID: {doc_id}")
            print(f"       Source: {meta['source']}  |  Language: {meta['language']}  |  Tokens: {meta['token_count']}")
            print(f"       Cosine Distance: {dist:.4f}")
            print(f"       Preview: {preview}...")

    # --- Cross-lingual Comparison ---
    cross_lingual_comparison(results_by_lang)


def cross_lingual_comparison(results_by_lang: dict[str, list[str]]) -> None:
    """Compare TR and EN retrieval results and print analysis."""
    tr_ids = results_by_lang["TR"]
    en_ids = results_by_lang["EN"]

    tr_set = set(tr_ids)
    en_set = set(en_ids)

    overlap = tr_set & en_set
    only_tr = tr_set - en_set
    only_en = en_set - tr_set

    print(f"\n{'='*70}")
    print("  CROSS-LINGUAL COMPARISON")
    print(f"{'='*70}")
    print(f"\n  TR top-5 IDs: {tr_ids}")
    print(f"  EN top-5 IDs: {en_ids}")
    print(f"\n  Overlap      : {len(overlap)}/5 — {sorted(overlap)}")
    print(f"  Only in TR   : {sorted(only_tr)}")
    print(f"  Only in EN   : {sorted(only_en)}")

    # Rank comparison for overlapping chunks
    if overlap:
        print(f"\n  Rank comparison for overlapping chunks:")
        for chunk_id in sorted(overlap):
            tr_rank = tr_ids.index(chunk_id) + 1
            en_rank = en_ids.index(chunk_id) + 1
            print(f"    {chunk_id}: TR rank #{tr_rank}, EN rank #{en_rank}")

    # Interpretation
    overlap_pct = len(overlap) / 5 * 100
    print(f"\n  INTERPRETATION:")
    if overlap_pct >= 80:
        print(f"  ✓ Excellent cross-lingual alignment ({overlap_pct:.0f}% overlap).")
        print(f"    The multilingual model successfully maps both Turkish and English")
        print(f"    queries to the same semantic region in the vector space.")
    elif overlap_pct >= 60:
        print(f"  ✓ Good cross-lingual alignment ({overlap_pct:.0f}% overlap).")
        print(f"    The model captures the core semantic meaning across languages,")
        print(f"    with minor differences in peripheral results.")
    elif overlap_pct >= 40:
        print(f"  ~ Moderate cross-lingual alignment ({overlap_pct:.0f}% overlap).")
        print(f"    The model partially bridges the language gap. Some language-specific")
        print(f"    lexical patterns may be influencing retrieval.")
    else:
        print(f"  ✗ Low cross-lingual alignment ({overlap_pct:.0f}% overlap).")
        print(f"    The model struggles to map these queries to the same semantic space.")
        print(f"    Consider using a larger or more capable multilingual embedding model.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  ShopVista — Embedding & Vector Database Pipeline")
    print("=" * 70)

    # Phase 1: Load chunks
    print("\n[1/3] Loading production chunks...")
    chunks = load_chunks()

    # Phase 2: Embed & index
    print("\n[2/3] Embedding & indexing into ChromaDB...")
    print(f"  Model: {MODEL_NAME}")
    print(f"  DB path: {VECTORDB_DIR}")

    t0 = time.perf_counter()
    embedding_fn = get_embedding_function()
    collection = create_collection(embedding_fn)
    index_chunks(collection, chunks)
    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f}s")

    # Phase 3: Retrieval test
    print("\n[3/3] Running cross-lingual retrieval test...")
    retrieval_test(collection)

    print(f"\n{'='*70}")
    print("  Pipeline complete.")
    print(f"  ChromaDB persisted at: {VECTORDB_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
