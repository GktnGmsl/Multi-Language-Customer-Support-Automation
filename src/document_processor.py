"""
ShopVista Document Processing & Chunking Pipeline
==================================================
Loads raw company documents, cleans them, splits into chunks with metadata,
and exports as .jsonl for downstream RAG ingestion.
"""

import json
import os
import re
import statistics
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DOCS_DIR = BASE_DIR / "data" / "raw_docs"
CHUNKS_DIR = BASE_DIR / "data" / "chunks"

# Language detection heuristic: files with these names are English
ENGLISH_FILES = {"shipping_policy.txt", "returns_en.txt"}

# Mapping for documents that contain mixed languages
MIXED_LANG_FILES = {"sss.txt"}

# Default chunking parameters
DEFAULT_CHUNK_SIZE = 500      # tokens
DEFAULT_CHUNK_OVERLAP = 50    # tokens


# ---------------------------------------------------------------------------
# Data Model
# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    text: str
    source: str
    language: str
    chunk_id: int
    chunk_method: str = ""
    chunk_size_setting: int = 0
    token_count: int = 0


# ---------------------------------------------------------------------------
# 1. Document Loading & Cleaning
# ---------------------------------------------------------------------------
def load_document(filepath: Path) -> str:
    """Load a text file, trying utf-8 first then latin-1 as fallback."""
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return filepath.read_text(encoding=encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"Cannot decode file: {filepath}")


def clean_text(text: str) -> str:
    """Normalize unicode, fix encoding artefacts, collapse excessive whitespace."""
    # Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)
    # Replace common encoding artefacts
    text = text.replace("\u00a0", " ")   # non-breaking space
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse 3+ consecutive newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse runs of spaces/tabs on the same line (preserve newlines)
    text = re.sub(r"[^\S\n]+", " ", text)
    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    return text.strip()


def detect_language(filename: str, text_segment: str = "") -> str:
    """Determine language based on filename and content heuristics."""
    if filename in ENGLISH_FILES:
        return "en"
    if filename in MIXED_LANG_FILES:
        return "tr+en"
    return "tr"


def detect_chunk_language(text: str, source_lang: str) -> str:
    """For mixed-language files, detect language of an individual chunk."""
    if source_lang != "tr+en":
        return source_lang
    # Simple heuristic: count Turkish-specific characters vs English markers
    tr_indicators = len(re.findall(r"[çğıöşüÇĞİÖŞÜ]", text))
    en_pattern = len(re.findall(r"\b(the|and|is|are|you|your|for|with|this|that|have|from|will|can)\b", text, re.I))
    if en_pattern > tr_indicators:
        return "en"
    if tr_indicators > 0:
        return "tr"
    return source_lang


# ---------------------------------------------------------------------------
# 2. Tokenization (word-level approximation)
# ---------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for chunk-size accounting."""
    return re.findall(r"\S+", text)


def token_count(text: str) -> int:
    return len(tokenize(text))


# ---------------------------------------------------------------------------
# 3. Chunking Strategies
# ---------------------------------------------------------------------------
def _split_into_chunks_by_tokens(segments: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Given a list of text segments, merge/split them into token-bounded chunks."""
    chunks: list[str] = []
    current_tokens: list[str] = []

    for segment in segments:
        seg_tokens = tokenize(segment)
        if not seg_tokens:
            continue

        if len(current_tokens) + len(seg_tokens) <= chunk_size:
            current_tokens.extend(seg_tokens)
        else:
            if current_tokens:
                chunks.append(" ".join(current_tokens))
                # Keep overlap tokens from the end
                current_tokens = current_tokens[-overlap:] if overlap > 0 else []
            # If this single segment exceeds chunk_size, force-split it
            while len(seg_tokens) > chunk_size:
                combined = current_tokens + seg_tokens[:chunk_size - len(current_tokens)]
                chunks.append(" ".join(combined))
                seg_tokens = seg_tokens[chunk_size - len(current_tokens) - overlap:]
                current_tokens = chunks[-1].split()[-overlap:] if overlap > 0 else []
            current_tokens.extend(seg_tokens)

    if current_tokens:
        chunks.append(" ".join(current_tokens))

    return chunks


def recursive_character_split(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE,
                               overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    """
    Recursive character text splitter — mirrors LangChain's approach.
    Tries to split on: section breaks > paragraphs > sentences > words.
    """
    separators = [
        r"(?=\n\n\d+\.\s+[A-ZÇĞİÖŞÜ])",   # numbered section headers (lookahead)
        "\n\n",                            # paragraph breaks
        "\n",                              # line breaks
        r"(?<=[.!?])\s+",                 # sentence boundaries
        " ",                               # word boundaries
    ]

    segments = [text]
    for sep in separators:
        new_segments: list[str] = []
        for seg in segments:
            if token_count(seg) <= chunk_size:
                new_segments.append(seg)
            else:
                parts = re.split(sep, seg)
                new_segments.extend(p for p in parts if p.strip())
        segments = new_segments

    return _split_into_chunks_by_tokens(segments, chunk_size, overlap)


def sentence_split(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE,
                   overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    """
    Sentence-based splitter — splits on sentence boundaries first,
    then groups sentences into token-bounded chunks.
    """
    sentences = re.split(r"(?<=[.!?:])[\s\n]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return _split_into_chunks_by_tokens(sentences, chunk_size, overlap)


def fixed_size_split(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE,
                     overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    """
    Naive fixed-size token splitter — no awareness of structure.
    Splits purely on token count boundaries.
    """
    tokens = tokenize(text)
    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(" ".join(tokens[start:end]))
        start += chunk_size - overlap
    return chunks


CHUNKING_METHODS = {
    "recursive": recursive_character_split,
    "sentence":  sentence_split,
    "fixed":     fixed_size_split,
}


# ---------------------------------------------------------------------------
# 4. Pipeline: process all documents
# ---------------------------------------------------------------------------
def process_documents(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    method: str = "recursive",
) -> list[Chunk]:
    """Load, clean, chunk all documents and return list of Chunk objects."""
    splitter = CHUNKING_METHODS[method]
    all_chunks: list[Chunk] = []

    doc_files = sorted(RAW_DOCS_DIR.glob("*.txt"))
    if not doc_files:
        raise FileNotFoundError(f"No .txt files found in {RAW_DOCS_DIR}")

    for filepath in doc_files:
        raw = load_document(filepath)
        cleaned = clean_text(raw)
        source_lang = detect_language(filepath.name)

        text_chunks = splitter(cleaned, chunk_size, overlap)

        for i, chunk_text in enumerate(text_chunks):
            lang = detect_chunk_language(chunk_text, source_lang)
            chunk = Chunk(
                text=chunk_text,
                source=filepath.name,
                language=lang,
                chunk_id=i,
                chunk_method=method,
                chunk_size_setting=chunk_size,
                token_count=token_count(chunk_text),
            )
            all_chunks.append(chunk)

    return all_chunks


# ---------------------------------------------------------------------------
# 5. Export & Reporting
# ---------------------------------------------------------------------------
def save_chunks_jsonl(chunks: list[Chunk], output_path: Path) -> None:
    """Save chunks as .jsonl (one JSON object per line)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def report_stats(chunks: list[Chunk], label: str = "") -> dict:
    """Print and return chunk statistics."""
    token_counts = [c.token_count for c in chunks]
    stats = {
        "label": label,
        "total_chunks": len(chunks),
        "avg_tokens": round(statistics.mean(token_counts), 1) if token_counts else 0,
        "min_tokens": min(token_counts) if token_counts else 0,
        "max_tokens": max(token_counts) if token_counts else 0,
        "median_tokens": round(statistics.median(token_counts), 1) if token_counts else 0,
        "by_source": {},
    }
    # Per-source breakdown
    sources = sorted(set(c.source for c in chunks))
    for src in sources:
        src_counts = [c.token_count for c in chunks if c.source == src]
        stats["by_source"][src] = {
            "chunks": len(src_counts),
            "avg_tokens": round(statistics.mean(src_counts), 1),
        }

    print(f"\n{'=' * 60}")
    print(f"  {label or 'Chunk Statistics'}")
    print(f"{'=' * 60}")
    print(f"  Total chunks    : {stats['total_chunks']}")
    print(f"  Avg tokens      : {stats['avg_tokens']}")
    print(f"  Min tokens      : {stats['min_tokens']}")
    print(f"  Max tokens      : {stats['max_tokens']}")
    print(f"  Median tokens   : {stats['median_tokens']}")
    print(f"  --- Per source ---")
    for src, info in stats["by_source"].items():
        print(f"    {src:30s} → {info['chunks']:3d} chunks, avg {info['avg_tokens']} tokens")
    print(f"{'=' * 60}\n")
    return stats


# ---------------------------------------------------------------------------
# 6. Experiment Runner
# ---------------------------------------------------------------------------
def run_experiments() -> list[dict]:
    """
    Run chunking experiments across different sizes and methods.
    Returns a list of stat dicts for README reporting.
    """
    experiments = []

    # --- Experiment A: Different chunk sizes (recursive method) ---
    for size in (256, 500, 1000):
        label = f"recursive | chunk_size={size} | overlap={DEFAULT_CHUNK_OVERLAP}"
        chunks = process_documents(chunk_size=size, overlap=DEFAULT_CHUNK_OVERLAP, method="recursive")
        stats = report_stats(chunks, label)
        experiments.append(stats)

        # Save each experiment
        out_path = CHUNKS_DIR / f"chunks_recursive_{size}.jsonl"
        save_chunks_jsonl(chunks, out_path)

    # --- Experiment B: Different methods (at chunk_size=500) ---
    for method in ("recursive", "sentence", "fixed"):
        label = f"{method} | chunk_size=500 | overlap={DEFAULT_CHUNK_OVERLAP}"
        chunks = process_documents(chunk_size=500, overlap=DEFAULT_CHUNK_OVERLAP, method=method)
        stats = report_stats(chunks, label)
        experiments.append(stats)

        out_path = CHUNKS_DIR / f"chunks_{method}_500.jsonl"
        save_chunks_jsonl(chunks, out_path)

    return experiments


# ---------------------------------------------------------------------------
# 7. Main Entry Point
# ---------------------------------------------------------------------------
def main():
    print("ShopVista Document Processing Pipeline")
    print("=" * 60)

    # --- Run all experiments and collect stats ---
    all_stats = run_experiments()

    # --- Produce the final production chunks (recursive, 500 tokens) ---
    print("\n>>> Generating production chunks: recursive, 500 tokens <<<")
    production_chunks = process_documents(chunk_size=500, overlap=50, method="recursive")
    final_stats = report_stats(production_chunks, "PRODUCTION: recursive | 500 tokens | overlap 50")

    output_path = CHUNKS_DIR / "chunks.jsonl"
    save_chunks_jsonl(production_chunks, output_path)
    print(f"Production chunks saved to: {output_path}")

    summary_path = CHUNKS_DIR / "experiment_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"production": final_stats, "experiments": all_stats}, f,
                  ensure_ascii=False, indent=2)
    print(f"Experiment summary saved to: {summary_path}")

    return production_chunks, all_stats


if __name__ == "__main__":
    main()
