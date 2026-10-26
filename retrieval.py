"""Load processed academic chunks and search them with Sentence Transformers + FAISS.

Supported JSON layouts are deliberately explicit: a chunk record (``text``,
``page_content``, ``content`` or ``chunk_text``), a list of records, or an object
containing a ``chunks``, ``documents``, ``pages``, ``items`` or ``data`` list.
Unknown layouts raise an error naming the file so the document-processing owner
can confirm the schema instead of silently losing content.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import faiss
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data" / "processed"
_TEXT_KEYS = ("text", "page_content", "content", "chunk_text")
_CONTAINER_KEYS = ("chunks", "documents", "pages", "items", "data")
_SOURCE_KEYS = ("source", "document", "document_name", "filename", "file_name")
_PAGE_KEYS = ("page", "page_number", "page_num")
_CHUNK_KEYS = ("chunk", "chunk_id", "chunk_index", "index")


def _first_value(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None and not isinstance(value, (dict, list)):
            return value
    return None


def _merge_metadata(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    """Carry only useful source/page identifiers from document wrappers."""
    merged = dict(parent)
    nested = child.get("metadata")
    if isinstance(nested, dict):
        merged.update(nested)
    for key in (*_SOURCE_KEYS, *_PAGE_KEYS, *_CHUNK_KEYS, "section"):
        if key in child and child[key] is not None:
            merged[key] = child[key]
    return merged


def _records_from_json(payload: Any, file_path: Path) -> list[dict[str, Any]]:
    extracted: list[dict[str, Any]] = []

    def visit(node: Any, inherited: dict[str, Any], location: str) -> None:
        if isinstance(node, list):
            if not node:
                return
            for index, item in enumerate(node):
                visit(item, inherited, f"{location}[{index}]")
            return

        if isinstance(node, str):
            if node.strip():
                record: dict[str, Any] = {"text": node}
                source = _first_value(inherited, _SOURCE_KEYS)
                record["source"] = source if source is not None else file_path.name
                page = _first_value(inherited, _PAGE_KEYS)
                chunk_id = _first_value(inherited, _CHUNK_KEYS)
                if page is not None:
                    record["page"] = page
                if chunk_id is not None:
                    record["chunk"] = chunk_id
                if inherited.get("section") is not None:
                    record["section"] = inherited["section"]
                extracted.append(record)
            return

        if not isinstance(node, dict):
            raise ValueError(f"{location} must be a chunk object, string, or list; found {type(node).__name__}.")

        metadata = node.get("metadata")
        combined = _merge_metadata(inherited, node)
        if isinstance(metadata, dict):
            combined.update(metadata)

        text_key = next((key for key in _TEXT_KEYS if isinstance(node.get(key), str)), None)
        if text_key is not None:
            text = node[text_key]
            if text.strip():
                record: dict[str, Any] = {"text": text}
                source = _first_value(combined, _SOURCE_KEYS)
                page = _first_value(combined, _PAGE_KEYS)
                chunk_id = _first_value(combined, _CHUNK_KEYS)
                if source is not None:
                    record["source"] = source
                else:
                    record["source"] = file_path.name
                if page is not None:
                    record["page"] = page
                if chunk_id is not None:
                    record["chunk"] = chunk_id
                section = combined.get("section")
                if section is not None:
                    record["section"] = section
                extracted.append(record)
            return

        containers = [key for key in _CONTAINER_KEYS if isinstance(node.get(key), list)]
        if len(containers) == 1:
            key = containers[0]
            for index, item in enumerate(node[key]):
                visit(item, combined, f"{location}.{key}[{index}]")
            return
        if len(containers) > 1:
            raise ValueError(f"{location} has multiple possible chunk lists ({', '.join(containers)}); confirm its schema.")

        raise ValueError(
            f"Unsupported JSON structure at {location}. Expected a chunk record with one of "
            f"{', '.join(_TEXT_KEYS)}, a list of records, or a supported list wrapper "
            f"({', '.join(_CONTAINER_KEYS)})."
        )

    visit(payload, {}, str(file_path))
    return extracted


def load_chunks(data_dir: str | Path = DEFAULT_DATA_DIR) -> list[dict[str, Any]]:
    """Read chunk JSON files recursively and validate each file's structure."""
    directory = Path(data_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Processed chunk directory does not exist: {directory}")
    files = sorted(directory.rglob("*.json"))
    if not files:
        raise FileNotFoundError(f"No .json chunk files found in {directory}")

    chunks: list[dict[str, Any]] = []
    for file_path in files:
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8-sig"))
            file_chunks = _records_from_json(payload, file_path)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            raise ValueError(f"Could not load processed chunks from {file_path}: {exc}") from exc
        if not file_chunks:
            raise ValueError(f"No text chunks found in {file_path}; confirm the JSON schema and content.")
        chunks.extend(file_chunks)

    if not chunks:
        raise ValueError(f"No non-empty chunks found under {directory}")
    return chunks


class FAISSRetriever:
    """Build and search a cosine-similarity FAISS index over academic chunks."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index: faiss.Index | None = None
        self.chunks: list[dict[str, Any]] = []

    def build_index(self, chunks: Iterable[str | dict[str, Any]]) -> None:
        prepared: list[dict[str, Any]] = []
        for chunk in chunks:
            if isinstance(chunk, str):
                record: dict[str, Any] = {"text": chunk}
            elif isinstance(chunk, dict) and isinstance(chunk.get("text"), str):
                record = dict(chunk)
            else:
                raise ValueError("Each chunk must be a string or a dictionary with a string 'text' field.")
            if record["text"].strip():
                prepared.append(record)
        if not prepared:
            raise ValueError("Provide at least one non-empty chunk.")

        embeddings = self.model.encode(
            [record["text"] for record in prepared],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embeddings = embeddings.astype("float32")
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        self.chunks = prepared
        self.index = index

    def search(self, question: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Return actual chunk text, metadata, and cosine similarity scores."""
        if self.index is None:
            raise RuntimeError("Build the FAISS index before searching.")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("The question cannot be empty.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query_vector = self.model.encode(
            [question], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        scores, indices = self.index.search(query_vector, min(top_k, len(self.chunks)))
        return [
            {**self.chunks[int(chunk_index)], "score": float(score)}
            for score, chunk_index in zip(scores[0], indices[0])
        ]


_DEFAULT_RETRIEVER: FAISSRetriever | None = None
_DEFAULT_CACHE_KEY: tuple[str, str] | None = None


def retrieve(
    question: str,
    top_k: int = 3,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    model_name: str = DEFAULT_MODEL,
) -> list[dict[str, Any]]:
    """Convenience function for the backend; builds the default index once per process."""
    global _DEFAULT_RETRIEVER, _DEFAULT_CACHE_KEY
    cache_key = (str(Path(data_dir).resolve()), model_name)
    if _DEFAULT_RETRIEVER is None or _DEFAULT_CACHE_KEY != cache_key:
        retriever = FAISSRetriever(model_name=model_name)
        retriever.build_index(load_chunks(data_dir))
        _DEFAULT_RETRIEVER = retriever
        _DEFAULT_CACHE_KEY = cache_key
    return _DEFAULT_RETRIEVER.search(question, top_k=top_k)

