"""
Internal Embedding RAG System for Healthcare Documents.

This module implements embedding-based retrieval as INTERNAL MEMORY for agents.
It is NOT a tool - it represents agent cognition/memory.

Uses Qdrant for vector storage (store/) and OpenAI text-embedding-3-small for embeddings.
Supports PDF, TXT, and DOCX.
"""

import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from utils.config import get_config, get_api_key

try:
    import pypdf
except ImportError:
    pypdf = None
try:
    import docx
except ImportError:
    docx = None

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}
# OpenAI text-embedding-3-small dimension
EMBEDDING_SIZE = 1536

# Project root (parent of rag_internal/) so store/ is always at repo root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# One Qdrant client per storage path (avoids AlreadyLocked when re-running notebooks)
_qdrant_clients: Dict[str, Any] = {}


def _get_qdrant_client(qdrant_path: Path) -> QdrantClient:
    """Get or create a QdrantClient for the given path. Reuses existing client to avoid lock errors."""
    key = str(qdrant_path.resolve())
    if key not in _qdrant_clients:
        _qdrant_clients[key] = QdrantClient(path=key)
    return _qdrant_clients[key]


class MedicalKnowledgeRetriever:
    """
    Embedding-based retrieval using Qdrant. Loads PDF, TXT, and DOCX from a directory.
    This is internal agent memory, not an external tool.
    """

    def __init__(
        self,
        docs_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
        reset_cache: bool = False,
    ):
        config = get_config()
        self.docs_directory = Path(
            docs_directory or config.get("data.docs_directory", "./data/medical_guides")
        )
        cache_path_cfg = config.get("rag.cache_path", "./store")
        cache_path = Path(cache_path_cfg)
        # Resolve store path relative to project root so it's not created in scratch_demos/
        self.store_path = cache_path if cache_path.is_absolute() else (_PROJECT_ROOT / cache_path).resolve()
        self.collection_name = collection_name or config.get(
            "rag.collection_name", "medical_knowledge"
        )
        self.chunk_size = chunk_size or config.get("rag.chunk_size", 800)
        self.overlap = overlap or config.get("rag.overlap", 100)
        self.embed_model = config.get("rag.embed_model", "text-embedding-3-small")
        if self.embed_model == "text-embedding-3-small":
            self.embed_model = "openai/text-embedding-3-small"
        self.max_k = config.get("rag.max_k", 4)
        self.score_threshold = config.get("rag.score_threshold", 0.18)

        self.openai_client = OpenAI(
            api_key=get_api_key("openrouter"),
            base_url="https://openrouter.ai/api/v1"
        )
        self.store_path.mkdir(parents=True, exist_ok=True)
        qdrant_path = self.store_path / "qdrant"
        qdrant_path.mkdir(parents=True, exist_ok=True)
        self._client = _get_qdrant_client(qdrant_path)
        self._ensure_collection(reset_cache=reset_cache)
        self._is_indexed = self._client.count(self.collection_name).count > 0

    def _ensure_collection(self, reset_cache: bool = False) -> None:
        if reset_cache and self._client.collection_exists(self.collection_name):
            self._client.delete_collection(self.collection_name)
        if not self._client.collection_exists(self.collection_name):
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=EMBEDDING_SIZE,
                    distance=Distance.COSINE,
                ),
            )

    def _read_txt(self, file_path: Path) -> str:
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read().replace("\x00", "").strip()
        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")
            return ""

    def _read_pdf(self, file_path: Path) -> str:
        if pypdf is None:
            print(f"PDF support not available (install pypdf). Skipping {file_path.name}")
            return ""
        try:
            reader = pypdf.PdfReader(str(file_path))
            parts = []
            for i, page in enumerate(reader.pages):
                t = page.extract_text()
                if t:
                    parts.append(f"\n--- Page {i + 1} ---\n{t}")
            return "".join(parts) if parts else ""
        except Exception as e:
            print(f"Error reading PDF {file_path.name}: {e}")
            return ""

    def _read_docx(self, file_path: Path) -> str:
        if docx is None:
            print(f"DOCX support not available (install python-docx). Skipping {file_path.name}")
            return ""
        try:
            doc = docx.Document(str(file_path))
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except Exception as e:
            print(f"Error reading DOCX {file_path.name}: {e}")
            return ""

    def _extract_text_from_file(self, file_path: Path) -> str:
        suffix = file_path.suffix.lower()
        if suffix == ".txt":
            return self._read_txt(file_path)
        if suffix == ".pdf":
            return self._read_pdf(file_path)
        if suffix == ".docx":
            return self._read_docx(file_path)
        print(f"Unsupported format {suffix} for {file_path.name}")
        return ""

    def _chunk_text(self, text: str, source: str) -> List[Dict[str, Any]]:
        chunks = []
        text = text.strip()
        if not text:
            return chunks
        start = 0
        chunk_id = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]

            # Try to break at sentence boundary if not at end
            if end < len(text):
                search_start = max(0, len(chunk_text) - 150)
                for delimiter in [". ", ".\n", "? ", "?\n", "! ", "!\n"]:
                    last_pos = chunk_text.rfind(delimiter, search_start)
                    if last_pos > 0:
                        chunk_text = chunk_text[: last_pos + len(delimiter)]
                        end = start + last_pos + len(delimiter)
                        break

            chunks.append(
                {"text": chunk_text.strip(), "source": source, "chunk_id": chunk_id}
            )

            # If we reached the end of the text, stop the loop
            if end >= len(text):
                break

            start = end - self.overlap

            # Safety check: ensure start is actually moving forward
            if start < 0:
                start = 0

            chunk_id += 1

        return chunks

    def _get_embedding(self, text: str) -> List[float]:
        response = self.openai_client.embeddings.create(
            model=self.embed_model, input=text
        )
        return response.data[0].embedding

    def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        safe = []
        for t in texts:
            if t is None or not isinstance(t, str):
                safe.append(" ")
            else:
                s = (t.strip() or " ").replace("\x00", "")
                safe.append(s[:30000] if len(s) > 30000 else s)
        response = self.openai_client.embeddings.create(
            model=self.embed_model, input=safe
        )
        return [[float(x) for x in item.embedding] for item in response.data]

    def _collect_document_paths(self) -> List[Path]:
        paths = []
        for ext in SUPPORTED_EXTENSIONS:
            paths.extend(self.docs_directory.glob(f"*{ext}"))
        return sorted(paths)

    def index_documents(self, force_reindex: bool = False) -> Dict[str, Any]:
        if self._is_indexed and not force_reindex:
            return {
                "status": "already_indexed",
                "document_count": self._client.count(self.collection_name).count,
            }
        if force_reindex:
            self._ensure_collection(reset_cache=True)
            self._is_indexed = False

        start_time = time.time()
        doc_paths = self._collect_document_paths()
        if not doc_paths:
            return {"status": "no_documents", "document_count": 0}

        print(f"Found {len(doc_paths)} document(s) to index (.txt, .pdf, .docx)...")
        all_chunks = []
        for path in doc_paths:
            print(f"Processing {path.name}...")
            text = self._extract_text_from_file(path)
            if text:
                all_chunks.extend(self._chunk_text(text, path.stem))

        all_chunks = [c for c in all_chunks if (c.get("text") or "").strip()]
        if not all_chunks:
            return {"status": "no_content", "document_count": 0}

        n_chunks = len(all_chunks)
        print(f"Created {n_chunks} chunks. Embedding and saving to Qdrant (store/)...")

        batch_size = 20
        for i in range(0, n_chunks, batch_size):
            batch = all_chunks[i : i + batch_size]
            texts = [
                str((c.get("text") or "").strip() or " ").replace("\x00", "")
                for c in batch
            ]
            embeddings = self._get_embeddings_batch(texts)
            points = [
                PointStruct(
                    id=i + j,
                    vector=embeddings[j],
                    payload={
                        "text": texts[j],
                        "source": str(c["source"]),
                        "chunk_id": int(c["chunk_id"]),
                    },
                )
                for j, c in enumerate(batch)
            ]
            self._client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            print(f"  {min(i + batch_size, n_chunks)}/{n_chunks} chunks")

        self._is_indexed = True
        elapsed = time.time() - start_time
        print(f"Done in {elapsed:.1f}s.")
        return {
            "status": "indexed",
            "document_count": len(doc_paths),
            "chunk_count": n_chunks,
            "elapsed_seconds": round(elapsed, 2),
        }

    def retrieve(
        self,
        query: str,
        max_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        if not self._is_indexed:
            return []
        k = max_k or self.max_k
        threshold = score_threshold or self.score_threshold
        query_embedding = self._get_embedding(query)
        response = self._client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=k,
        )
        if response is None:
            hits = []
        elif hasattr(response, "points"):
            hits = response.points
        elif isinstance(response, list):
            hits = response
        else:
            hits = list(response) if hasattr(response, "__iter__") else []
        # Qdrant COSINE: score is similarity (higher = better)
        out = []
        for h in hits:
            sim = float(h.score) if h.score is not None else 0.0
            if sim >= threshold and h.payload:
                out.append({
                    "text": h.payload.get("text", ""),
                    "source": h.payload.get("source", "unknown"),
                    "chunk_id": h.payload.get("chunk_id"),
                    "similarity": round(sim, 4),
                })
        return out

    def get_context(
        self,
        query: str,
        max_k: Optional[int] = None,
        include_metadata: bool = True,
    ) -> str:
        chunks = self.retrieve(query, max_k=max_k)
        if not chunks:
            return "No relevant information found in internal knowledge base."
        parts = [
            f"[Source {idx}: {chunk['source']}]\n{chunk['text']}"
            if include_metadata
            else chunk["text"]
            for idx, chunk in enumerate(chunks, 1)
        ]
        return "\n\n---\n\n".join(parts)

    def get_stats(self) -> Dict[str, Any]:
        count = self._client.count(self.collection_name).count
        return {
            "indexed": self._is_indexed,
            "total_chunks": count,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "embed_model": self.embed_model,
            "collection_name": self.collection_name,
        }


_retriever_instance: Optional[MedicalKnowledgeRetriever] = None


def get_retriever(reset_cache: bool = False) -> MedicalKnowledgeRetriever:
    global _retriever_instance
    if _retriever_instance is None or reset_cache:
        _retriever_instance = MedicalKnowledgeRetriever(reset_cache=reset_cache)
    return _retriever_instance
