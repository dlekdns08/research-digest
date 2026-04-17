"""Embeddings for papers.

Primary backend: Voyage AI (`voyage-3`) via httpx.
Fallback: a tiny local TF-IDF (hashed bag-of-words) when `VOYAGE_API_KEY`
is unset or the call fails.

Vectors are cached in the feedback sqlite DB's `embeddings` table.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

import httpx
import numpy as np

from . import feedback
from .config import Settings

_VOYAGE_URL = "https://api.voyageai.com/v1/embeddings"
_TFIDF_DIM = 512
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]+")


@dataclass(frozen=True)
class PaperText:
    arxiv_id: str
    title: str
    abstract: str

    def text(self) -> str:
        return f"{self.title}\n\n{self.abstract}".strip()


# ---------- numpy <-> bytes ----------


def _vec_to_bytes(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def _bytes_to_vec(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32).copy()


# ---------- fallback: hashed TF-IDF ----------


def _hash_bucket(token: str, dim: int) -> int:
    h = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") % dim


def _tfidf_vector(text: str, dim: int = _TFIDF_DIM) -> np.ndarray:
    """A deterministic hashed TF with sublinear scaling — no external deps.

    This is intentionally minimal: good enough for cosine similarity on
    title+abstract-sized texts when no embedding API is available.
    """
    vec = np.zeros(dim, dtype=np.float32)
    tokens = [t.lower() for t in _WORD_RE.findall(text or "")]
    if not tokens:
        return vec
    counts: dict[int, int] = {}
    for tok in tokens:
        if len(tok) < 3:
            continue
        idx = _hash_bucket(tok, dim)
        counts[idx] = counts.get(idx, 0) + 1
    for idx, c in counts.items():
        vec[idx] = 1.0 + math.log(c)
    n = float(np.linalg.norm(vec))
    if n > 0:
        vec /= n
    return vec


# ---------- Voyage AI ----------


def _voyage_embed(
    texts: list[str],
    api_key: str,
    model: str,
    timeout: float = 30.0,
) -> list[np.ndarray]:
    payload = {"input": texts, "model": model, "input_type": "document"}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = httpx.post(_VOYAGE_URL, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    out: list[np.ndarray] = []
    for item in data.get("data", []):
        out.append(np.asarray(item["embedding"], dtype=np.float32))
    return out


# ---------- public API ----------


def embed_papers(
    papers: list[PaperText],
    settings: Settings,
    use_cache: bool = True,
) -> dict[str, np.ndarray]:
    """Return {arxiv_id: vector} for each input paper.

    - Looks up cached embeddings first.
    - For misses, calls Voyage if a key is set, else the TF-IDF fallback.
    - Writes new vectors back to the cache.
    """
    if not papers:
        return {}

    out: dict[str, np.ndarray] = {}
    missing: list[PaperText] = []

    if use_cache:
        cached = feedback.get_embeddings([p.arxiv_id for p in papers])
        for p in papers:
            blob = cached.get(p.arxiv_id)
            if blob is not None:
                out[p.arxiv_id] = _bytes_to_vec(blob)
            else:
                missing.append(p)
    else:
        missing = list(papers)

    if not missing:
        return out

    new_vecs: list[np.ndarray] = []
    used_voyage = False
    if settings.voyage_api_key:
        try:
            new_vecs = _voyage_embed(
                [p.text() for p in missing],
                api_key=settings.voyage_api_key,
                model=settings.voyage_model,
            )
            used_voyage = len(new_vecs) == len(missing)
        except Exception:  # noqa: BLE001 - any failure falls back locally
            new_vecs = []

    if not used_voyage:
        new_vecs = [_tfidf_vector(p.text()) for p in missing]

    for p, v in zip(missing, new_vecs, strict=True):
        out[p.arxiv_id] = v
        if use_cache:
            feedback.put_embedding(p.arxiv_id, _vec_to_bytes(v))

    return out


def embed_text(text: str, settings: Settings) -> np.ndarray:
    """One-off embedding for an arbitrary string (no cache)."""
    if settings.voyage_api_key:
        try:
            return _voyage_embed(
                [text], api_key=settings.voyage_api_key, model=settings.voyage_model
            )[0]
        except Exception:  # noqa: BLE001
            pass
    return _tfidf_vector(text)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def centroid(vectors: list[np.ndarray]) -> np.ndarray | None:
    if not vectors:
        return None
    m = np.mean(np.stack(vectors, axis=0), axis=0)
    n = float(np.linalg.norm(m))
    if n > 0:
        m = m / n
    return m.astype(np.float32)
