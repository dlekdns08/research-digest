"""Personalized ranking combining arxiv-graph importance with user history."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import embed, feedback
from .config import Settings
from .db import PaperRow


@dataclass
class ScoredPaper:
    paper: PaperRow
    score: float
    importance_norm: float
    personalization: float
    vector: np.ndarray


def _normalize_importance(papers: list[PaperRow]) -> dict[str, float]:
    if not papers:
        return {}
    scores = [float(p.importance_score) for p in papers]
    lo, hi = min(scores), max(scores)
    span = hi - lo
    if span <= 0:
        return {p.arxiv_id: 0.5 for p in papers}
    return {p.arxiv_id: (float(p.importance_score) - lo) / span for p in papers}


def build_centroid(
    liked_ids: list[str],
    settings: Settings,
) -> np.ndarray | None:
    """Build the centroid of cached liked-paper embeddings.

    Only already-cached embeddings are used: we don't fetch abstracts for
    historical liked papers here, since they may no longer be in the
    candidate pool. If nothing is cached, returns None.
    """
    if not liked_ids:
        return None
    cached = feedback.get_embeddings(liked_ids)
    vecs = [embed._bytes_to_vec(b) for b in cached.values()]
    return embed.centroid(vecs)


def rank_personalized(
    papers: list[PaperRow],
    settings: Settings,
    personalize: bool = True,
    top_k: int | None = None,
) -> list[ScoredPaper]:
    """Rank a candidate pool with optional personalization.

    Final score = 0.6 * cosine(sim, liked_centroid) + 0.4 * normalized_importance.
    If personalization is off or there is no liked centroid, falls back to
    importance alone.
    """
    if not papers:
        return []

    # Always embed the candidates — needed for clustering downstream.
    texts = [
        embed.PaperText(arxiv_id=p.arxiv_id, title=p.title, abstract=p.abstract)
        for p in papers
    ]
    vectors = embed.embed_papers(texts, settings=settings)
    importance = _normalize_importance(papers)

    centroid_vec: np.ndarray | None = None
    if personalize:
        liked = feedback.recent_liked(limit=20)
        centroid_vec = build_centroid(liked, settings=settings)

    scored: list[ScoredPaper] = []
    for p in papers:
        v = vectors.get(p.arxiv_id)
        if v is None:
            continue
        imp = importance.get(p.arxiv_id, 0.0)
        if centroid_vec is not None:
            sim = embed.cosine(v, centroid_vec)
            # cosine may be negative; squash to [0,1] for blending.
            sim01 = (sim + 1.0) / 2.0
            score = 0.6 * sim01 + 0.4 * imp
            scored.append(
                ScoredPaper(
                    paper=p,
                    score=score,
                    importance_norm=imp,
                    personalization=sim,
                    vector=v,
                )
            )
        else:
            scored.append(
                ScoredPaper(
                    paper=p,
                    score=imp,
                    importance_norm=imp,
                    personalization=0.0,
                    vector=v,
                )
            )

    scored.sort(key=lambda s: s.score, reverse=True)
    if top_k is not None:
        scored = scored[:top_k]
    return scored
