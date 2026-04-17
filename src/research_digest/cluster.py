"""Agglomerative clustering by cosine distance (numpy-only).

Used to de-duplicate near-identical papers in the daily digest. For each
cluster of size >= 2 we keep the highest-scoring paper as the "head" and
annotate it with the count of related papers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rank import ScoredPaper


@dataclass
class ClusteredPaper:
    scored: ScoredPaper
    related_count: int  # number of additional papers merged into this head
    related_ids: list[str]


def _cosine_distance_matrix(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    unit = mat / norms
    sim = unit @ unit.T
    # Clamp for numerical drift.
    sim = np.clip(sim, -1.0, 1.0)
    return 1.0 - sim


def _agglomerate(dist: np.ndarray, threshold: float) -> list[list[int]]:
    """Average-linkage agglomerative clustering.

    Merges the two closest clusters while the linkage distance is below
    `threshold`. Linkage distance between two clusters is the average of
    pairwise distances across their members.
    """
    n = dist.shape[0]
    if n == 0:
        return []
    clusters: list[list[int]] = [[i] for i in range(n)]

    while len(clusters) > 1:
        best = (float("inf"), -1, -1)
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                ci, cj = clusters[i], clusters[j]
                # Average linkage.
                sub = dist[np.ix_(ci, cj)]
                d = float(sub.mean())
                if d < best[0]:
                    best = (d, i, j)
        d, i, j = best
        if d >= threshold:
            break
        merged = clusters[i] + clusters[j]
        # Remove higher index first to keep the lower one valid.
        del clusters[j]
        del clusters[i]
        clusters.append(merged)

    return clusters


def cluster_and_dedup(
    scored: list[ScoredPaper],
    threshold: float = 0.25,
) -> list[ClusteredPaper]:
    """Return the cluster heads, preserving the original score ordering.

    `threshold` is a cosine *distance* (1 - cosine similarity).
    """
    if not scored:
        return []

    mat = np.stack([s.vector for s in scored], axis=0)
    dist = _cosine_distance_matrix(mat)
    groups = _agglomerate(dist, threshold=threshold)

    heads: list[ClusteredPaper] = []
    for group in groups:
        members = [scored[i] for i in group]
        members.sort(key=lambda s: s.score, reverse=True)
        head = members[0]
        related = [m.paper.arxiv_id for m in members[1:]]
        heads.append(
            ClusteredPaper(
                scored=head,
                related_count=len(related),
                related_ids=related,
            )
        )

    heads.sort(key=lambda c: c.scored.score, reverse=True)
    return heads
