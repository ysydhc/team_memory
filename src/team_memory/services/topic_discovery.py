"""Topic Discovery — lightweight clustering of experiences for high-level retrieval.

Uses existing experience embeddings from PG (no additional LLM calls).
K-means clustering groups semantically similar experiences into topics.
Each topic gets a name derived from its most frequent tags/entities.

Used by:
- WikiCompiler: generates wiki/topics/ pages
- SearchPipeline: topic-level retrieval as supplementary signal
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from sqlalchemy import text

from team_memory.storage.database import get_session


@dataclass
class Topic:
    """A discovered topic cluster."""

    id: str
    name: str
    experience_ids: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    center: list[float] = field(default_factory=list)  # cluster centroid


@dataclass
class TopicResult:
    """Result of topic discovery."""

    topics: list[Topic] = field(default_factory=list)
    total_experiences: int = 0
    unclustered: int = 0


def _simple_kmeans(
    vectors: dict[str, list[float]],
    k: int,
    max_iter: int = 50,
    seed: int = 42,
) -> dict[int, list[str]]:
    """Simple K-means clustering using numpy.

    Args:
        vectors: {id: embedding_vector}
        k: number of clusters
        max_iter: max iterations
        seed: random seed for reproducibility

    Returns:
        {cluster_id: [id1, id2, ...]}
    """
    import numpy as np

    if not vectors or k <= 0:
        return {}

    ids = list(vectors.keys())
    mats = np.array([vectors[vid] for vid in ids], dtype=np.float32)

    # Normalize
    norms = np.linalg.norm(mats, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    mats = mats / norms

    n = len(ids)
    if k >= n:
        return {i: [ids[i]] for i in range(n)}

    rng = np.random.RandomState(seed)
    # K-means++ initialization
    centers = np.zeros((k, mats.shape[1]), dtype=np.float32)
    centers[0] = mats[rng.randint(n)]
    for ci in range(1, k):
        dists = np.min(
            np.clip(1 - np.dot(mats, centers[:ci].T), 0, None), axis=1
        )
        total = dists.sum()
        if total < 1e-10:
            # Fallback: uniform distribution
            probs = np.ones(n) / n
        else:
            probs = dists / total
        centers[ci] = mats[rng.choice(n, p=probs)]

    # Iterate
    assignments = np.zeros(n, dtype=np.int32)
    for _ in range(max_iter):
        # Assign
        sims = np.dot(mats, centers.T)  # (n, k)
        new_assignments = np.argmax(sims, axis=1)
        if np.array_equal(new_assignments, assignments):
            break
        assignments = new_assignments

        # Update centers
        for ci in range(k):
            mask = assignments == ci
            if mask.sum() > 0:
                centers[ci] = mats[mask].mean(axis=0)
                centers[ci] = centers[ci] / (np.linalg.norm(centers[ci]) + 1e-8)

    # Group by cluster
    clusters: dict[int, list[str]] = {}
    for i, ci in enumerate(assignments):
        ci = int(ci)
        if ci not in clusters:
            clusters[ci] = []
        clusters[ci].append(ids[i])

    return clusters


async def discover_topics(
    db_url: str,
    k: int = 0,
    min_cluster_size: int = 2,
    max_topics: int = 20,
) -> TopicResult:
    """Discover topics from experience embeddings.

    Args:
        db_url: PostgreSQL connection string
        k: Number of clusters (0 = auto via sqrt(N)/2)
        min_cluster_size: Minimum experiences per topic
        max_topics: Maximum number of topics

    Returns:
        TopicResult with discovered topics.
    """
    # 1. Load embeddings
    embeddings: dict[str, list[float]] = {}
    tags_map: dict[str, list[str]] = {}
    entity_map: dict[str, list[str]] = {}

    try:
        import numpy as np
    except ImportError:
        return TopicResult()

    async with get_session(db_url) as session:
        # Load experience embeddings
        result = await session.execute(text("""
            SELECT id, embedding, tags FROM experiences
            WHERE embedding IS NOT NULL AND exp_status = 'published'
        """))
        import json
        for row in result.fetchall():
            exp_id = str(row[0])
            emb = row[1]
            t = row[2]
            if emb is not None:
                try:
                    if isinstance(emb, str):
                        vec = json.loads(emb)
                    elif isinstance(emb, list):
                        vec = emb
                    elif hasattr(emb, "tolist"):
                        vec = emb.tolist()
                    else:
                        continue
                    embeddings[exp_id] = vec
                    tags_map[exp_id] = list(t) if t else []
                except Exception:
                    continue

        # Load entity associations
        result2 = await session.execute(text("""
            SELECT ee.experience_id, e.name
            FROM experience_entities ee
            JOIN entities e ON e.id = ee.entity_id
        """))
        for row in result2.fetchall():
            exp_id = str(row[0])
            entity_name = row[1]
            if exp_id not in entity_map:
                entity_map[exp_id] = []
            entity_map[exp_id].append(entity_name)

    if not embeddings:
        return TopicResult()

    # 2. Auto-determine k
    n = len(embeddings)
    if k <= 0:
        k = max(2, min(int(n**0.5) // 2, max_topics))
    k = min(k, n, max_topics)

    # 3. Cluster
    clusters = _simple_kmeans(embeddings, k)

    # 4. Build topics
    topics: list[Topic] = []
    unclustered = 0
    for ci, exp_ids in clusters.items():
        if len(exp_ids) < min_cluster_size:
            unclustered += len(exp_ids)
            continue

        # Derive topic name from most frequent tags + entities
        all_tags: dict[str, int] = {}
        all_entities: dict[str, int] = {}
        for eid in exp_ids:
            for tag in tags_map.get(eid, []):
                all_tags[tag] = all_tags.get(tag, 0) + 1
            for ent in entity_map.get(eid, []):
                all_entities[ent] = all_entities.get(ent, 0) + 1

        # Top tags and entities
        top_tags = sorted(all_tags, key=all_tags.get, reverse=True)[:5]
        top_entities = sorted(all_entities, key=all_entities.get, reverse=True)[:5]

        # Name: combine top entity + top tag
        name_parts = []
        if top_entities:
            name_parts.append(top_entities[0])
        if top_tags and top_tags[0] not in name_parts:
            name_parts.append(top_tags[0])
        name = " / ".join(name_parts) if name_parts else f"Topic-{ci}"

        # Compute centroid
        vecs = [embeddings[eid] for eid in exp_ids if eid in embeddings]
        if vecs:
            center = np.mean(vecs, axis=0).tolist()
        else:
            center = []

        topics.append(Topic(
            id=str(uuid.uuid4())[:8],
            name=name,
            experience_ids=exp_ids,
            tags=top_tags,
            entities=top_entities,
            center=center,
        ))

    # Sort by size descending
    topics.sort(key=lambda t: len(t.experience_ids), reverse=True)

    return TopicResult(
        topics=topics,
        total_experiences=n,
        unclustered=unclustered,
    )


async def find_by_topic(
    query_embedding: list[float],
    db_url: str,
    top_k: int = 3,
    min_similarity: float = 0.5,
) -> list[Topic]:
    """Find topics most relevant to a query embedding.

    Args:
        query_embedding: Query embedding vector.
        db_url: PostgreSQL connection string.
        top_k: Number of top topics to return.
        min_similarity: Minimum cosine similarity.

    Returns:
        List of Topic objects sorted by relevance.
    """
    try:
        import numpy as np
    except ImportError:
        return []

    result = await discover_topics(db_url)
    if not result.topics:
        return []

    q_vec = np.array(query_embedding, dtype=np.float32)
    q_norm = np.linalg.norm(q_vec)
    if q_norm < 1e-8:
        return []

    scored: list[tuple[float, Topic]] = []
    for topic in result.topics:
        if not topic.center:
            continue
        c_vec = np.array(topic.center, dtype=np.float32)
        sim = float(np.dot(q_vec, c_vec) / (q_norm * np.linalg.norm(c_vec) + 1e-8))
        if sim >= min_similarity:
            scored.append((sim, topic))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored[:top_k]]
