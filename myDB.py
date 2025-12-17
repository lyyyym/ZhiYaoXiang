import os
import json
import uuid
from typing import List, Dict, Optional, Tuple, Callable

import numpy as np


class FlatIndex:
    def __init__(self, dim: int, metric: str = "cosine"):
        self.dim = dim
        self.metric = metric
        self.vectors = np.empty((0, dim), dtype=np.float32)
        self.vectors_norm = np.empty((0, dim), dtype=np.float32)

    def _refresh_norm(self):
        if self.metric == "cosine":
            if self.vectors.shape[0] == 0:
                self.vectors_norm = np.empty((0, self.dim), dtype=np.float32)
            else:
                norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                self.vectors_norm = self.vectors / norms

    def add(self, vecs: np.ndarray):
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        if vecs.shape[1] != self.dim:
            raise ValueError("dim mismatch")
        vecs = vecs.astype(np.float32)
        if self.vectors.size == 0:
            self.vectors = vecs
        else:
            self.vectors = np.vstack([self.vectors, vecs])
        self._refresh_norm()

    def remove(self, idxs: List[int]):
        if not idxs:
            return
        mask = np.ones((self.vectors.shape[0],), dtype=bool)
        mask[np.asarray(idxs, dtype=int)] = False
        self.vectors = self.vectors[mask]
        self._refresh_norm()

    def search(self, query: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        if query.ndim == 1:
            q = query.reshape(1, -1).astype(np.float32)
        else:
            q = query.astype(np.float32)
        if q.shape[1] != self.dim:
            raise ValueError("dim mismatch")
        if self.vectors.shape[0] == 0:
            return np.array([], dtype=int), np.array([], dtype=np.float32)
        if self.metric == "cosine":
            qn = q / max(np.linalg.norm(q), 1e-12)
            scores = (self.vectors_norm @ qn.T).squeeze(-1)
            k = min(top_k, scores.shape[0])
            idxs = np.argpartition(-scores, k - 1)[:k]
            order = np.argsort(-scores[idxs])
            return idxs[order], scores[idxs][order]
        else:
            diffs = self.vectors - q
            dists = np.linalg.norm(diffs, axis=1)
            k = min(top_k, dists.shape[0])
            idxs = np.argpartition(dists, k - 1)[:k]
            order = np.argsort(dists[idxs])
            return idxs[order], dists[idxs][order]


class SimpleVectorDB:
    def __init__(self, dim: int, storage_dir: str, metric: str = "cosine"):
        self.dim = dim
        self.storage_dir = storage_dir
        self.metric = metric
        os.makedirs(storage_dir, exist_ok=True)
        self.index = FlatIndex(dim, metric)
        self.ids: List[str] = []
        self.payloads: List[Dict] = []

    def count(self) -> int:
        return len(self.ids)

    def upsert(
        self,
        embeddings: np.ndarray,
        payloads: List[Dict],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        if embeddings.shape[1] != self.dim:
            raise ValueError("dim mismatch")
        n = embeddings.shape[0]
        if ids is None:
            new_ids = [str(uuid.uuid4()) for _ in range(n)]
        else:
            new_ids = ids
            if len(new_ids) != n:
                raise ValueError("ids length mismatch")
        if len(payloads) != n:
            raise ValueError("payloads length mismatch")
        self.index.add(embeddings)
        self.ids.extend(new_ids)
        self.payloads.extend(payloads)
        return new_ids

    def delete(self, ids: List[str]):
        if not ids:
            return
        idset = set(ids)
        to_remove = [i for i, x in enumerate(self.ids) if x in idset]
        self.index.remove(to_remove)
        mask = np.ones((len(self.ids),), dtype=bool)
        if to_remove:
            mask[np.asarray(to_remove, dtype=int)] = False
        self.ids = [x for i, x in enumerate(self.ids) if mask[i]]
        self.payloads = [x for i, x in enumerate(self.payloads) if mask[i]]

    def search(self, query: np.ndarray, top_k: int = 5):
        idxs, scores = self.index.search(query, top_k)
        results = []
        for i, s in zip(idxs.tolist(), scores.tolist()):
            results.append({"id": self.ids[i], "score": float(s), "payload": self.payloads[i]})
        return results

    def add_texts(
        self,
        texts: List[str],
        embedder: Callable[[str], np.ndarray],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        embs = np.asarray([embedder(t) for t in texts], dtype=np.float32)
        payloads = metadatas if metadatas is not None else [{} for _ in texts]
        return self.upsert(embs, payloads, ids)

    def search_text(self, text: str, embedder: Callable[[str], np.ndarray], top_k: int = 5):
        q = np.asarray(embedder(text), dtype=np.float32)
        return self.search(q, top_k)

    def save(self):
        vec_path = os.path.join(self.storage_dir, "vectors.npy")
        meta_path = os.path.join(self.storage_dir, "meta.json")
        np.save(vec_path, self.index.vectors)
        meta = {"dim": self.dim, "metric": self.metric, "ids": self.ids, "payloads": self.payloads, "model": os.environ.get("EMBED_MODEL_ID", "moka-ai/m3e-base")}
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)

    @classmethod
    def load(cls, storage_dir: str):
        meta_path = os.path.join(storage_dir, "meta.json")
        vec_path = os.path.join(storage_dir, "vectors.npy")
        if not os.path.exists(meta_path) or not os.path.exists(vec_path):
            raise FileNotFoundError("missing files")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        vectors = np.load(vec_path).astype(np.float32)
        db = cls(meta["dim"], storage_dir, meta.get("metric", "cosine"))
        db.index.vectors = vectors
        db.index._refresh_norm()
        db.ids = meta["ids"]
        db.payloads = meta["payloads"]
        return db