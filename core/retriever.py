"""
core/retriever.py
TF-IDF retrieval engine.  Grounds every chat response in actual training data.

Pipeline
--------
  fit(docs)         — chunk + index all training documents
  search(query, k)  — return top-k chunks with cosine-similarity scores
  answer(query, …)  — retrieve + generate; returns None if out of scope

Out-of-scope detection
-----------------------
  If the best cosine similarity score is below SCOPE_THRESHOLD, the question
  is considered outside the training data's topic space and the engine returns
  None.  The caller shows a polite "not in training data" message.
"""

from __future__ import annotations
import math
import re
from collections import Counter


SCOPE_THRESHOLD = 0.07    # minimum similarity to be considered "in scope"
CHUNK_MIN_LEN   = 8       # discard chunks shorter than this (characters)


class Retriever:
    """
    Lightweight TF-IDF retrieval over a corpus of training document strings.

    Attributes
    ----------
    chunks  : list[str]         — all indexed text chunks
    vocab   : dict[str, int]    — word → index
    idf     : dict[str, float]  — word → inverse document frequency
    matrix  : list[dict]        — TF-IDF sparse vectors for each chunk
    """

    def __init__(self, threshold: float = SCOPE_THRESHOLD):
        self.threshold = threshold
        self.chunks:  list[str]         = []
        self.vocab:   dict[str, int]    = {}
        self.idf:     dict[str, float]  = {}
        self.matrix:  list[dict]        = []

    # ── Indexing ──────────────────────────────────────────────────────────────

    def fit(self, docs: list[str]) -> "Retriever":
        """
        Chunk all documents, build vocabulary, compute TF-IDF vectors.
        Call once after loading the training corpus.
        """
        self.chunks = self._chunk(docs)

        word_bags: list[Counter] = [
            Counter(self._tok(c)) for c in self.chunks
        ]

        # Vocabulary from all words seen at least once
        all_words = sorted({w for bag in word_bags for w in bag})
        self.vocab = {w: i for i, w in enumerate(all_words)}

        # IDF (smoothed)
        n = len(self.chunks)
        doc_freq: Counter = Counter()
        for bag in word_bags:
            for w in bag:
                if w in self.vocab:
                    doc_freq[w] += 1

        self.idf = {
            w: math.log((n + 1) / (df + 1)) + 1.0
            for w, df in doc_freq.items()
        }

        # TF-IDF sparse vectors
        self.matrix = []
        for bag in word_bags:
            total = sum(bag.values()) or 1
            vec   = {
                w: (cnt / total) * self.idf.get(w, 1.0)
                for w, cnt in bag.items()
                if w in self.vocab
            }
            self.matrix.append(vec)

        return self

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query: str, k: int = 4) -> list[tuple[float, str]]:
        """
        Return up to k (score, chunk) pairs sorted by descending similarity.
        Scores are cosine similarity in TF-IDF space.
        """
        qvec = self._query_vec(query)
        if not qvec:
            return []

        scores: list[tuple[float, int]] = []
        for idx, dvec in enumerate(self.matrix):
            s = self._cosine(qvec, dvec)
            if s > 0:
                scores.append((s, idx))

        scores.sort(reverse=True)
        return [
            (score, self.chunks[idx])
            for score, idx in scores[:k]
        ]

    def in_scope(self, query: str) -> bool:
        """True if the query's best match exceeds the threshold."""
        hits = self.search(query, k=1)
        return bool(hits) and hits[0][0] >= self.threshold

    # ── Answer generation ─────────────────────────────────────────────────────

    def answer(
        self,
        query:       str,
        forge,       # Forge instance (for inference)
        cipher,      # Cipher instance (for encode/decode)
        model,       # Lattice instance
        temperature: float = 0.35,
    ) -> dict | None:
        """
        Retrieve the most relevant training chunk, then let the model
        generate a continuation seeded with that chunk.

        Returns
        -------
        dict  with keys: answer (str), continuation (str), score (float)
        None  if the query is outside the training data scope
        """
        hits = self.search(query, k=3)
        if not hits or hits[0][0] < self.threshold:
            return None

        best_score, best_chunk = hits[0]

        # Seed the model with the retrieved passage and let it continue
        seed_text  = best_chunk[: model.horizon // 2]
        seed_ids   = [cipher.seal] + cipher.encipher(seed_text)
        if len(seed_ids) < 2:
            seed_ids = [cipher.seal]

        gen_ids = forge.infer(
            context    = seed_ids,
            n_steps    = min(model.horizon // 2, 80),
            temperature = temperature,
            stop_token = cipher.seal,
        )
        continuation = cipher.decipher(gen_ids).strip()

        return {
            "answer":       best_chunk,
            "continuation": continuation,
            "score":        best_score,
            "all_hits":     hits,
        }

    # ── Internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _tok(text: str) -> list[str]:
        """Simple lower-case word tokeniser."""
        return re.findall(r"[a-zA-Z0-9]+", text.lower())

    @staticmethod
    def _chunk(docs: list[str]) -> list[str]:
        """
        Split documents into sentence-level chunks.
        Falls back to the raw documents if splitting yields nothing useful.
        """
        chunks: list[str] = []
        splitter = re.compile(r"(?<=[.!?\n])\s+")
        for doc in docs:
            parts = splitter.split(doc.strip())
            for part in parts:
                part = part.strip()
                if len(part) >= CHUNK_MIN_LEN:
                    chunks.append(part)
        return chunks if chunks else [d for d in docs if len(d) >= CHUNK_MIN_LEN]

    def _query_vec(self, text: str) -> dict[str, float]:
        words  = self._tok(text)
        bag    = Counter(words)
        total  = sum(bag.values()) or 1
        return {
            w: (cnt / total) * self.idf.get(w, 1.0)
            for w, cnt in bag.items()
            if w in self.vocab
        }

    @staticmethod
    def _cosine(v1: dict, v2: dict) -> float:
        dot  = sum(v1.get(w, 0.0) * v2.get(w, 0.0) for w in v1)
        mag1 = math.sqrt(sum(x * x for x in v1.values()))
        mag2 = math.sqrt(sum(x * x for x in v2.values()))
        denom = mag1 * mag2
        return dot / denom if denom > 1e-10 else 0.0
