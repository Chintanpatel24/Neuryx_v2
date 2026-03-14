"""
core/lattice.py
Decoder-only causal transformer.  General-purpose; no domain assumptions.

Default config is intentionally large — 4 blocks, depth 64, 8 streams —
for richer pattern learning.  Override via cfg dict.
"""

import random as _random
from .flux import Flux

DEFAULT_CFG = {
    "rifts":   4,    # transformer blocks
    "depth":   64,   # hidden dimension
    "horizon": 128,  # context window (tokens)
    "streams": 8,    # attention heads
}


def _slab(nrow: int, ncol: int, spread: float = 0.04) -> list[list[Flux]]:
    return [
        [Flux(_random.gauss(0.0, spread)) for _ in range(ncol)]
        for _ in range(nrow)
    ]


class Lattice:
    """
    Causal transformer that maps token sequences to next-token logits.

    Parameters
    ----------
    vocab_sz : int
    cfg      : dict  — keys: rifts, depth, horizon, streams
    """

    def __init__(self, vocab_sz: int, cfg: dict | None = None):
        self.vocab_sz = vocab_sz
        cfg = {**DEFAULT_CFG, **(cfg or {})}

        self.rifts   = cfg["rifts"]
        self.depth   = cfg["depth"]
        self.horizon = cfg["horizon"]
        self.streams = cfg["streams"]
        self.channel = self.depth // self.streams

        # Weight registry
        self.manifold: dict[str, list[list[Flux]]] = {
            "tok_slab": _slab(vocab_sz,     self.depth),
            "pos_slab": _slab(self.horizon, self.depth),
            "proj_out": _slab(vocab_sz,     self.depth),
        }
        for i in range(self.rifts):
            p = f"r{i}"
            self.manifold[f"{p}.Wq"] = _slab(self.depth, self.depth)
            self.manifold[f"{p}.Wk"] = _slab(self.depth, self.depth)
            self.manifold[f"{p}.Wv"] = _slab(self.depth, self.depth)
            self.manifold[f"{p}.Wo"] = _slab(self.depth, self.depth)
            self.manifold[f"{p}.Wu"] = _slab(4 * self.depth, self.depth)
            self.manifold[f"{p}.Wd"] = _slab(self.depth, 4 * self.depth)

        self.params: list[Flux] = [
            p for mat in self.manifold.values() for row in mat for p in row
        ]

    # ── primitives ────────────────────────────────────────────────────────────

    @staticmethod
    def _weave(x: list[Flux], W: list[list[Flux]]) -> list[Flux]:
        return [sum(w * xi for w, xi in zip(row, x)) for row in W]

    @staticmethod
    def _scatter(logits: list[Flux]) -> list[Flux]:
        peak = max(z.val for z in logits)
        exps = [(z - peak).expe() for z in logits]
        tot  = sum(exps)
        return [e / tot for e in exps]

    @staticmethod
    def _norm(x: list[Flux]) -> list[Flux]:
        ms    = sum(h * h for h in x) / len(x)
        scale = (ms + 1e-6) ** -0.5
        return [h * scale for h in x]

    # ── forward pass ──────────────────────────────────────────────────────────

    def emit(
        self,
        token_idx: int,
        pos_idx:   int,
        k_shelf:   list[list[list[Flux]]],
        v_shelf:   list[list[list[Flux]]],
    ) -> list[Flux]:
        h = [t + p for t, p in zip(
            self.manifold["tok_slab"][token_idx],
            self.manifold["pos_slab"][pos_idx],
        )]
        h = self._norm(h)

        for i in range(self.rifts):
            p = f"r{i}"
            res = h; h = self._norm(h)

            q = self._weave(h, self.manifold[f"{p}.Wq"])
            k = self._weave(h, self.manifold[f"{p}.Wk"])
            v = self._weave(h, self.manifold[f"{p}.Wv"])
            k_shelf[i].append(k); v_shelf[i].append(v)
            T = len(k_shelf[i])

            heads = []
            for s in range(self.streams):
                lo, hi = s * self.channel, (s + 1) * self.channel
                qh = q[lo:hi]
                kh = [k_shelf[i][t][lo:hi] for t in range(T)]
                vh = [v_shelf[i][t][lo:hi] for t in range(T)]
                scores = [
                    sum(qh[d] * kh[t][d] for d in range(self.channel))
                    / self.channel ** 0.5
                    for t in range(T)
                ]
                aw  = self._scatter(scores)
                ctx = [sum(aw[t] * vh[t][d] for t in range(T))
                       for d in range(self.channel)]
                heads.extend(ctx)

            h = [a + b for a, b in zip(self._weave(heads, self.manifold[f"{p}.Wo"]), res)]
            res = h; h = self._norm(h)
            h = [z.thresh() for z in self._weave(h, self.manifold[f"{p}.Wu"])]
            h = [a + b for a, b in zip(self._weave(h, self.manifold[f"{p}.Wd"]), res)]

        return self._weave(h, self.manifold["proj_out"])
