"""
core/forge.py
Training loop that feeds tokenised sequences through a Lattice model,
computes cross-entropy loss, and calls Apex to update weights.

Internal names:
  corpus         — list of token-ID sequences used for training
  chronicle      — list of loss values recorded during training
  anneal()       — execute training for N steps
  _ignite()      — single forward + backward pass (returns loss only)
  _ignite_full() — same but also returns final-position logits for live graphs
"""

import random as _random
import time   as _time

from .lattice import Lattice
from .apex    import Apex
from .flux    import Flux


class Forge:
    """
    Manages the training lifecycle for a Lattice model.

    Parameters
    ----------
    model     : Lattice
    optimizer : Apex
    corpus    : list[list[int]] — pre-built training sequences
    """

    def __init__(self, model: Lattice, optimizer: Apex, corpus: list[list[int]]):
        self.model     = model
        self.optimizer = optimizer
        self.corpus    = corpus
        self.chronicle: list[float] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def anneal(
        self,
        steps:       int,
        on_step_cb=None,    # lightweight terminal progress callback
                            # signature: (step, total, loss_float, eta_seconds)
        on_live_cb=None,    # richer live-graph callback for LiveWire
                            # signature: (step, total, loss_float, lr_float,
                            #             last_logits, params)
    ) -> list[float]:
        """
        Train for `steps` gradient steps.

        on_step_cb — terminal progress bar callback
        on_live_cb — live graph callback; receives raw logits + param list
                     so LiveWire can refresh all 5 windows each step

        Returns the full loss chronicle (one float per step).
        """
        _random.shuffle(self.corpus)
        t0 = _time.time()

        for step in range(steps):
            seq = self.corpus[step % len(self.corpus)]

            if on_live_cb is not None:
                loss, last_logits = self._ignite_full(seq)
            else:
                loss        = self._ignite(seq)
                last_logits = None

            decay      = 1.0 - step / steps
            current_lr = self.optimizer.pulse * decay
            self.optimizer.step(decay)

            self.chronicle.append(loss)

            if on_step_cb:
                elapsed = _time.time() - t0
                eta     = elapsed / (step + 1) * (steps - step - 1)
                on_step_cb(step, steps, loss, eta)

            if on_live_cb:
                on_live_cb(
                    step, steps, loss, current_lr,
                    last_logits, self.optimizer.params,
                )

        return self.chronicle

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _ignite(self, seq: list[int]) -> float:
        """Forward → loss → backward → return scalar loss value."""
        model   = self.model
        L       = min(model.horizon, len(seq) - 1)
        k_shelf = [[] for _ in range(model.rifts)]
        v_shelf = [[] for _ in range(model.rifts)]
        pieces: list[Flux] = []

        for pos in range(L):
            logits = model.emit(seq[pos], pos, k_shelf, v_shelf)
            probs  = model._scatter(logits)
            pieces.append(-probs[seq[pos + 1]].loge())

        loss = (1.0 / L) * sum(pieces)
        loss.diffuse()
        return loss.val

    def _ignite_full(self, seq: list[int]) -> tuple[float, list]:
        """
        Same as _ignite() but also returns the raw logits from the final
        position so LiveWire can show the token probability heatmap.

        Returns (loss_float, last_logits_list)
        """
        model   = self.model
        L       = min(model.horizon, len(seq) - 1)
        k_shelf = [[] for _ in range(model.rifts)]
        v_shelf = [[] for _ in range(model.rifts)]
        pieces: list[Flux] = []
        last_logits: list  = []

        for pos in range(L):
            logits = model.emit(seq[pos], pos, k_shelf, v_shelf)
            probs  = model._scatter(logits)
            pieces.append(-probs[seq[pos + 1]].loge())
            last_logits = logits        # keep the last position's logits

        loss = (1.0 / L) * sum(pieces)
        loss.diffuse()
        return loss.val, last_logits

    def infer(
        self,
        context:     list[int],
        n_steps:     int,
        temperature: float = 0.5,
        stop_token:  int | None = None,
    ) -> list[int]:
        """
        Auto-regressive inference: feed `context`, predict `n_steps` tokens.
        Returns predicted token IDs (not including the seed context).
        """
        import random as _r

        model   = self.model
        k_shelf = [[] for _ in range(model.rifts)]
        v_shelf = [[] for _ in range(model.rifts)]
        output  = []

        for pos, tok in enumerate(context):
            last_logits = model.emit(tok, pos, k_shelf, v_shelf)

        cur_tok = context[-1]
        pos     = len(context)

        for _ in range(n_steps):
            if pos >= model.horizon:
                break
            logits   = model.emit(cur_tok, pos, k_shelf, v_shelf)
            tempered = [Flux(z.val / max(temperature, 1e-6)) for z in logits]
            probs    = model._scatter(tempered)
            weights  = [p.val for p in probs]
            cur_tok  = _r.choices(range(model.vocab_sz), weights=weights)[0]
            if stop_token is not None and cur_tok == stop_token:
                break
            output.append(cur_tok)
            pos += 1

        return output
