"""
render/livewire.py
Five separate live matplotlib windows that update in real time during training.

W1  Loss Pulse      — raw + EMA-smoothed cross-entropy loss
W2  Neural Flow     — animated diagram showing data flowing through the network
W3  Token Heatmap   — per-token softmax probability at every step
W4  Gradient Health — mean |∇| magnitude  (log scale)
W5  Live Output     — text samples generated during training

Backend selection
-----------------
The module probes available GUI backends using backend_registry.load_backend_module()
BEFORE calling matplotlib.use(), avoiding the ModuleNotFoundError that occurs when
tkinter/Qt/wx is absent.  Agg (headless) is the guaranteed final fallback.
"""

from __future__ import annotations
import math
import time
import random as _rng

# ══════════════════════════════════════════════════════════════════════════════
#  BACKEND PROBE  (must run before any pyplot import)
# ══════════════════════════════════════════════════════════════════════════════

_MPL_OK = False
_plt    = None
_LSC    = None
_mpl    = None

try:
    import matplotlib as _mpl

    # Probe each GUI backend by attempting to load its module — no side-effects.
    # Only call matplotlib.use() once we have a confirmed-working backend name.
    _chosen_be = "Agg"
    from matplotlib.backends import backend_registry as _br
    for _be in ("TkAgg", "Qt5Agg", "QtAgg", "WXAgg", "MacOSX"):
        try:
            _br.load_backend_module(_be)
            _chosen_be = _be
            break
        except Exception:
            continue

    _mpl.use(_chosen_be)                          # single, safe use() call
    import matplotlib.pyplot as _plt              # import after backend is set
    from matplotlib.colors import LinearSegmentedColormap as _LSC
    _MPL_OK = True

except ImportError:
    pass

# ── Palette ───────────────────────────────────────────────────────────────────
_BG    = "#09091a"
_PAN   = "#0f0f28"
_TEAL  = "#00e5c8"
_VIO   = "#8b5cf6"
_CORAL = "#f87171"
_AMBER = "#fbbf24"
_GREEN = "#34d399"
_PINK  = "#f472b6"
_FG    = "#e2e8f0"
_DIM   = "#1e1e3f"
_BLUE  = "#60a5fa"
_ORG   = "#fb923c"


def _dark(fig, axes: list) -> None:
    fig.patch.set_facecolor(_BG)
    for ax in axes:
        ax.set_facecolor(_PAN)
        ax.tick_params(colors=_FG, labelsize=8)
        ax.xaxis.label.set_color(_FG)
        ax.yaxis.label.set_color(_FG)
        ax.title.set_color(_FG)
        for sp in ax.spines.values():
            sp.set_edgecolor(_DIM)
        ax.grid(True, color=_DIM, alpha=0.3, linewidth=0.4)


def _place(fig, x: int, y: int, w: int, h: int) -> None:
    try:
        mgr = fig.canvas.manager
        win = getattr(mgr, "window", None)
        if win is None:
            return
        if hasattr(win, "wm_geometry"):
            win.wm_geometry(f"{w}x{h}+{x}+{y}")
        elif hasattr(win, "setGeometry"):
            win.setGeometry(x, y, w, h)
        elif hasattr(win, "SetPosition"):
            win.SetPosition((x, y)); win.SetSize((w, h))
    except Exception:
        pass


def _stl(fig) -> None:
    try:
        fig.tight_layout(rect=[0, 0, 1, 0.90])
    except Exception:
        pass


def _softmax(vals: list[float]) -> list[float]:
    mx = max(vals) if vals else 0.0
    ex = [math.exp(min(v - mx, 88.0)) for v in vals]
    s  = sum(ex) or 1.0
    return [e / s for e in ex]


# ══════════════════════════════════════════════════════════════════════════════
#  NEURAL FLOW LAYOUT  (used by W2)
# ══════════════════════════════════════════════════════════════════════════════

# Column positions (x) and per-column labels
_COL_X      = [0.08, 0.22, 0.42, 0.62, 0.78, 0.92]
_COL_LABELS = ["Input\nTokens", "Embed", "Attn\nBlock 1", "Attn\nBlock 2", "FFN", "Output\nLogits"]
_NODES_PER_COL = 7


def _node_positions() -> list[list[tuple[float, float]]]:
    """Return a list-of-columns, each containing (x, y) tuples for nodes."""
    cols = []
    for cx in _COL_X:
        ys   = [0.15 + i * (0.70 / (_NODES_PER_COL - 1)) for i in range(_NODES_PER_COL)]
        cols.append([(cx, y) for y in ys])
    return cols


_NODE_POS = _node_positions()


# ══════════════════════════════════════════════════════════════════════════════
#  LiveWire
# ══════════════════════════════════════════════════════════════════════════════

class LiveWire:
    """
    Manages 5 separate live matplotlib windows during Neuryx training.

    Parameters
    ----------
    vocab_registry : list[str]   full vocabulary (Cipher.registry)
    cipher         : Cipher      for decoding generated samples
    model          : Lattice     the model being trained
    update_every   : int         redraw all windows every N steps
    sample_every   : int         generate a text sample every N steps
    max_vocab_show : int         tokens shown in heatmap
    """

    def __init__(
        self,
        vocab_registry: list[str],
        cipher,
        model,
        update_every:   int = 8,
        sample_every:   int = 60,
        max_vocab_show: int = 40,
    ):
        self._ok          = _MPL_OK
        self.vocab        = vocab_registry
        self.cipher       = cipher
        self.model        = model
        self.every        = max(update_every, 1)
        self.sample_every = max(sample_every, 1)
        self.n_show       = min(max_vocab_show, len(vocab_registry))

        # Accumulators
        self._steps:   list[int]   = []
        self._losses:  list[float] = []
        self._smooth:  list[float] = []
        self._lrs:     list[float] = []
        self._gnorms:  list[float] = []
        self._samples: list[str]   = []

        # Matplotlib handles
        self._figs:  dict = {}
        self._axes:  dict = {}
        self._arts:  dict = {}

        # Neural flow state
        self._heat = [[0.0] * _NODES_PER_COL for _ in _COL_X]  # per-node brightness
        self._wave_pos = 0.0   # wave front position (0.0–1.0 across columns)

        self._t0 = time.time()

    # ── Public ────────────────────────────────────────────────────────────────

    def open(self) -> bool:
        if not self._ok:
            print("[LiveWire] matplotlib not available — live graphs disabled.")
            return False
        try:
            _plt.ion()
            self._spawn_w1_loss()
            self._spawn_w2_flow()
            self._spawn_w3_heat()
            self._spawn_w4_grad()
            self._spawn_w5_out()
            _plt.pause(0.05)
            return True
        except Exception as exc:
            print(f"[LiveWire] Window build error: {exc}")
            self._ok = False
            return False

    def tick(
        self,
        step:    int,
        total:   int,
        loss:    float,
        lr:      float,
        logits:  list | None = None,
        params:  list | None = None,
    ) -> None:
        if not self._ok:
            return

        self._steps.append(step + 1)
        self._losses.append(loss)
        self._lrs.append(lr)

        alpha = 0.10
        prev  = self._smooth[-1] if self._smooth else loss
        self._smooth.append(alpha * loss + (1 - alpha) * prev)

        if params:
            gsum = sum(abs(p.delta) for p in params)
            self._gnorms.append(gsum / max(len(params), 1))
        else:
            self._gnorms.append(0.0)

        # Advance neural flow wave
        self._advance_wave(loss, step, total)

        if (step + 1) % self.every == 0 or step == total - 1:
            self._draw_w1_loss()
            self._draw_w2_flow()
            if logits is not None:
                self._draw_w3_heat(logits)
            self._draw_w4_grad()
            try:
                _plt.pause(0.001)
            except Exception:
                pass

        if (step + 1) % self.sample_every == 0 or step == total - 1:
            txt = self._sample()
            self._samples.append(f"Step {step+1:>5}:  {txt}")
            self._draw_w5_out()
            try:
                _plt.pause(0.001)
            except Exception:
                pass

    def close(self) -> None:
        if not self._ok:
            return
        _plt.ioff()
        for fig in self._figs.values():
            try:
                fig.canvas.draw(); fig.canvas.flush_events()
            except Exception:
                pass

    # ── W1 · Loss Pulse ───────────────────────────────────────────────────────

    def _spawn_w1_loss(self):
        fig, ax = _plt.subplots(figsize=(7, 4), num="W1 · Neuryx — Loss Pulse")
        fig.patch.set_facecolor(_BG)
        _dark(fig, [ax])
        fig.suptitle("W1  ·  Loss Pulse", color=_TEAL, fontweight="bold",
                     fontsize=11, x=0.02, ha="left")
        _place(fig, 20, 30, 700, 420)

        r, = ax.plot([], [], color=_VIO,  lw=0.7, alpha=0.45, label="raw")
        s, = ax.plot([], [], color=_TEAL, lw=2.4, label="EMA smooth")
        ax.set_xlabel("Step", color=_FG)
        ax.set_ylabel("Cross-Entropy Loss", color=_FG)
        ax.legend(fontsize=8, facecolor=_PAN, edgecolor=_DIM, labelcolor=_FG, loc="upper right")
        _stl(fig)

        self._figs["w1"] = fig; self._axes["w1"] = ax
        self._arts["w1r"] = r;  self._arts["w1s"] = s

    def _draw_w1_loss(self):
        fig, ax = self._figs["w1"], self._axes["w1"]
        s = self._steps
        self._arts["w1r"].set_data(s, self._losses)
        self._arts["w1s"].set_data(s, self._smooth)
        if s:
            ax.set_xlim(0, max(s) + 1)
            lo = min(self._losses) * 0.88; hi = max(self._losses) * 1.12
            if lo < hi: ax.set_ylim(lo, hi)
        for c in list(ax.collections): c.remove()
        if len(s) > 1:
            ax.fill_between(s, self._smooth, alpha=0.09, color=_TEAL)
        for t in list(ax.texts): t.remove()
        if self._smooth:
            ax.text(0.97, 0.95, f"loss  {self._smooth[-1]:.4f}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=9,
                    color=_TEAL, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc=_PAN, ec=_DIM, alpha=0.85))
        try: fig.canvas.draw(); fig.canvas.flush_events()
        except Exception: pass

    # ── W2 · Neural Flow Animation ────────────────────────────────────────────

    def _spawn_w2_flow(self):
        fig, ax = _plt.subplots(figsize=(11, 5), num="W2 · Neuryx — Neural Flow")
        fig.patch.set_facecolor(_BG)
        ax.set_facecolor(_BG)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis("off")
        fig.suptitle("W2  ·  Neural Flow  — data moving through the network",
                     color=_CORAL, fontweight="bold", fontsize=11, x=0.02, ha="left")
        _place(fig, 730, 30, 1050, 500)
        _stl(fig)
        self._figs["w2"] = fig; self._axes["w2"] = ax

    def _advance_wave(self, loss: float, step: int, total: int) -> None:
        """Move the wave front and update per-node heat values."""
        # Wave speed: proportional to step progress, faster early on
        speed = 0.04 + 0.02 * (1 - step / max(total, 1))
        self._wave_pos = (self._wave_pos + speed) % (len(_COL_X) + 2)

        # Intensity inversely proportional to (normalised) loss
        # — brighter as loss falls
        base_loss = max(self._losses[0], 1e-6) if self._losses else 3.0
        cur_loss  = self._losses[-1] if self._losses else 3.0
        intensity = 1.0 - min(cur_loss / base_loss, 1.0)  # 0 = bad, 1 = perfect

        for ci, col_nodes in enumerate(self._heat):
            # Distance from wave front (column index vs wave position)
            dist = abs(ci - self._wave_pos % len(_COL_X))
            wave_val = max(0.0, 1.0 - dist / 2.0)

            for ni in range(_NODES_PER_COL):
                # Organic noise: each node has its own jitter
                jitter     = 0.12 * math.sin(step * 0.3 + ci * 1.7 + ni * 0.9)
                target_heat = wave_val * (0.55 + 0.45 * intensity) + jitter
                target_heat = max(0.0, min(1.0, target_heat))
                # Smooth toward target (momentum)
                self._heat[ci][ni] = (
                    0.65 * self._heat[ci][ni] + 0.35 * target_heat
                )

    def _draw_w2_flow(self):
        fig, ax = self._figs["w2"], self._axes["w2"]
        ax.clear()
        ax.set_facecolor(_BG)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis("off")

        n_col  = len(_COL_X)
        loss_progress = 0.0
        if len(self._smooth) > 1:
            lo = self._smooth[0]; hi = self._smooth[-1]
            loss_progress = 1.0 - min(hi / max(lo, 1e-6), 1.0)

        # ── Draw connections (lines between adjacent columns) ────────────────
        for ci in range(n_col - 1):
            for ni, (x0, y0) in enumerate(_NODE_POS[ci]):
                for nj, (x1, y1) in enumerate(_NODE_POS[ci + 1]):
                    heat_avg = (self._heat[ci][ni] + self._heat[ci + 1][nj]) * 0.5
                    alpha    = 0.04 + 0.22 * heat_avg
                    lw       = 0.4 + 1.2 * heat_avg
                    # Color: cool → warm based on heat
                    r = heat_avg
                    g = heat_avg * 0.6
                    b = 1.0 - heat_avg * 0.7
                    ax.plot([x0, x1], [y0, y1], color=(r, g, b), alpha=alpha,
                            linewidth=lw, solid_capstyle="round", zorder=1)

        # ── Draw animated data packets along connections ──────────────────────
        wave_frac = (self._wave_pos % (n_col + 1)) / (n_col + 1)
        for ci in range(n_col - 1):
            x0c = _COL_X[ci]; x1c = _COL_X[ci + 1]
            col_frac = (wave_frac * (n_col + 1) - ci)
            if 0.0 <= col_frac <= 1.0:
                # Packet travels along this column pair
                for ni in range(0, _NODES_PER_COL, 2):
                    x0, y0 = _NODE_POS[ci][ni]
                    x1, y1 = _NODE_POS[ci + 1][min(ni + 1, _NODES_PER_COL - 1)]
                    px = x0 + (x1 - x0) * col_frac
                    py = y0 + (y1 - y0) * col_frac
                    # Glow effect: outer circle + inner dot
                    ax.scatter([px], [py], s=80, c=[_TEAL], alpha=0.25,
                               zorder=3, linewidths=0)
                    ax.scatter([px], [py], s=22, c=["white"], alpha=0.9,
                               zorder=4, linewidths=0)

        # ── Draw nodes ────────────────────────────────────────────────────────
        cmap_node = _LSC.from_list("nh", [_DIM, _BLUE, _TEAL, _AMBER, _CORAL])
        for ci, col in enumerate(_NODE_POS):
            for ni, (nx, ny) in enumerate(col):
                heat = self._heat[ci][ni]
                node_color = cmap_node(heat)
                # Outer glow
                ax.scatter([nx], [ny], s=320 * (0.3 + 0.7 * heat),
                           c=[node_color], alpha=0.18, zorder=2, linewidths=0)
                # Core
                ax.scatter([nx], [ny], s=90 * (0.5 + 0.5 * heat),
                           c=[node_color], alpha=0.92, zorder=5, linewidths=0)

        # ── Column labels ─────────────────────────────────────────────────────
        for ci, (cx, label) in enumerate(zip(_COL_X, _COL_LABELS)):
            col_heat = sum(self._heat[ci]) / _NODES_PER_COL
            alpha    = 0.55 + 0.45 * col_heat
            ax.text(cx, 0.05, label, ha="center", va="top",
                    fontsize=7.5, color=_FG, alpha=alpha,
                    fontweight="bold", fontfamily="monospace")

        # ── Live metrics overlay ──────────────────────────────────────────────
        if self._smooth:
            ax.text(0.5, 0.97,
                    f"Loss: {self._smooth[-1]:.4f}   "
                    f"Step: {self._steps[-1]}   "
                    f"Depth: {self.model.depth}   "
                    f"Rifts: {self.model.rifts}",
                    ha="center", va="top", transform=ax.transAxes,
                    fontsize=8, color=_FG, fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.25", fc=_DIM, ec=_TEAL, alpha=0.7))

        try: fig.canvas.draw(); fig.canvas.flush_events()
        except Exception: pass

    # ── W3 · Token Heatmap ────────────────────────────────────────────────────

    def _spawn_w3_heat(self):
        n = self.n_show
        fig, ax = _plt.subplots(figsize=(10, 3.5), num="W3 · Neuryx — Token Heatmap")
        fig.patch.set_facecolor(_BG)
        _dark(fig, [ax])
        fig.suptitle("W3  ·  Token Probability  (softmax over vocabulary)",
                     color=_AMBER, fontweight="bold", fontsize=11, x=0.02, ha="left")
        _place(fig, 20, 560, 980, 400)

        cmap = _LSC.from_list("th", [_BG, _VIO, _AMBER])
        im   = ax.imshow([[0.0] * n], aspect="auto", cmap=cmap,
                         vmin=0.0, vmax=0.1, interpolation="nearest")
        labs = [str(t)[:7] for t in self.vocab[:n]]
        ax.set_xticks(range(n)); ax.set_xticklabels(labs, rotation=65, fontsize=6.5, color=_FG)
        ax.set_yticks([])
        ax.set_xlabel("Vocabulary token", color=_FG, fontsize=9)
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.38)
        cb.ax.tick_params(colors=_FG, labelsize=7)
        cb.set_label("Probability", color=_FG, fontsize=8)
        _stl(fig)

        self._figs["w3"] = fig; self._axes["w3"] = ax; self._arts["w3"] = im

    def _draw_w3_heat(self, logits: list):
        n   = self.n_show
        fig, ax, im = self._figs["w3"], self._axes["w3"], self._arts["w3"]
        raw   = [v.val if hasattr(v, "val") else float(v) for v in logits[:n]]
        while len(raw) < n: raw.append(0.0)
        probs = _softmax(raw)
        im.set_data([probs])
        im.set_clim(0.0, max(probs) or 0.01)
        for t in list(ax.texts): t.remove()
        top3 = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:3]
        for idx in top3:
            if probs[idx] > 0.01:
                ax.text(idx, 0, f"{probs[idx]:.2f}", ha="center", va="center",
                        fontsize=6.5, color="white", fontweight="bold")
        try: fig.canvas.draw(); fig.canvas.flush_events()
        except Exception: pass

    # ── W4 · Gradient Health ─────────────────────────────────────────────────

    def _spawn_w4_grad(self):
        fig, ax = _plt.subplots(figsize=(6, 3.5), num="W4 · Neuryx — Gradient Health")
        fig.patch.set_facecolor(_BG)
        _dark(fig, [ax])
        fig.suptitle("W4  ·  Gradient Health  (mean |∇|)",
                     color=_GREEN, fontweight="bold", fontsize=11, x=0.02, ha="left")
        _place(fig, 1010, 560, 590, 400)
        line, = ax.plot([], [], color=_GREEN, lw=1.8)
        ax.set_xlabel("Step", color=_FG); ax.set_ylabel("Mean |grad|", color=_FG)
        _stl(fig)
        self._figs["w4"] = fig; self._axes["w4"] = ax; self._arts["w4"] = line

    def _draw_w4_grad(self):
        fig, ax, line = self._figs["w4"], self._axes["w4"], self._arts["w4"]
        pairs = [(s, g) for s, g in zip(self._steps, self._gnorms) if g > 1e-12]
        if len(pairs) < 2: return
        vs, vg = zip(*pairs)
        line.set_data(vs, vg)
        for c in list(ax.collections): c.remove()
        ax.fill_between(vs, vg, alpha=0.18, color=_GREEN)
        ax.set_xlim(0, max(vs) + 1)
        lo, hi = min(vg), max(vg)
        if lo > 0 and hi > lo:
            try: ax.set_yscale("log"); ax.set_ylim(lo * 0.4, hi * 2.5)
            except Exception: ax.set_yscale("linear"); ax.set_ylim(0, hi * 1.15)
        for t in list(ax.texts): t.remove()
        ax.text(0.97, 0.93, f"∇  {vg[-1]:.2e}", transform=ax.transAxes, ha="right",
                fontsize=9, color=_GREEN, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc=_PAN, ec=_DIM, alpha=0.85))
        try: fig.canvas.draw(); fig.canvas.flush_events()
        except Exception: pass

    # ── W5 · Live Output ─────────────────────────────────────────────────────

    def _spawn_w5_out(self):
        fig, ax = _plt.subplots(figsize=(7, 10), num="W5 · Neuryx — Live Output")
        fig.patch.set_facecolor(_BG)
        _dark(fig, [ax])
        fig.suptitle("W5  ·  Live Output  (generated samples during training)",
                     color=_PINK, fontweight="bold", fontsize=11, x=0.02, ha="left")
        _place(fig, 1610, 30, 680, 930)
        ax.axis("off")
        ax.text(0.5, 0.5, "Training in progress…\nFirst sample appears soon.",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color=_DIM, style="italic")
        _stl(fig)
        self._figs["w5"] = fig; self._axes["w5"] = ax

    def _draw_w5_out(self):
        fig, ax = self._figs["w5"], self._axes["w5"]
        ax.clear(); ax.axis("off"); _dark(fig, [ax])
        elapsed = time.time() - self._t0
        ax.text(0.01, 0.987,
                f"Samples: {len(self._samples)}   Steps: {len(self._steps)}   "
                f"Elapsed: {elapsed:.0f}s",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=8, color=_DIM, fontfamily="monospace")
        ax.axhline(0.975, color=_DIM, lw=0.6)
        y = 0.955; row_h = 0.038
        for entry in self._samples[-24:]:
            if ":" in entry:
                lbl, _, txt = entry.partition(":")
                ax.text(0.01, y, lbl + ":", transform=ax.transAxes, ha="left", va="top",
                        fontsize=8, color=_TEAL, fontweight="bold", fontfamily="monospace")
                ax.text(0.25, y, txt.strip()[:72], transform=ax.transAxes, ha="left", va="top",
                        fontsize=8, color=_FG, fontfamily="monospace")
            else:
                ax.text(0.01, y, entry[:80], transform=ax.transAxes, ha="left", va="top",
                        fontsize=8, color=_FG, fontfamily="monospace")
            y -= row_h
            if y < 0.01: break
        try: fig.canvas.draw(); fig.canvas.flush_events()
        except Exception: pass

    # ── Internal: generate live sample ───────────────────────────────────────

    def _sample(self) -> str:
        try:
            import sys, os
            _r = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if _r not in sys.path: sys.path.insert(0, _r)
            from core.forge import Forge
            from core.apex  import Apex
            dummy = Apex(self.model.params, pulse=0.0)
            forge = Forge(self.model, dummy, [])
            ids   = forge.infer(
                context=[self.cipher.seal],
                n_steps=min(self.model.horizon // 2, 48),
                temperature=0.42,
                stop_token=self.cipher.seal,
            )
            return self.cipher.decipher(ids) or "<empty>"
        except Exception as e:
            return f"<{e}>"
