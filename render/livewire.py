"""
render/livewire.py
Five separate live matplotlib windows that update in real time during training.

W1  Loss Pulse     — raw + EMA-smoothed cross-entropy loss
W2  Neural Flow    — animated: nodes light up, data packets travel layer→layer
W3  Token Heatmap  — softmax probability over the vocabulary at every step
W4  Gradient Health— mean |∇| magnitude (log scale)
W5  Live Output    — text samples generated periodically during training

Backend detection  (Linux Mint / Ubuntu / Windows / macOS)
──────────────────────────────────────────────────────────
1.  Try to import tkinter directly  →  if it works, use TkAgg
2.  Try Qt5 / Qt6                   →  fallback option
3.  Final fallback: Agg (no windows, but no crash either)

The probe happens BEFORE matplotlib.pyplot is imported to avoid the
"No module named tkinter" crash that occurs when pyplot triggers
a backend load inside its own import chain.
"""

from __future__ import annotations
import math
import time

# ══════════════════════════════════════════════════════════════════════════════
#  BACKEND SELECTION  — done at module load, before any pyplot import
# ══════════════════════════════════════════════════════════════════════════════

_MPL_OK     = False
_plt        = None
_LSC        = None
_BACKEND    = "Agg"           # will be updated by the probe below
_HAS_WINDOW = False           # True only when a real GUI backend loaded

try:
    import matplotlib as _mpl

    # ── Probe 1: TkAgg — available on Linux Mint, Windows, macOS ─────────────
    _tk_ok = False
    try:
        import tkinter as _tk_probe
        _tk_probe.Tk().destroy()          # verify display connection
        _tk_ok = True
    except Exception:
        pass

    if _tk_ok:
        _BACKEND    = "TkAgg"
        _HAS_WINDOW = True
    else:
        # ── Probe 2: Qt backends ─────────────────────────────────────────────
        for _be in ("Qt5Agg", "QtAgg", "Qt6Agg", "WXAgg", "MacOSX"):
            try:
                from matplotlib.backends import backend_registry as _breg
                _breg.load_backend_module(_be)
                _BACKEND    = _be
                _HAS_WINDOW = True
                break
            except Exception:
                continue

    # ── Apply the chosen backend, then import pyplot ──────────────────────────
    _mpl.use(_BACKEND)
    import matplotlib.pyplot as _plt
    from matplotlib.colors import LinearSegmentedColormap as _LSC
    _MPL_OK = True

except ImportError:
    pass          # matplotlib not installed — all windows silently disabled

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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dark(fig, axes: list) -> None:
    """Apply dark theme to a figure and its axes."""
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
    """Move the figure window to screen position (x,y) with size w×h pixels."""
    try:
        mgr = fig.canvas.manager
        win = getattr(mgr, "window", None)
        if win is None:
            return
        if hasattr(win, "wm_geometry"):       # TkAgg
            win.wm_geometry(f"{w}x{h}+{x}+{y}")
        elif hasattr(win, "setGeometry"):      # Qt
            win.setGeometry(x, y, w, h)
        elif hasattr(win, "SetPosition"):      # WX
            win.SetPosition((x, y)); win.SetSize((w, h))
    except Exception:
        pass


def _tight(fig) -> None:
    """tight_layout — suppress all errors (empty axes, log scale, etc.)."""
    try:
        fig.tight_layout(rect=[0, 0, 1, 0.90])
    except Exception:
        pass


def _softmax(vals: list[float]) -> list[float]:
    mx = max(vals) if vals else 0.0
    ex = [math.exp(min(v - mx, 88.0)) for v in vals]
    s  = sum(ex) or 1.0
    return [e / s for e in ex]


def _flush(fig) -> None:
    """Draw and flush a figure — swallows all errors so a bad draw never crashes."""
    try:
        fig.canvas.draw()
        fig.canvas.flush_events()
    except Exception:
        pass


# ── Neural flow layout constants ──────────────────────────────────────────────
_COL_X      = [0.08, 0.22, 0.40, 0.58, 0.76, 0.92]
_COL_LABELS = ["Input\nTokens", "Embed", "Attn\nBlock 1",
               "Attn\nBlock 2", "FFN", "Output\nLogits"]
_N_NODES    = 7


def _node_pos() -> list[list[tuple[float, float]]]:
    cols = []
    for cx in _COL_X:
        ys = [0.15 + i * (0.70 / (_N_NODES - 1)) for i in range(_N_NODES)]
        cols.append([(cx, y) for y in ys])
    return cols


_NODE_POS = _node_pos()


# ══════════════════════════════════════════════════════════════════════════════
#  LiveWire
# ══════════════════════════════════════════════════════════════════════════════

class LiveWire:
    """
    Manages 5 separate live matplotlib windows during Neuryx training.

    Designed to be completely crash-proof:
    - Every draw operation is wrapped in try/except
    - If a window dies mid-training the rest continue
    - Backend falls back to Agg if no display is available

    Parameters
    ----------
    vocab_registry : list[str]
    cipher         : Cipher
    model          : Lattice
    update_every   : int     — redraw all windows every N steps
    sample_every   : int     — generate a text sample every N steps
    max_vocab_show : int     — tokens shown in the heatmap
    """

    def __init__(
        self,
        vocab_registry: list[str],
        cipher,
        model,
        update_every:   int = 6,
        sample_every:   int = 50,
        max_vocab_show: int = 40,
    ):
        self._ok          = _MPL_OK
        self._has_win     = _HAS_WINDOW
        self.vocab        = vocab_registry
        self.cipher       = cipher
        self.model        = model
        self.every        = max(update_every, 1)
        self.sample_every = max(sample_every, 1)
        self.n_show       = min(max_vocab_show, len(vocab_registry))

        # Data accumulators
        self._steps:   list[int]   = []
        self._losses:  list[float] = []
        self._smooth:  list[float] = []
        self._lrs:     list[float] = []
        self._gnorms:  list[float] = []
        self._samples: list[str]   = []

        # Figure/axes/artist handles — keyed by window id
        self._figs: dict = {}
        self._axes: dict = {}
        self._arts: dict = {}

        # Neural-flow animation state
        self._heat     = [[0.0] * _N_NODES for _ in _COL_X]
        self._wave_pos = 0.0

        self._t0 = time.time()

    # ═════════════════════════════════════════════════════════════════════════
    #  Public API
    # ═════════════════════════════════════════════════════════════════════════

    def open(self) -> bool:
        """
        Spawn all five windows.
        Returns True if at least one window opened successfully.
        """
        if not self._ok:
            print("[LiveWire] matplotlib not installed — pip install matplotlib")
            return False

        try:
            _plt.ion()
        except Exception:
            pass

        any_ok = False
        for build_fn in [
            self._spawn_w1_loss,
            self._spawn_w2_flow,
            self._spawn_w3_heat,
            self._spawn_w4_grad,
            self._spawn_w5_out,
        ]:
            try:
                build_fn()
                any_ok = True
            except Exception as e:
                print(f"[LiveWire] Could not open {build_fn.__name__}: {e}")

        try:
            _plt.pause(0.05)
        except Exception:
            pass

        if not _HAS_WINDOW:
            print(f"[LiveWire] Backend: {_BACKEND} (no display) — "
                  "graphs saved to neuryx_dashboard.png after training")

        return any_ok

    def tick(
        self,
        step:    int,
        total:   int,
        loss:    float,
        lr:      float,
        logits:  list | None = None,
        params:  list | None = None,
    ) -> None:
        """Feed one training step into all windows. Never raises."""
        if not self._ok:
            return

        try:
            self._accumulate(step, loss, lr, params)
            self._advance_wave(loss, step, total)

            if (step + 1) % self.every == 0 or step == total - 1:
                self._safe_draw(self._draw_w1_loss)
                self._safe_draw(self._draw_w2_flow)
                if logits is not None:
                    self._safe_draw(self._draw_w3_heat, logits)
                self._safe_draw(self._draw_w4_grad)
                try:
                    _plt.pause(0.001)
                except Exception:
                    pass

            if (step + 1) % self.sample_every == 0 or step == total - 1:
                txt = self._make_sample()
                self._samples.append(f"Step {step+1:>5}:  {txt}")
                self._safe_draw(self._draw_w5_out)
                try:
                    _plt.pause(0.001)
                except Exception:
                    pass

        except Exception:
            pass   # never crash the training loop

    def close(self) -> None:
        """Freeze all windows so they stay visible after training ends."""
        if not self._ok:
            return
        try:
            _plt.ioff()
        except Exception:
            pass
        for fig in self._figs.values():
            try:
                fig.canvas.draw()
                fig.canvas.flush_events()
            except Exception:
                pass

    # ─────────────────────────────────────────────────────────────────────────

    def _safe_draw(self, fn, *args):
        """Call a draw function; swallow all exceptions."""
        try:
            fn(*args)
        except Exception:
            pass

    def _accumulate(self, step: int, loss: float, lr: float, params) -> None:
        self._steps.append(step + 1)
        self._losses.append(loss)
        self._lrs.append(lr)
        alpha = 0.10
        prev  = self._smooth[-1] if self._smooth else loss
        self._smooth.append(alpha * loss + (1 - alpha) * prev)
        if params:
            g = sum(abs(p.delta) for p in params) / max(len(params), 1)
            self._gnorms.append(g)
        else:
            self._gnorms.append(0.0)

    # ═════════════════════════════════════════════════════════════════════════
    #  W1 — Loss Pulse
    # ═════════════════════════════════════════════════════════════════════════

    def _spawn_w1_loss(self):
        fig, ax = _plt.subplots(figsize=(7, 4), num="W1 · Loss Pulse")
        _dark(fig, [ax])
        fig.suptitle("W1  ·  Loss Pulse", color=_TEAL,
                     fontweight="bold", fontsize=11, x=0.02, ha="left")
        _place(fig, 20, 30, 700, 420)
        r, = ax.plot([], [], color=_VIO,  lw=0.7, alpha=0.5, label="raw")
        s, = ax.plot([], [], color=_TEAL, lw=2.4, label="EMA")
        ax.set_xlabel("Step"); ax.set_ylabel("Loss")
        ax.legend(fontsize=8, facecolor=_PAN, edgecolor=_DIM,
                  labelcolor=_FG, loc="upper right")
        _tight(fig)
        self._figs["w1"] = fig; self._axes["w1"] = ax
        self._arts["w1r"] = r;  self._arts["w1s"] = s

    def _draw_w1_loss(self):
        fig = self._figs["w1"]; ax = self._axes["w1"]
        s   = self._steps
        self._arts["w1r"].set_data(s, self._losses)
        self._arts["w1s"].set_data(s, self._smooth)
        if s:
            ax.set_xlim(0, max(s) + 1)
            lo = min(self._losses) * 0.88
            hi = max(self._losses) * 1.12
            if lo < hi:
                ax.set_ylim(lo, hi)
        for c in list(ax.collections): c.remove()
        if len(s) > 1:
            ax.fill_between(s, self._smooth, alpha=0.09, color=_TEAL)
        for t in list(ax.texts): t.remove()
        if self._smooth:
            ax.text(0.97, 0.95, f"loss  {self._smooth[-1]:.4f}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=9, color=_TEAL, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc=_PAN,
                              ec=_DIM, alpha=0.9))
        _flush(fig)

    # ═════════════════════════════════════════════════════════════════════════
    #  W2 — Neural Flow  (animated)
    # ═════════════════════════════════════════════════════════════════════════

    def _spawn_w2_flow(self):
        fig, ax = _plt.subplots(figsize=(11, 5), num="W2 · Neural Flow")
        fig.patch.set_facecolor(_BG); ax.set_facecolor(_BG)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        fig.suptitle("W2  ·  Neural Flow — data moving through the network",
                     color=_CORAL, fontweight="bold", fontsize=11,
                     x=0.02, ha="left")
        _place(fig, 730, 30, 1050, 500)
        _tight(fig)
        self._figs["w2"] = fig; self._axes["w2"] = ax

    def _advance_wave(self, loss: float, step: int, total: int) -> None:
        speed = 0.06 + 0.03 * (1 - step / max(total, 1))
        self._wave_pos = (self._wave_pos + speed) % (len(_COL_X) + 2)
        base = max(self._losses[0], 1e-6) if self._losses else 3.0
        cur  = self._losses[-1] if self._losses else 3.0
        intensity = 1.0 - min(cur / base, 1.0)
        for ci in range(len(_COL_X)):
            dist = abs(ci - self._wave_pos % len(_COL_X))
            wave = max(0.0, 1.0 - dist / 2.0)
            for ni in range(_N_NODES):
                jitter = 0.12 * math.sin(step * 0.3 + ci * 1.7 + ni * 0.9)
                target = max(0.0, min(1.0, wave * (0.55 + 0.45 * intensity) + jitter))
                self._heat[ci][ni] = 0.65 * self._heat[ci][ni] + 0.35 * target

    def _draw_w2_flow(self):
        fig = self._figs["w2"]; ax = self._axes["w2"]
        ax.clear()
        ax.set_facecolor(_BG); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

        cmap = _LSC.from_list("nh", [_DIM, _BLUE, _TEAL, _AMBER, _CORAL])
        nc   = len(_COL_X)

        # Connections
        for ci in range(nc - 1):
            for ni, (x0, y0) in enumerate(_NODE_POS[ci]):
                for nj, (x1, y1) in enumerate(_NODE_POS[ci + 1]):
                    h = (self._heat[ci][ni] + self._heat[ci+1][nj]) * 0.5
                    r = h; g = h * 0.55; b = 1.0 - h * 0.65
                    ax.plot([x0, x1], [y0, y1],
                            color=(r, g, b),
                            alpha=0.04 + 0.22 * h,
                            linewidth=0.3 + 1.4 * h,
                            solid_capstyle="round", zorder=1)

        # Travelling data packets
        wf = (self._wave_pos % (nc + 1)) / (nc + 1)
        for ci in range(nc - 1):
            col_frac = wf * (nc + 1) - ci
            if 0.0 <= col_frac <= 1.0:
                for ni in range(0, _N_NODES, 2):
                    x0, y0 = _NODE_POS[ci][ni]
                    x1, y1 = _NODE_POS[ci+1][min(ni+1, _N_NODES-1)]
                    px = x0 + (x1 - x0) * col_frac
                    py = y0 + (y1 - y0) * col_frac
                    ax.scatter([px], [py], s=90, c=[_TEAL],
                               alpha=0.22, zorder=3, linewidths=0)
                    ax.scatter([px], [py], s=25, c=["white"],
                               alpha=0.92, zorder=4, linewidths=0)

        # Nodes
        for ci, col in enumerate(_NODE_POS):
            for ni, (nx, ny) in enumerate(col):
                h  = self._heat[ci][ni]
                nc_ = cmap(h)
                ax.scatter([nx], [ny], s=300 * (0.3 + 0.7 * h),
                           c=[nc_], alpha=0.18, zorder=2, linewidths=0)
                ax.scatter([nx], [ny], s=80 * (0.5 + 0.5 * h),
                           c=[nc_], alpha=0.95, zorder=5, linewidths=0)

        # Column labels
        for ci, (cx, lbl) in enumerate(zip(_COL_X, _COL_LABELS)):
            ch = sum(self._heat[ci]) / _N_NODES
            ax.text(cx, 0.04, lbl, ha="center", va="top",
                    fontsize=7.5, color=_FG,
                    alpha=0.55 + 0.45 * ch,
                    fontweight="bold", fontfamily="monospace")

        # Metrics overlay
        if self._smooth:
            ax.text(0.5, 0.97,
                    f"Loss: {self._smooth[-1]:.4f}   "
                    f"Step: {self._steps[-1]}   "
                    f"Blocks: {self.model.rifts}   "
                    f"Dim: {self.model.depth}",
                    ha="center", va="top", transform=ax.transAxes,
                    fontsize=8, color=_FG, fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.25",
                              fc=_DIM, ec=_TEAL, alpha=0.75))
        _flush(fig)

    # ═════════════════════════════════════════════════════════════════════════
    #  W3 — Token Heatmap
    # ═════════════════════════════════════════════════════════════════════════

    def _spawn_w3_heat(self):
        n = self.n_show
        fig, ax = _plt.subplots(figsize=(10, 3.5), num="W3 · Token Heatmap")
        _dark(fig, [ax])
        fig.suptitle("W3  ·  Token Probability  (softmax each step)",
                     color=_AMBER, fontweight="bold", fontsize=11,
                     x=0.02, ha="left")
        _place(fig, 20, 560, 980, 400)
        cmap = _LSC.from_list("th", [_BG, _VIO, _AMBER])
        im   = ax.imshow([[0.0] * n], aspect="auto", cmap=cmap,
                         vmin=0.0, vmax=0.1, interpolation="nearest")
        labs = [str(t)[:7] for t in self.vocab[:n]]
        ax.set_xticks(range(n))
        ax.set_xticklabels(labs, rotation=65, fontsize=6.5, color=_FG)
        ax.set_yticks([])
        ax.set_xlabel("Vocabulary token", color=_FG, fontsize=9)
        cb = fig.colorbar(im, ax=ax, orientation="horizontal",
                          fraction=0.05, pad=0.40)
        cb.ax.tick_params(colors=_FG, labelsize=7)
        cb.set_label("Probability", color=_FG, fontsize=8)
        _tight(fig)
        self._figs["w3"] = fig; self._axes["w3"] = ax; self._arts["w3"] = im

    def _draw_w3_heat(self, logits: list):
        n   = self.n_show
        fig = self._figs["w3"]; ax = self._axes["w3"]; im = self._arts["w3"]
        raw = [v.val if hasattr(v, "val") else float(v) for v in logits[:n]]
        while len(raw) < n:
            raw.append(0.0)
        probs = _softmax(raw)
        im.set_data([probs])
        im.set_clim(0.0, max(probs) or 0.01)
        for t in list(ax.texts): t.remove()
        top3 = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:3]
        for idx in top3:
            if probs[idx] > 0.01:
                ax.text(idx, 0, f"{probs[idx]:.2f}",
                        ha="center", va="center",
                        fontsize=6.5, color="white", fontweight="bold")
        _flush(fig)

    # ═════════════════════════════════════════════════════════════════════════
    #  W4 — Gradient Health
    # ═════════════════════════════════════════════════════════════════════════

    def _spawn_w4_grad(self):
        fig, ax = _plt.subplots(figsize=(6, 3.8), num="W4 · Gradient Health")
        _dark(fig, [ax])
        fig.suptitle("W4  ·  Gradient Health  (mean |grad|)",
                     color=_GREEN, fontweight="bold", fontsize=11,
                     x=0.02, ha="left")
        _place(fig, 1010, 560, 590, 400)
        line, = ax.plot([], [], color=_GREEN, lw=1.8)
        ax.set_xlabel("Step"); ax.set_ylabel("Mean |grad|")
        _tight(fig)
        self._figs["w4"] = fig; self._axes["w4"] = ax; self._arts["w4"] = line

    def _draw_w4_grad(self):
        fig = self._figs["w4"]; ax = self._axes["w4"]; line = self._arts["w4"]
        pairs = [(s, g) for s, g in zip(self._steps, self._gnorms) if g > 1e-12]
        if len(pairs) < 2:
            return
        vs, vg = zip(*pairs)
        line.set_data(vs, vg)
        for c in list(ax.collections): c.remove()
        ax.fill_between(vs, vg, alpha=0.18, color=_GREEN)
        ax.set_xlim(0, max(vs) + 1)
        lo, hi = min(vg), max(vg)
        if lo > 0 and hi > lo * 1.01:
            try:
                ax.set_yscale("log")
                ax.set_ylim(lo * 0.4, hi * 2.5)
            except Exception:
                ax.set_yscale("linear")
                ax.set_ylim(0, hi * 1.15)
        for t in list(ax.texts): t.remove()
        ax.text(0.97, 0.93, f"grad  {vg[-1]:.2e}",
                transform=ax.transAxes, ha="right",
                fontsize=9, color=_GREEN, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc=_PAN,
                          ec=_DIM, alpha=0.9))
        _flush(fig)

    # ═════════════════════════════════════════════════════════════════════════
    #  W5 — Live Output
    # ═════════════════════════════════════════════════════════════════════════

    def _spawn_w5_out(self):
        fig, ax = _plt.subplots(figsize=(7, 10), num="W5 · Live Output")
        _dark(fig, [ax]); ax.axis("off")
        fig.suptitle("W5  ·  Live Output  (text samples during training)",
                     color=_PINK, fontweight="bold", fontsize=11,
                     x=0.02, ha="left")
        _place(fig, 1610, 30, 680, 930)
        ax.text(0.5, 0.5, "Training in progress...\nFirst sample appears soon.",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color=_DIM, style="italic")
        _tight(fig)
        self._figs["w5"] = fig; self._axes["w5"] = ax

    def _draw_w5_out(self):
        fig = self._figs["w5"]; ax = self._axes["w5"]
        ax.clear(); ax.axis("off"); _dark(fig, [ax])
        elapsed = time.time() - self._t0
        ax.text(0.01, 0.988,
                f"Samples: {len(self._samples)}  "
                f"Steps: {len(self._steps)}  "
                f"Elapsed: {elapsed:.0f}s",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=8, color=_DIM, fontfamily="monospace")
        y = 0.955; row = 0.038
        for entry in self._samples[-24:]:
            if ":" in entry:
                lbl, _, txt = entry.partition(":")
                ax.text(0.01, y, lbl + ":", transform=ax.transAxes,
                        ha="left", va="top", fontsize=8,
                        color=_TEAL, fontweight="bold", fontfamily="monospace")
                ax.text(0.25, y, txt.strip()[:72], transform=ax.transAxes,
                        ha="left", va="top", fontsize=8,
                        color=_FG, fontfamily="monospace")
            else:
                ax.text(0.01, y, entry[:80], transform=ax.transAxes,
                        ha="left", va="top", fontsize=8,
                        color=_FG, fontfamily="monospace")
            y -= row
            if y < 0.01:
                break
        _flush(fig)

    # ═════════════════════════════════════════════════════════════════════════
    #  Internal — generate a live text sample
    # ═════════════════════════════════════════════════════════════════════════

    def _make_sample(self) -> str:
        try:
            import sys, os
            _r = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if _r not in sys.path:
                sys.path.insert(0, _r)
            from core.forge import Forge
            from core.apex  import Apex
            dummy = Apex(self.model.params, pulse=0.0)
            forge = Forge(self.model, dummy, [])
            ids   = forge.infer(
                context    = [self.cipher.seal],
                n_steps    = min(self.model.horizon // 2, 48),
                temperature = 0.45,
                stop_token = self.cipher.seal,
            )
            return self.cipher.decipher(ids) or "<empty>"
        except Exception as e:
            return f"<{e}>"
