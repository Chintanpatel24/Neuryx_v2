#!/usr/bin/env python3
"""
neuryx.py  —  Neuryx  ·  General-Purpose Neural AI  v2.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Train a deep transformer on YOUR data, then chat with it.

Flow
────
  1. Choose input file format  (menu appears FIRST)
  2. Enter file path
  3. Training runs — 5 live matplotlib windows update in real time
  4. Chat window opens automatically when training is done
  5. Ask questions about your data — out-of-scope queries are detected

Usage
─────
  python neuryx.py
  python neuryx.py --train mydata.csv --steps 800 --mode word
  python neuryx.py --help
"""

from __future__ import annotations
import sys, os, random, argparse

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

random.seed(42)

from core    import Lattice, Apex, Forge, Retriever, DEFAULT_CFG
from intake  import Portal, Cipher, FORMAT_LABELS
from intake.portal import FMT_TXT, FMT_CSV, FMT_XLSX, FMT_JSON, FMT_TSV, FMT_AUTO
from shell   import (
    LOGO, blank, heading, double_rule, rule,
    section_open, section_close,
    kv, bullet, ok, warn, err, info_line,
    progress, progress_done,
    format_menu, ask_file, ask_int, ask_str,
    teal, green, red, yellow, grey, white, bold,
    W,
)
from render import render_dashboard, LiveWire, launch_chat

_EXT_FMT = {".txt": FMT_TXT, ".csv": FMT_CSV, ".xlsx": FMT_XLSX,
             ".xls": FMT_XLSX, ".json": FMT_JSON, ".tsv": FMT_TSV}
_ALL_EXT      = list(_EXT_FMT.keys())
_NEEDS_COLUMN = {FMT_CSV, FMT_XLSX, FMT_TSV}


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — FORMAT SELECTION  (always before file path)
# ══════════════════════════════════════════════════════════════════════════════

def select_format() -> int:
    """
    Show the format menu and return the chosen FMT_* constant.
    This is always the FIRST interaction — before the user enters a file path.
    """
    blank()
    section_open("Choose Input Format  ·  Select BEFORE entering the file path")
    blank()
    print(f"  {grey('What kind of file will you provide?')}\n")
    format_menu(FORMAT_LABELS)
    blank()
    section_close()
    return ask_int("Format number", list(FORMAT_LABELS.keys()))


def resolve_format(chosen: int, path: str) -> int:
    """If the user chose Auto-detect, resolve from the file extension."""
    if chosen != FMT_AUTO:
        return chosen
    ext = os.path.splitext(path)[1].lower()
    if ext in _EXT_FMT:
        resolved = _EXT_FMT[ext]
        ok(f"Auto-detected → {FORMAT_LABELS[resolved]}")
        return resolved
    warn(f"Cannot auto-detect '{ext}'.")
    return ask_int("Format number", [i for i in FORMAT_LABELS if i != FMT_AUTO])


def ask_column(fmt: int) -> str | None:
    if fmt not in _NEEDS_COLUMN:
        return None
    col = ask_str("Column to use as text  (blank = auto-pick first text column)", "")
    return col.strip() or None


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(cli_path: str | None = None) -> list[str]:
    """
    Interactive data-loading flow.
    Format is selected FIRST, then the file path is requested.
    """
    blank()
    heading("Load Training Data",
            "Format selection comes first — then the file path.")

    # 1. Format
    fmt = select_format()

    # 2. File path
    path = cli_path or ask_file("Training file path", allowed_exts=_ALL_EXT)
    fmt  = resolve_format(fmt, path)

    # 3. Column (tabular formats only)
    column = ask_column(fmt)

    # 4. Load
    blank()
    info_line(f"Loading  {path} …")
    try:
        docs = Portal().ingest(path, fmt=fmt, column=column)
    except ImportError as exc:
        err(str(exc)); sys.exit(1)
    except ValueError as exc:
        err(str(exc)); sys.exit(1)

    ok(f"{len(docs):,} documents loaded  ←  {os.path.basename(path)}")
    return docs


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING  (with live graphs)
# ══════════════════════════════════════════════════════════════════════════════

def run_training(
    docs:        list[str],
    cfg:         dict,
    steps:       int,
    tok_mode:    str,
    live_graphs: bool,
) -> tuple:
    """
    Build vocab → build model → train.
    Returns (model, cipher, retriever, forge, optimizer, loss_history).
    """

    # Vocabulary
    section_open("Vocabulary Builder")
    cipher = Cipher(docs, mode=tok_mode)
    blank()
    kv("Vocabulary size",  cipher.vocab_sz)
    kv("Tokeniser mode",   cipher.mode)
    kv("Seal (BOS) id",    cipher.seal)
    kv("Sample tokens",    str(cipher.registry[:16]))
    section_close()

    # Model
    section_open("Model Architecture  (deep transformer)")
    model = Lattice(vocab_sz=cipher.vocab_sz, cfg=cfg)
    blank()
    kv("Transformer blocks",  model.rifts)
    kv("Hidden dimension",    model.depth)
    kv("Attention streams",   model.streams)
    kv("Per-stream channel",  model.channel)
    kv("Context horizon",     f"{model.horizon} tokens")
    kv("Learnable params",    f"{len(model.params):,}")
    section_close()

    # Corpus
    section_open("Corpus Compilation")
    corpus = cipher.make_sequences(docs, model.horizon)
    random.shuffle(corpus)
    blank()
    kv("Training sequences", f"{len(corpus):,}")
    kv("Max sequence length", model.horizon)
    section_close()

    # TF-IDF retriever
    section_open("Building Retrieval Index  (for chat Q&A)")
    retriever = Retriever()
    retriever.fit(docs)
    blank()
    kv("Indexed chunks",   f"{len(retriever.chunks):,}")
    kv("Vocabulary size",  f"{len(retriever.vocab):,}")
    kv("Scope threshold",  retriever.threshold)
    section_close()

    # Live graph windows
    lw: LiveWire | None = None
    if live_graphs:
        section_open("Live Training Graphs  (5 separate windows)")
        blank()
        print(f"  {teal('W1')}  Loss Pulse        {grey('— raw + EMA-smoothed loss')}")
        print(f"  {teal('W2')}  Neural Flow       {grey('— animated: data moving through the network')}")
        print(f"  {teal('W3')}  Token Heatmap     {grey('— softmax probability per vocabulary token')}")
        print(f"  {teal('W4')}  Gradient Health   {grey('— mean |∇| magnitude (log scale)')}")
        print(f"  {teal('W5')}  Live Output       {grey('— text samples generated during training')}")
        blank()
        lw = LiveWire(cipher.registry, cipher, model,
                      update_every=8, sample_every=60,
                      max_vocab_show=min(40, cipher.vocab_sz))
        if lw.open():
            ok("5 windows opened — watch data flow through the network in W2.")
        else:
            warn("Live graphs unavailable (pip install matplotlib).  Continuing without.")
            lw = None
        section_close()

    # Training loop
    section_open("Training Loop")
    blank()
    optimizer    = Apex(model.params)
    forge        = Forge(model, optimizer, corpus)
    actual_steps = min(steps, len(corpus) * 4)
    kv("Planned steps", actual_steps)
    kv("Live graphs",   green("ON  (5 windows)") if lw else yellow("OFF"))
    blank()

    def _term_cb(step, total, loss_val, eta):
        progress(step + 1, total, label=f"loss {loss_val:.4f}  ETA {eta:.0f}s  ")

    def _live_cb(step, total, loss_val, lr, logits, params):
        if lw:
            lw.tick(step, total, loss_val, lr, logits, params)

    loss_hist = forge.anneal(
        actual_steps,
        on_step_cb = _term_cb,
        on_live_cb = _live_cb if lw else None,
    )
    progress_done()

    if lw:
        lw.close()
        info_line("Live graph windows frozen — they stay open alongside the chat.")

    blank()
    ok(f"Training complete  |  final loss {bold(f'{loss_hist[-1]:.4f}')}")
    kv("Loss at step 1",    f"{loss_hist[0]:.4f}")
    kv("Loss at last step", f"{loss_hist[-1]:.4f}")
    kv("Improvement",
       f"{(1 - loss_hist[-1] / max(loss_hist[0], 1e-9)) * 100:.1f}%")
    section_close()

    return model, cipher, retriever, forge, optimizer, loss_hist


# ══════════════════════════════════════════════════════════════════════════════
#  REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_report(model, cipher, loss_hist: list[float], docs: list[str]) -> None:
    blank()
    double_rule()
    print(f"  {teal('NEURYX')}  ·  Training Complete — Launching Chat")
    rule("─")
    kv("Documents trained on", f"{len(docs):,}")
    kv("Vocabulary size",      cipher.vocab_sz)
    kv("Learnable parameters", f"{len(model.params):,}")
    kv("Steps run",            len(loss_hist))
    rule("─")
    kv("Initial loss",  f"{loss_hist[0]:.4f}")
    kv("Final loss",    f"{loss_hist[-1]:.4f}")
    rule("─")
    ratio = loss_hist[-1] / max(loss_hist[0], 1e-9)
    if ratio < 0.65:
        v = green("Strong convergence — model learned well.")
    elif ratio < 0.88:
        v = yellow("Partial convergence — more steps or data may improve quality.")
    else:
        v = red("Weak convergence — try larger dataset or more training steps.")
    print(f"  {v}")
    double_rule()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="neuryx",
        description="Neuryx — Train on your data, then chat with it.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
examples:
  python neuryx.py
  python neuryx.py --train data/sample_names.txt
  python neuryx.py --train data/sample_weather.csv --mode word --steps 800
  python neuryx.py --depth 128 --rifts 6 --no-live
        """,
    )
    parser.add_argument("--train",       metavar="FILE",  help="Training dataset path (skips format menu)")
    parser.add_argument("--steps",       type=int,   default=600,
                        help="Training steps  (default: 600)")
    parser.add_argument("--temperature", type=float, default=0.35,
                        help="Chat generation temperature  (default: 0.35)")
    parser.add_argument("--mode",        default="char",
                        choices=["char", "word", "token"],
                        help="Tokenisation mode  (default: char)")
    parser.add_argument("--depth",       type=int,   default=64,
                        help="Embedding dimension  (default: 64)")
    parser.add_argument("--rifts",       type=int,   default=4,
                        help="Transformer blocks  (default: 4)")
    parser.add_argument("--horizon",     type=int,   default=128,
                        help="Context window  (default: 128)")
    parser.add_argument("--no-live",     action="store_true",
                        help="Disable live training graph windows")
    parser.add_argument("--no-chart",    action="store_true",
                        help="Skip the post-training summary dashboard")
    args = parser.parse_args()

    # Splash
    print(LOGO)
    double_rule()
    print(f"  {bold('Configuration')}  — change anything via flags  (--help)")
    rule("─")
    kv("--depth   hidden dim",    args.depth)
    kv("--rifts   blocks",        args.rifts)
    kv("--horizon context len",   args.horizon)
    kv("--steps   train steps",   args.steps)
    kv("--mode    tokeniser",     args.mode)
    kv("--temperature chat heat", args.temperature)
    kv("--no-live  skip graphs",  args.no_live)
    double_rule()
    blank()
    print(f"  {teal('How it works:')}")
    print(f"  {grey('1.')} Select your file format  {grey('→')}  Enter file path")
    print(f"  {grey('2.')} Watch training live in 5 separate windows")
    print(f"  {grey('3.')} Chat window opens automatically after training")
    blank()

    cfg = {**DEFAULT_CFG, "depth": args.depth, "rifts": args.rifts,
           "horizon": args.horizon, "streams": min(8, args.depth // 8)}

    # Load data
    docs = load_dataset(cli_path=args.train)

    # Train
    heading("Training Phase",
            "Deep transformer  ·  5 live graph windows  ·  TF-IDF retriever")
    model, cipher, retriever, forge, optimizer, loss_hist = run_training(
        docs, cfg, args.steps, args.mode, live_graphs=not args.no_live,
    )

    # Report
    print_report(model, cipher, loss_hist, docs)

    # Optional dashboard
    if not args.no_chart:
        info_line("Saving training dashboard …")
        render_dashboard(
            docs, loss_hist,
            generated=[cipher.decipher(
                forge.infer([cipher.seal], 40, 0.4, cipher.seal))
                for _ in range(10)],
            vocab_registry=cipher.registry,
        )

    # Launch chat
    blank()
    model_info = {
        "params": len(model.params),
        "vocab":  cipher.vocab_sz,
        "docs":   len(docs),
    }
    heading(
        "Chat Phase  — Neuryx is ready to talk",
        "Tkinter window (or terminal fallback) launching now…",
    )
    blank()
    info_line("Opening chat interface…")
    blank()

    launch_chat(
        retriever    = retriever,
        forge        = forge,
        cipher       = cipher,
        model        = model,
        model_info   = model_info,
        temperature  = args.temperature,
    )

    blank()
    print(f"  {teal('Neuryx')} session ended.  {grey('Goodbye.')}")
    blank()


if __name__ == "__main__":
    main()
