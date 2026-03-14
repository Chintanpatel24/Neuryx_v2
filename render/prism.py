"""
render/prism.py
Post-training summary dashboard — one figure, 6 panels.
Saved to neuryx_dashboard.png after training + inference.
"""
from __future__ import annotations


def render_dashboard(
    train_docs:     list[str],
    loss_chronicle: list[float],
    generated:      list[str],
    vocab_registry: list[str],
    output_path:    str = "neuryx_dashboard.png",
) -> None:
    try:
        import matplotlib
        import matplotlib.pyplot   as plt
        import matplotlib.gridspec as gridspec
        from   collections         import Counter
    except ImportError:
        print("[prism] matplotlib not installed — skipping dashboard.")
        return

    BG = "#09091a"; PAN = "#0f0f28"; TEAL = "#00e5c8"; VIO = "#8b5cf6"
    CORAL = "#f87171"; AMBER = "#fbbf24"; FG = "#e2e8f0"; DIM = "#1e1e3f"
    GREEN = "#34d399"; PINK = "#f472b6"

    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": PAN,
        "axes.edgecolor": DIM, "axes.labelcolor": FG,
        "axes.titlecolor": FG, "xtick.color": FG, "ytick.color": FG,
        "text.color": FG, "grid.color": DIM, "font.family": "monospace",
    })

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("NEURYX  ·  Training Session Dashboard",
                 fontsize=14, fontweight="bold", color=TEAL,
                 y=0.998, x=0.01, ha="left")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.35)

    def style(ax, title):
        ax.set_title(f"  {title}", fontweight="bold", fontsize=9.5, loc="left", pad=6)
        ax.grid(True, alpha=0.15)

    # 1 Loss
    ax1 = fig.add_subplot(gs[0, 0])
    steps = list(range(1, len(loss_chronicle) + 1))
    ax1.plot(steps, loss_chronicle, color=VIO, lw=0.7, alpha=0.4)
    w = max(1, len(steps) // 15)
    sm = [sum(loss_chronicle[max(0,i-w):i+1])/len(loss_chronicle[max(0,i-w):i+1])
          for i in range(len(loss_chronicle))]
    ax1.plot(steps, sm, color=TEAL, lw=2)
    ax1.fill_between(steps, sm, alpha=0.08, color=TEAL)
    ax1.set_xlabel("Step"); ax1.set_ylabel("Loss")
    style(ax1, "[1] Training Loss")

    # 2 Doc lengths
    ax2 = fig.add_subplot(gs[0, 1])
    lens = [len(d) for d in train_docs]
    ax2.hist(lens, bins=min(40, len(set(lens))), color=TEAL, alpha=0.78, edgecolor=BG)
    ax2.axvline(sum(lens)/max(len(lens),1), color=AMBER, lw=1.5, ls="--",
                label=f"mean={sum(lens)/max(len(lens),1):.0f}")
    ax2.set_xlabel("Length (chars)"); ax2.set_ylabel("Count"); ax2.legend(fontsize=8)
    style(ax2, "[2] Training Doc Lengths")

    # 3 Top symbols
    ax3 = fig.add_subplot(gs[0, 2])
    counts = Counter("".join(train_docs)).most_common(20)
    if counts:
        labels, vals = zip(*counts)
        labels = [repr(c)[1:-1] for c in labels]
        ax3.barh(range(len(vals)), list(vals), color=VIO, alpha=0.82, edgecolor=BG)
        ax3.set_yticks(range(len(labels))); ax3.set_yticklabels(labels, fontsize=8)
        ax3.set_xlabel("Frequency")
    style(ax3, "[3] Symbol Frequencies")

    # 4 Convergence halves
    ax4 = fig.add_subplot(gs[1, 0])
    half = len(sm) // 2
    ax4.plot(steps[:half], sm[:half], color=CORAL, lw=1.5, label="first half")
    ax4.plot(steps[half:], sm[half:], color=GREEN, lw=1.5, label="second half")
    ax4.fill_between(steps[:half], sm[:half], alpha=0.10, color=CORAL)
    ax4.fill_between(steps[half:], sm[half:], alpha=0.10, color=GREEN)
    ax4.set_xlabel("Step"); ax4.set_ylabel("Smoothed Loss"); ax4.legend(fontsize=8)
    style(ax4, "[4] Convergence (first vs second half)")

    # 5 Output lengths
    ax5 = fig.add_subplot(gs[1, 1])
    gl = [len(g) for g in generated] if generated else [0]
    ax5.hist(gl, bins=min(20, max(len(set(gl)), 2)),
             color=AMBER, alpha=0.82, edgecolor=BG)
    ax5.set_xlabel("Length (chars)"); ax5.set_ylabel("Count")
    style(ax5, "[5] Generated Output Lengths")

    # 6 Vocab coverage
    ax6 = fig.add_subplot(gs[1, 2])
    n_vocab = len(vocab_registry)
    n_used  = len(set("".join(train_docs)) & set(vocab_registry))
    ax6.bar(["Used", "Unseen"], [n_used, n_vocab - n_used],
            color=[GREEN, CORAL], alpha=0.85, edgecolor=BG, width=0.5)
    for i, v in enumerate([n_used, n_vocab - n_used]):
        ax6.text(i, v + 0.3, str(v), ha="center", fontsize=10, fontweight="bold")
    style(ax6, "[6] Vocabulary Coverage")

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  Dashboard saved → {output_path}")
    try:
        plt.show(block=False)
    except Exception:
        pass
