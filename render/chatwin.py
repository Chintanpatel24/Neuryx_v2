"""
render/chatwin.py
Neuryx Chat Window — conversational interface backed by the trained model.

Primary:   Tkinter GUI  (dark theme, message bubbles, typing indicator)
Fallback:  Rich ANSI terminal chat  (if tkinter is unavailable)

The window is launched after training completes. Every user message is:
  1. Checked against the TF-IDF retriever for scope
  2. If in scope  → retrieve best passage + model continuation
  3. If out scope → polite refusal message

No external API is used. All responses come from the trained Lattice model
and the indexed training corpus.
"""

from __future__ import annotations
import threading
import time
import os
import sys

# ── Probe tkinter ─────────────────────────────────────────────────────────────
_TK_OK = False
try:
    import tkinter as tk
    import tkinter.font as tkfont
    import tkinter.scrolledtext as tkst
    _TK_OK = True
except ImportError:
    pass

# ── Palette (mirrors livewire.py) ─────────────────────────────────────────────
_BG     = "#09091a"
_PAN    = "#0f0f28"
_TEAL   = "#00e5c8"
_VIO    = "#8b5cf6"
_GREEN  = "#34d399"
_AMBER  = "#fbbf24"
_CORAL  = "#f87171"
_FG     = "#e2e8f0"
_DIM    = "#4a4a6a"
_INPUT  = "#13132e"
_BORDER = "#1e1e3f"

OUT_OF_SCOPE_MSG = (
    "⚠  That question falls outside my training data.\n"
    "   I can only answer based on what I was trained on.\n"
    "   Try asking something more closely related to your dataset."
)

# ══════════════════════════════════════════════════════════════════════════════
#  Shared response generator  (used by both GUI and terminal modes)
# ══════════════════════════════════════════════════════════════════════════════

class _Engine:
    """
    Wraps the retriever + forge + cipher into a single .respond(query) call.
    Returns a string (the model's reply) or the out-of-scope message.
    """

    def __init__(self, retriever, forge, cipher, model, temperature: float = 0.35):
        self.retriever   = retriever
        self.forge       = forge
        self.cipher      = cipher
        self.model       = model
        self.temperature = temperature

    def respond(self, query: str) -> str:
        query = query.strip()
        if not query:
            return ""

        result = self.retriever.answer(
            query, self.forge, self.cipher, self.model,
            temperature=self.temperature,
        )

        if result is None:
            return OUT_OF_SCOPE_MSG

        answer       = result["answer"]
        continuation = result["continuation"]
        score        = result["score"]

        parts = [answer]
        if continuation and continuation not in answer:
            parts.append(continuation)

        response = " ".join(parts).strip()

        # Add confidence hint for borderline matches
        if score < 0.18:
            response += f"\n\n[Confidence: {score:.2f} — answer based on closest matching passage]"

        return response


# ══════════════════════════════════════════════════════════════════════════════
#  Tkinter Chat Window
# ══════════════════════════════════════════════════════════════════════════════

class _TkChat:
    """Full Tkinter chat window with dark theme and threaded response generation."""

    _USER_TAG = "user_msg"
    _BOT_TAG  = "bot_msg"
    _SYS_TAG  = "sys_msg"
    _DIM_TAG  = "dim_msg"
    _WARN_TAG = "warn_msg"

    def __init__(self, engine: _Engine, model_info: dict):
        self.engine     = engine
        self.model_info = model_info
        self._typing    = False

        self.root = tk.Tk()
        self.root.title("Neuryx — AI Chat")
        self.root.configure(bg=_BG)
        self.root.geometry("860x680")
        self.root.minsize(640, 480)

        self._build_ui()
        self._post_system(
            f"Neuryx AI  ·  v2.0  ·  Model trained on your data\n"
            f"Params: {model_info.get('params','?'):,}  |  "
            f"Vocab: {model_info.get('vocab','?')}  |  "
            f"Documents: {model_info.get('docs','?')}\n"
            f"Type a question and press Enter or click Send."
        )

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # Title bar
        title_bar = tk.Frame(self.root, bg=_PAN, height=46)
        title_bar.pack(fill="x", side="top")
        title_bar.pack_propagate(False)
        tk.Label(title_bar,
                 text="  ◈  NEURYX AI CHAT",
                 bg=_PAN, fg=_TEAL,
                 font=("Courier New", 13, "bold")).pack(side="left", padx=10, pady=10)
        tk.Label(title_bar,
                 text="Powered by your training data  ·  No external API",
                 bg=_PAN, fg=_DIM,
                 font=("Courier New", 8)).pack(side="right", padx=12)

        # Chat history (scrolled text)
        chat_frame = tk.Frame(self.root, bg=_BORDER, bd=0)
        chat_frame.pack(fill="both", expand=True, padx=8, pady=(6, 0))

        self.chat_box = tk.Text(
            chat_frame,
            bg=_BG, fg=_FG,
            font=("Courier New", 10),
            wrap="word",
            state="disabled",
            bd=0, relief="flat",
            padx=12, pady=8,
            cursor="arrow",
            spacing3=4,
        )
        self.chat_box.pack(fill="both", expand=True, side="left")

        scroll = tk.Scrollbar(chat_frame, command=self.chat_box.yview,
                              bg=_PAN, troughcolor=_BG, activebackground=_TEAL,
                              bd=0, relief="flat", width=8)
        scroll.pack(fill="y", side="right")
        self.chat_box.config(yscrollcommand=scroll.set)

        # Tag styles
        try:
            mono = tkfont.Font(family="Courier New", size=10)
            bold = tkfont.Font(family="Courier New", size=10, weight="bold")
        except Exception:
            mono = bold = None

        self.chat_box.tag_config(self._USER_TAG, foreground=_TEAL,
                                 font=bold, lmargin1=12, lmargin2=24)
        self.chat_box.tag_config(self._BOT_TAG,  foreground=_FG,
                                 font=mono, lmargin1=12, lmargin2=24)
        self.chat_box.tag_config(self._SYS_TAG,  foreground=_DIM,
                                 font=mono, lmargin1=8)
        self.chat_box.tag_config(self._WARN_TAG, foreground=_AMBER,
                                 font=mono, lmargin1=12, lmargin2=24)
        self.chat_box.tag_config(self._DIM_TAG,  foreground=_DIM,
                                 font=mono, lmargin1=8)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status = tk.Label(self.root, textvariable=self.status_var,
                          bg=_PAN, fg=_DIM,
                          font=("Courier New", 8), anchor="w", padx=10)
        status.pack(fill="x", side="bottom")

        # Input row
        input_row = tk.Frame(self.root, bg=_BG)
        input_row.pack(fill="x", side="bottom", padx=8, pady=6)

        self.entry = tk.Entry(
            input_row,
            bg=_INPUT, fg=_FG,
            insertbackground=_TEAL,
            font=("Courier New", 11),
            bd=0, relief="flat",
            highlightthickness=1,
            highlightcolor=_TEAL,
            highlightbackground=_BORDER,
        )
        self.entry.pack(fill="x", side="left", expand=True, ipady=7, padx=(0, 8))
        self.entry.bind("<Return>", lambda e: self._on_send())
        self.entry.focus_set()

        self.send_btn = tk.Button(
            input_row,
            text="Send  ▶",
            command=self._on_send,
            bg=_TEAL, fg=_BG,
            font=("Courier New", 10, "bold"),
            activebackground=_GREEN,
            activeforeground=_BG,
            bd=0, relief="flat",
            cursor="hand2",
            padx=14, pady=7,
        )
        self.send_btn.pack(side="right")

    # ── Message posting ───────────────────────────────────────────────────────

    def _post(self, text: str, tag: str, prefix: str = ""):
        self.chat_box.config(state="normal")
        if prefix:
            self.chat_box.insert("end", prefix + "\n", self._DIM_TAG)
        self.chat_box.insert("end", text + "\n\n", tag)
        self.chat_box.config(state="disabled")
        self.chat_box.see("end")

    def _post_system(self, text: str):
        self._post(text, self._SYS_TAG, "─" * 64)

    def _post_user(self, text: str):
        self._post(f"You  ›  {text}", self._USER_TAG)

    def _post_bot(self, text: str):
        tag = self._WARN_TAG if "⚠" in text else self._BOT_TAG
        self._post(f"Neuryx  ›  {text}", tag)

    # ── Send logic ────────────────────────────────────────────────────────────

    def _on_send(self):
        if self._typing:
            return
        query = self.entry.get().strip()
        if not query:
            return
        self.entry.delete(0, "end")
        self._post_user(query)
        self._typing = True
        self.send_btn.config(state="disabled", text="…")
        self.status_var.set("Neuryx is thinking…")
        threading.Thread(target=self._generate, args=(query,), daemon=True).start()

    def _generate(self, query: str):
        try:
            reply = self.engine.respond(query)
        except Exception as exc:
            reply = f"[Engine error: {exc}]"
        self.root.after(0, lambda: self._on_reply(reply))

    def _on_reply(self, reply: str):
        self._post_bot(reply)
        self._typing = False
        self.send_btn.config(state="normal", text="Send  ▶")
        self.status_var.set("Ready")

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(self):
        self.root.mainloop()


# ══════════════════════════════════════════════════════════════════════════════
#  Terminal Chat Fallback
# ══════════════════════════════════════════════════════════════════════════════

class _TermChat:
    """ANSI terminal chat — used when tkinter is not installed."""

    RST  = "\033[0m"
    TEAL = "\033[96m\033[1m"
    VIO  = "\033[95m"
    DIM  = "\033[90m"
    YEL  = "\033[93m"
    WHT  = "\033[97m\033[1m"
    GRN  = "\033[92m"

    def __init__(self, engine: _Engine, model_info: dict):
        self.engine = engine
        self.info   = model_info

    def run(self):
        r = self.RST; t = self.TEAL; d = self.DIM; w = self.WHT
        print(f"\n{t}{'═'*62}{r}")
        print(f"{t}  NEURYX AI CHAT  ·  Terminal Mode{r}")
        print(f"{d}  Params: {self.info.get('params','?'):,}  |  "
              f"Vocab: {self.info.get('vocab','?')}  |  "
              f"Docs: {self.info.get('docs','?')}{r}")
        print(f"{t}{'═'*62}{r}")
        print(f"{d}  Type your question and press Enter.  'exit' to quit.{r}\n")

        while True:
            try:
                query = input(f"  {t}You ›{r} ").strip()
            except (EOFError, KeyboardInterrupt):
                print(f"\n{d}  Session ended.{r}\n")
                break
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q", ":q"):
                print(f"\n{d}  Goodbye.{r}\n")
                break

            print(f"\n  {d}Neuryx is thinking…{r}", end="\r")
            reply = self.engine.respond(query)
            print(" " * 40, end="\r")   # clear "thinking" line

            if "⚠" in reply:
                print(f"\n  {self.YEL}Neuryx › {self.RST}{self.YEL}{reply}{r}\n")
            else:
                print(f"\n  {self.VIO}Neuryx ›{r} {w}{reply}{r}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  Public entry-point
# ══════════════════════════════════════════════════════════════════════════════

def launch_chat(
    retriever,
    forge,
    cipher,
    model,
    model_info:  dict,
    temperature: float = 0.35,
) -> None:
    """
    Open the Neuryx chat interface after training is complete.

    Uses Tkinter GUI if available, otherwise falls back to the terminal.

    Parameters
    ----------
    retriever   : Retriever   — fitted TF-IDF index
    forge       : Forge       — trained model inference wrapper
    cipher      : Cipher      — vocabulary encoder/decoder
    model       : Lattice     — the trained transformer
    model_info  : dict        — {params, vocab, docs} for display
    temperature : float       — sampling temperature for generation
    """
    engine = _Engine(retriever, forge, cipher, model, temperature)

    if _TK_OK:
        try:
            win = _TkChat(engine, model_info)
            win.run()
            return
        except Exception as exc:
            print(f"[chatwin] Tkinter error ({exc}) — falling back to terminal.")

    # Terminal fallback
    _TermChat(engine, model_info).run()
