# Neuryx
**Train a deep transformer on your data — then chat with it.**

```
  ███╗   ██╗███████╗██╗   ██╗██████╗ ██╗   ██╗██╗  ██╗
  General-Purpose Neural AI  ·  v2.0.0
```

No cloud. No API key. Pure Python + optional matplotlib.

---

## Session Flow

```
  1. Choose input format  [1]TXT  [2]CSV  [3]Excel  [4]JSON  [5]TSV  [6]Auto
  2. Enter file path
  ↓  Training starts — 5 LIVE windows:
       W1 Loss Pulse   W2 Neural Flow (animated!)   W3 Token Heatmap
       W4 Gradient Health              W5 Live Output
  ↓  Chat window opens automatically
  ↓  Ask questions from your data  |  Out-of-scope queries are refused
```

## Quick Start

```bash
pip install matplotlib openpyxl     # optional — for graphs
sudo apt-get install python3-tk     # optional — for chat GUI (Linux)
python neuryx.py
```

## Sample Datasets (included)

| File | Format | Use |
|------|--------|-----|
| `data/sample_names.txt`     | TXT  | Name generation |
| `data/sample_weather.csv`   | CSV  | Weather Q&A |
| `data/sample_logs.json`     | JSON | Log event queries |
| `data/sample_events.tsv`    | TSV  | UI action queries |
| `data/sample_text.txt`      | TXT  | Phrase Q&A |
| `data/sample_sequences.xlsx`| XLSX | Sequence data |

## CLI Flags

```
python neuryx.py --train FILE --steps 600 --mode char
                 --depth 64 --rifts 4 --horizon 128
                 --temperature 0.35 --no-live --no-chart
```

## Architecture

- **core/flux.py** — scalar autograd engine (no ML deps)
- **core/lattice.py** — 4-block causal transformer (depth=64, streams=8)
- **core/retriever.py** — TF-IDF retrieval + out-of-scope detection
- **render/livewire.py** — 5 separate live matplotlib windows
- **render/chatwin.py** — Tkinter chat window (terminal fallback)

## Requirements

```
Python >= 3.10
matplotlib  (pip install matplotlib)     # graphs
openpyxl    (pip install openpyxl)       # Excel support
tkinter     (apt-get install python3-tk) # chat GUI
```

MIT License.
