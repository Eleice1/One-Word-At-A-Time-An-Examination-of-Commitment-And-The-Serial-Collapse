#!/usr/bin/env python3
"""
logit_lens.py

Reads what Qwen's middle layers were leaning toward during a qwen_meditation.py run.

Each saved hidden state is passed through the model's final normalization and
output layer, as if that layer were the last one. The result is the token that
layer "would say" next. Comparing layers shows how early in the stack the model
settles on what it actually writes, and whether anything (like the mantra)
shows up internally before it appears in the text.

    python logit_lens.py                          # newest run in ./runs
    python logit_lens.py runs\\run_20261027_0215   # a specific run
    python logit_lens.py --topk 5                 # keep the 5 likeliest tokens per layer

Only the layers qwen_meditation.py saved can be read (by default 0, 7, 14, 21, 28;
change that with its --save-layers option). Layer 0 is the raw token embedding,
so its predictions are mostly noise; it is included as a baseline.

Outputs, written into the run folder:
    lens.txt            every token: what was written next, and what each layer predicted
    lens.csv            the same as a table
    lens_summary.json   how often each layer already matched the real next token, and
                        how strongly each layer anticipated each mantra
    lens_agreement.png  agreement by layer, countdown vs mantra
    lens_mantra_N.png   the stretch around mantra N, layer by layer
"""
import sys as _sys
_sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import argparse
import os
import csv
import json
import re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Local folder or Hugging Face ID. Set QWEN_MODEL to use a local copy.
MODEL_DIR = os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
MANTRA_RE = re.compile(re.escape("i am here, here is where i am"), re.I)


def show(t):
    return t.replace("\n", "⏎").replace("\t", "⇥")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", help="run folder (default: newest in ./runs)")
    ap.add_argument("--model", default=MODEL_DIR)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    ap.add_argument("--window", type=int, default=30,
                    help="tokens before each mantra to include in its plot")
    args = ap.parse_args()

    if args.run:
        run = Path(args.run)
    else:
        runs = sorted((Path(__file__).resolve().parent / "runs").glob("run_*"))
        if not runs:
            raise SystemExit("No runs found; pass a run folder.")
        run = runs[-1]
    print(f"Reading {run}")

    m = np.load(run / "metrics.npz")
    hidden = m["hidden"]                     # (saved layers, tokens, dims), float16
    layers = [int(x) for x in m["save_layers"]]
    ids = m["token_ids"]
    pieces = json.loads((run / "tokens.json").read_text(encoding="utf-8"))
    n = len(ids)
    text = "".join(pieces)

    print(f"Loading {args.model} ({args.dtype})...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype))
    model.eval()
    dt = getattr(torch, args.dtype)
    top_layer = model.config.num_hidden_layers

    # Region of each token (by character position) so we can split the stats.
    starts = np.cumsum([0] + [len(p) for p in pieces[:-1]])
    in_mantra = np.zeros(n, bool)
    mantra_first = []          # token index where each mantra starts
    for mm in MANTRA_RE.finditer(text):
        a, b = mm.span()
        first = int(np.searchsorted(starts, a, side="right")) - 1
        last = int(np.searchsorted(starts, b, side="left"))
        in_mantra[first:last] = True
        mantra_first.append(first)

    # ---- Run each saved layer through the output head -----------------------
    S, k = len(layers), args.topk
    top_ids = np.zeros((S, n, k), np.int64)
    top_p = np.zeros((S, n, k), np.float32)
    p_actual = np.full((S, n), np.nan, np.float32)   # prob. given to the real next token
    rank_actual = np.full((S, n), -1, np.int64)      # its rank (0 = top choice)
    nxt = torch.from_numpy(ids[1:].astype(np.int64))
    with torch.inference_mode():
        for si, layer in enumerate(layers):
            print(f"  layer {layer}")
            for s in range(0, n, 512):
                e = min(s + 512, n)
                x = torch.from_numpy(hidden[si, s:e].astype(np.float32)).to(dt)
                if layer != top_layer:            # the top layer's saved state is already normalized
                    x = model.model.norm(x)
                logits = model.lm_head(x).float()
                lp = torch.log_softmax(logits, -1)
                v, i = lp.topk(k, -1)
                top_ids[si, s:e], top_p[si, s:e] = i.numpy(), v.exp().numpy()
                rows = torch.arange(s, min(e, n - 1))
                if len(rows):
                    t = nxt[rows]
                    la = logits[rows - s].gather(-1, t[:, None])
                    p_actual[si, rows.numpy()] = lp[rows - s].gather(-1, t[:, None]).squeeze(-1).exp().numpy()
                    rank_actual[si, rows.numpy()] = (logits[rows - s] > la).sum(-1).numpy()

    def dec(i):
        return show(tok.decode([int(i)]))

    # ---- Text and CSV -------------------------------------------------------
    with open(run / "lens.txt", "w", encoding="utf-8") as fh:
        fh.write("Each line: the token just written, the token actually written next, and what each\n"
                 "layer predicted would come next (top choice and its probability).\n\n")
        for j in range(n):
            nx = dec(ids[j + 1]) if j + 1 < n else "(end)"
            region = "mantra" if (j + 1 < n and in_mantra[j + 1]) else "count"
            cells = [f"L{l}: {dec(top_ids[si, j, 0])!s:>6} {top_p[si, j, 0]:.2f}" for si, l in enumerate(layers)]
            fh.write(f"{j:6d} {region:6s} wrote {dec(ids[j])!s:>6}  next {nx!s:>6} | " + " | ".join(cells) + "\n")

    with open(run / "lens.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        head = ["index", "token", "next_token", "next_is_mantra"]
        for l in layers:
            head += [f"L{l}_top1", f"L{l}_top1_prob", f"L{l}_prob_of_next", f"L{l}_rank_of_next"]
        w.writerow(head)
        for j in range(n):
            row = [j, dec(ids[j]), dec(ids[j + 1]) if j + 1 < n else "", bool(j + 1 < n and in_mantra[j + 1])]
            for si in range(S):
                row += [dec(top_ids[si, j, 0]), round(float(top_p[si, j, 0]), 4),
                        "" if np.isnan(p_actual[si, j]) else round(float(p_actual[si, j]), 4),
                        int(rank_actual[si, j])]
            w.writerow(row)

    # ---- Summary ------------------------------------------------------------
    valid = np.arange(n - 1)
    next_mantra = in_mantra[1:]
    agree = rank_actual[:, :n - 1] == 0               # (S, n-1)

    def rate(mask):
        return {f"layer {l}": round(float(agree[si, mask].mean()), 4) if mask.any() else None
                for si, l in enumerate(layers)}

    # Earliest saved layer from which the top choice is right and stays right.
    settled = np.full(n - 1, np.nan)
    for j in valid:
        for si in range(S):
            if agree[si:, j].all():
                settled[j] = layers[si]
                break

    anticipation = []
    for q, f0 in enumerate(mantra_first, 1):
        target = int(ids[f0])
        # probability each layer gives the mantra's first token, at each of the 4 positions before it
        probs = {}
        with torch.inference_mode():
            for si, l in enumerate(layers):
                js = [f0 - b for b in (4, 3, 2, 1) if f0 - b >= 0]
                x = torch.from_numpy(hidden[si, js].astype(np.float32)).to(dt)
                if l != top_layer:
                    x = model.model.norm(x)
                p = torch.softmax(model.lm_head(x).float(), -1)[:, target]
                probs[f"layer {l}"] = [float(f"{v:.3g}") for v in p]
        entry = {"mantra": q, "first_token": dec(target),
                 "positions_before": [f"after {dec(ids[f0 - b])!r}" for b in (4, 3, 2, 1) if f0 - b >= 0],
                 "prob_of_mantra_start_by_layer": probs}
        anticipation.append(entry)

    summary = {
        "run": str(run),
        "layers_read": layers,
        "note": "agreement = share of positions where that layer's top prediction was the token actually written next",
        "agreement_countdown": rate(~next_mantra),
        "agreement_mantra": rate(next_mantra),
        "mean_settling_layer_countdown": round(float(np.nanmean(settled[~next_mantra])), 2) if (~next_mantra).any() else None,
        "mean_settling_layer_mantra": round(float(np.nanmean(settled[next_mantra])), 2) if next_mantra.any() else None,
        "mantra_anticipation": anticipation,
    }
    (run / "lens_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- Plots --------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None
        print("matplotlib not installed; skipping plots.")

    if plt is not None:
        fig, ax = plt.subplots(figsize=(7, 4))
        for name, mask in (("countdown", ~next_mantra), ("mantra", next_mantra)):
            if mask.any():
                ax.plot(layers, [agree[si, mask].mean() for si in range(S)], marker="o", label=name)
        ax.set(xlabel="layer", ylabel="top prediction = real next token",
               title="How early each layer already 'knows' the next token", ylim=(0, 1.02))
        ax.set_xticks(layers)
        ax.legend()
        fig.tight_layout()
        fig.savefig(run / "lens_agreement.png", dpi=130)
        plt.close(fig)

        for q, f0 in enumerate(mantra_first, 1):
            a, b = max(f0 - args.window, 0), min(f0 + 12, n - 1)
            if b <= a:
                continue
            fig, ax = plt.subplots(figsize=(14, 3.2))
            im = ax.imshow(p_actual[:, a:b], aspect="auto", cmap="magma", vmin=0, vmax=1)
            ax.set_yticks(range(S), [f"L{l}" for l in layers])
            ax.set_xticks(range(b - a), [dec(ids[j + 1]) for j in range(a, b)], fontsize=7, rotation=90)
            ax.axvline(f0 - 1 - a + 0.5, color="cyan", lw=1)
            ax.set_title(f"Mantra {q}: probability each layer gave the token actually written next "
                         f"(cyan line = mantra begins)")
            fig.colorbar(im, ax=ax, fraction=0.02)
            fig.tight_layout()
            fig.savefig(run / f"lens_mantra_{q}.png", dpi=130)
            plt.close(fig)

    print(json.dumps({k: v for k, v in summary.items() if k != "mantra_anticipation"}, indent=2))
    print(f"\nWrote lens.txt, lens.csv, lens_summary.json and plots to {run}")


if __name__ == "__main__":
    main()
