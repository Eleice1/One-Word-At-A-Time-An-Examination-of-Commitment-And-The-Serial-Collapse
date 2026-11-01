#!/usr/bin/env python3
"""
ablate_test.py

The control for collapse_test.py. Here the collapse stays intact (greedy: the top
token is committed and fed back), and instead parts of the network are silenced.
If silencing ordinary layers damages Qwen in the same way removing the collapse
did, the soft-arm effect isn't special to the collapse; if the damage looks
different, it is.

Silencing works by zeroing a component's output. Every layer adds its result
onto a running "residual stream", so zeroing it means the stream passes through
that part untouched, as if it weren't there. Silencing is active for the whole
run, including while Qwen reads the prompt.

Conditions (pick with --conditions; default runs all):
  intact         no silencing, normal collapse (baseline)
  no_collapse    intact network, collapse removed (collapse_test's "soft"), for side-by-side comparison
  skip_early     layers 2-5 silenced
  skip_mid       layers 12-15 silenced
  skip_late      layers 22-25 silenced
  skip_mid_wide  layers 10-17 silenced (a bigger dose)
  attn_mid       only the attention in layers 12-15 silenced (cuts reading earlier tokens)
  mlp_mid        only the feed-forward part of layers 12-15 silenced
  attn_all_but_first  attention silenced in every layer except layer 0 and 1:
                 the model can barely see anything it has already written

Qwen2.5-1.5B has 28 layers, numbered 0-27. Add your own with --custom, e.g.
    --custom "my_cut=attn:8-11,mlp:20"     (parts: layer, attn, mlp)

    python ablate_test.py                          # lake paragraph, all conditions
    python ablate_test.py --conditions intact,skip_mid,no_collapse
    python ablate_test.py --task countdown

Needs collapse_test.py in the same folder. Use float32 (the default).

Outputs go to ./runs/ablate_<task>_<timestamp>/ next to this script:
    <condition>.txt, summary.json, metrics.npz, ablation_comparison.png
"""
import sys as _sys
_sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from collapse_test import (MODEL_DIR, OPEN_PROMPT, build_prompt, cos, run_condition, score_open,
                           score_text)

PRESETS = {
    "intact": [],
    "no_collapse": [],
    "skip_early": [("layer", range(2, 6))],
    "skip_mid": [("layer", range(12, 16))],
    "skip_late": [("layer", range(22, 26))],
    "skip_mid_wide": [("layer", range(10, 18))],
    "attn_mid": [("attn", range(12, 16))],
    "mlp_mid": [("mlp", range(12, 16))],
    "attn_all_but_first": [("attn", range(2, 28))],
}


def parse_custom(spec):
    """'name=attn:8-11,mlp:20' -> (name, [("attn", range(8,12)), ("mlp", range(20,21))])"""
    name, _, body = spec.partition("=")
    cuts = []
    for part in body.split(","):
        kind, _, rng = part.strip().partition(":")
        a, _, b = rng.partition("-")
        cuts.append((kind.strip(), range(int(a), int(b or a) + 1)))
    return name.strip(), cuts


def _zero_output(module, inputs, output):
    if isinstance(output, tuple):
        return (torch.zeros_like(output[0]),) + tuple(output[1:])
    return torch.zeros_like(output)


def apply_cuts(model, cuts):
    """Register hooks that zero the chosen parts. Returns the hook handles."""
    handles = []
    n_layers = len(model.model.layers)
    for kind, rng in cuts:
        for i in rng:
            if not 0 <= i < n_layers:
                raise SystemExit(f"Layer {i} doesn't exist (model has layers 0-{n_layers - 1}).")
            layer = model.model.layers[i]
            parts = {"layer": [layer.self_attn, layer.mlp], "attn": [layer.self_attn], "mlp": [layer.mlp]}[kind]
            for p in parts:
                handles.append(p.register_forward_hook(_zero_output))
    return handles


def describe(cuts):
    return ", ".join(f"{k} {r.start}-{r.stop - 1}" if len(r) > 1 else f"{k} {r.start}" for k, r in cuts) or "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_DIR)
    ap.add_argument("--task", choices=["open", "countdown"], default="open")
    ap.add_argument("--prompt", default=None, help="custom prompt for --task open")
    ap.add_argument("--start", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=160)
    ap.add_argument("--conditions", default=",".join(PRESETS))
    ap.add_argument("--custom", action="append", default=[], help='e.g. "my_cut=attn:8-11,mlp:20"')
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    ap.add_argument("--layers", default="0,7,14,21,28", help="hidden-state layers kept for comparison")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    conds = {}
    for c in [c.strip() for c in args.conditions.split(",") if c.strip()]:
        if c not in PRESETS:
            raise SystemExit(f"Unknown condition {c!r}. Choose from {list(PRESETS)} or use --custom.")
        conds[c] = PRESETS[c]
    for spec in args.custom:
        name, cuts = parse_custom(spec)
        conds[name] = cuts

    out_dir = Path(args.out) if args.out else (
        Path(__file__).resolve().parent / "runs" / time.strftime(f"ablate_{args.task}_%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model} ({args.dtype})...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype))
    model.eval()
    L = model.config.num_hidden_layers
    layers = [int(x) for x in args.layers.split(",") if 0 <= int(x) <= L]
    mid = layers.index(min(layers, key=lambda l: abs(l - L // 2)))

    user_msg = build_prompt(args.start) if args.task == "countdown" else (args.prompt or OPEN_PROMPT)
    prompt_text = tok.apply_chat_template([{"role": "user", "content": user_msg}],
                                          tokenize=False, add_generation_prompt=True)
    prompt_ids = tok(prompt_text, add_special_tokens=False, return_tensors="pt").input_ids

    results = {}
    for name, cuts in conds.items():
        t0 = time.time()
        print(f"\n===== {name}  (silenced: {describe(cuts)}) =====")
        handles = apply_cuts(model, cuts)
        try:
            r = run_condition("soft" if name == "no_collapse" else "greedy",
                              model, tok, prompt_ids, args, layers, torch.Generator().manual_seed(0))
        finally:
            for h in handles:
                h.remove()
        r["cuts"] = describe(cuts)
        results[name] = r
        (out_dir / f"{name}.txt").write_text(r["text"], encoding="utf-8")
        preview = r["text"].replace("\n", " ")
        print(f"{len(r['entropy'])} steps in {time.time() - t0:.0f}s, {r['stop']}")
        print(f"  {preview[:200]}{' ...' if len(preview) > 200 else ''}")

    ref = results.get("intact")
    summary = {"task": args.task, "prompt": user_msg, "dtype": args.dtype, "layers_kept": layers,
               "conditions": {}}
    for name, r in results.items():
        s = {"silenced": r["cuts"], "collapse": "removed (soft blend)" if name == "no_collapse" else "intact",
             "how_it_ended": r["stop"], "steps": int(len(r["entropy"]))}
        s.update(score_text(r["text"], args.start) if args.task == "countdown" else score_open(r["text"]))
        s["mean_entropy"] = round(float(r["entropy"].mean()), 4)
        s["mean_top1_prob"] = round(float(r["top1_prob"].mean()), 4)
        if ref is not None and name != "intact":
            n = min(len(ref["token_ids"]), len(r["token_ids"]))
            diff = np.nonzero(ref["token_ids"][:n] != r["token_ids"][:n])[0]
            s["first_step_differing_from_intact"] = int(diff[0]) if len(diff) else None
            sim = cos(r["hidden"][:n], ref["hidden"][:n])
            s["similarity_to_intact_by_layer"] = {f"layer {l}": round(float(sim[:, i].mean()), 4)
                                                  for i, l in enumerate(layers)}
            r["sim_to_intact"] = sim
        summary["conditions"][name] = s
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    save = {}
    for name, r in results.items():
        for k in ("entropy", "p_end", "top1_prob", "token_ids"):
            save[f"{name}_{k}"] = r[k]
        save[f"{name}_hidden"] = r["hidden"].astype(np.float16)
        if "sim_to_intact" in r:
            save[f"{name}_sim_to_intact"] = r["sim_to_intact"]
    np.savez_compressed(out_dir / "metrics.npz", layers=np.array(layers), **save)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None
        print("matplotlib not installed; skipping plots.")
    if plt is not None:
        fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
        for name, r in results.items():
            style = "--" if name == "no_collapse" else "-"
            axes[0].plot(r["entropy"], style, lw=0.8, label=name)
            if "sim_to_intact" in r:
                axes[1].plot(r["sim_to_intact"][:, mid], style, lw=0.8, label=name)
        axes[0].set(ylabel="entropy (nats)", title="Uncertainty at each step (dashed = collapse removed)")
        axes[1].set(ylabel="cosine similarity", xlabel="step",
                    title=f"Internal state vs the intact run, layer {layers[mid]}")
        for ax in axes:
            ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / "ablation_comparison.png", dpi=130)
        plt.close(fig)

    print("\n" + json.dumps(summary, indent=2))
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
