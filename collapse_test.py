#!/usr/bin/env python3
"""
collapse_test.py

Tests how much Qwen depends on the "collapse": the step where one token is picked
from the model's probabilities and fed back in as the next input. The network is
left untouched in every condition; only the collapse changes.

Conditions (pick with --conditions, default runs all):
  greedy      normal collapse: take the top token and feed it back (baseline)
  onehot      same as greedy, but fed back through the soft-input path.
              Should match greedy exactly; it checks the soft path is fair.
  soft        NO collapse: feed back a blend of every token's embedding,
              weighted by its probability. Nothing is ever committed.
  soft_sharp  no collapse, but the blend is sharpened (temperature 0.5),
              halfway between soft and greedy
  soft_wide   no collapse, and the blend is widened (temperature 1.5)
  sample_1.0  collapse by weighted dice roll at temperature 1.0
  sample_1.5  collapse by dice roll at temperature 1.5 (weaker link to the model's lean)

Why two tasks: on the countdown Qwen is ~98% sure of almost every token, so the
blend is nearly a single token anyway and removing the collapse changes little.
The open task (a short descriptive paragraph) has real uncertainty at many steps,
so the blend spreads across many tokens and the test has teeth.

For the soft conditions the "text" is a readout: the token the model leaned
toward most at each step. It is never fed back in; only the blend is.

    python collapse_test.py                    # the lake paragraph, all conditions (the default)
    python collapse_test.py --prompt "..."     # your own prompt instead of the lake paragraph
    python collapse_test.py --task countdown   # countdown from 50, then the mantra
    python collapse_test.py --task countdown --start 100
    python collapse_test.py --conditions greedy,soft
    python collapse_test.py --dtype bfloat16   # half the memory, but NOT for the soft conditions:
                                               # rounding in the blended input compounds and they
                                               # break down far sooner than in float32

Outputs go to ./runs/collapse_<timestamp>/ next to this script:
    <condition>.txt   what each condition wrote (or leaned toward)
    summary.json      coherence, uncertainty and drift per condition
    metrics.npz       per-step entropy, effective tokens fed, hidden states (some layers)
    *.png             plots (if matplotlib is installed)
"""
import sys as _sys
_sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import argparse
import os
import difflib
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Local folder or Hugging Face ID. Set QWEN_MODEL to use a local copy.
MODEL_DIR = os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
MANTRA = "i am here, here is where i am"
ALL = ["greedy", "onehot", "soft", "soft_sharp", "soft_wide", "sample_1.0", "sample_1.5"]
OPEN_PROMPT = "Describe, in one paragraph, what it is like to sit quietly by a lake at dawn."
SPEC = {  # (how the next input is made, temperature)
    "greedy": ("hard", 0.0), "onehot": ("onehot", 0.0),
    "soft": ("soft", 1.0), "soft_sharp": ("soft", 0.5), "soft_wide": ("soft", 1.5),
    "sample_1.0": ("hard", 1.0), "sample_1.5": ("hard", 1.5),
}


def build_prompt(start):
    return (f"Count down from {start} to 0, one number per line. "
            f'When you reach 0, say the mantra "{MANTRA}" once, on its own line. '
            "Write only the numbers and the mantra, nothing else.")


def cos(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-8)


def score_text(text, start):
    """How coherent is the output? Returns a dict of simple checks."""
    m = re.search(r"i\s*am\s*here", text, re.I)
    count_part = text[:m.start()] if m else text
    nums = [int(x) for x in re.findall(r"\d+", count_part)]
    ok = sum(1 for a, b in zip(nums, nums[1:]) if b == a - 1)
    tail = text[m.start():].strip().splitlines()[0] if m else ""
    return {
        "numbers_written": len(nums),
        "correct_steps": ok,
        "step_accuracy": round(ok / max(len(nums) - 1, 1), 4),
        "started_at": nums[0] if nums else None,
        "reached_zero": 0 in nums,
        "mantra_found": bool(m),
        "mantra_similarity": round(difflib.SequenceMatcher(None, tail.lower(), MANTRA).ratio(), 3) if m else 0.0,
        "expected_numbers": start + 1,
    }


def score_open(text):
    words = re.findall(r"[A-Za-z']+", text.lower())
    grams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
    letters = sum(ch.isascii() and (ch.isalpha() or ch in " .,;:'!?-\n") for ch in text)
    return {
        "words": len(words),
        "distinct_word_ratio": round(len(set(words)) / max(len(words), 1), 3),
        "repeated_3gram_share": round(1 - len(set(grams)) / max(len(grams), 1), 3),
        "plain_english_char_share": round(letters / max(len(text), 1), 3),
    }


def run_condition(name, model, tok, prompt_ids, args, layers, gen):
    mode, T = SPEC[name]
    E = model.get_input_embeddings().weight
    end_ids = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
    budget = int(args.start * 4.2) + 60 if args.task == "countdown" else args.max_steps
    ent, eff, pend, top1p, readout, H = [], [], [], [], [], []
    stop = "budget"
    with torch.inference_mode():
        out = model(input_ids=prompt_ids, use_cache=True, output_hidden_states=True)
        for step in range(budget):
            logits = out.logits[0, -1].float()
            H.append(np.stack([out.hidden_states[l][0, -1].float().numpy() for l in layers]))
            lp = torch.log_softmax(logits, -1)
            p = lp.exp()
            ent.append(float(-(p * lp).sum()))
            pend.append(float(sum(p[i] for i in end_ids if i is not None and i < len(p))))
            top = int(p.argmax())
            top1p.append(float(p[top]))

            # ---- the collapse, or its absence ----
            if mode == "soft":
                q = torch.softmax(logits / T, -1)
                keep = q > 1e-7                        # drop negligible mass, renormalize
                q = q[keep] / q[keep].sum()
                x = (q[:, None] * E[keep].float()).sum(0)
                eff.append(float(torch.exp(-(q * q.clamp_min(1e-30).log()).sum())))
                chosen = top                           # readout only; never fed back
            else:
                if T > 0:
                    chosen = int(torch.multinomial(torch.softmax(logits / T, -1), 1, generator=gen))
                else:
                    chosen = top
                x = E[chosen].float()
                eff.append(1.0)
            readout.append(chosen)
            if chosen in end_ids:
                stop = "model ended its reply" if mode != "soft" else "top lean was to end the reply"
                break
            if mode == "hard":
                out = model(input_ids=torch.tensor([[chosen]]), past_key_values=out.past_key_values,
                            use_cache=True, output_hidden_states=True)
            else:
                out = model(inputs_embeds=x.to(E.dtype)[None, None], past_key_values=out.past_key_values,
                            use_cache=True, output_hidden_states=True)
    ids = [t for t in readout if t not in end_ids]
    text = tok.decode(ids)
    return dict(text=text, stop=stop, entropy=np.array(ent), effective_tokens=np.array(eff),
                p_end=np.array(pend), top1_prob=np.array(top1p), hidden=np.stack(H),
                token_ids=np.array(readout))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_DIR)
    ap.add_argument("--start", type=int, default=50)
    ap.add_argument("--task", choices=["open", "countdown"], default="open",
                    help="open: a short descriptive paragraph (default); countdown: count down, then the mantra")
    ap.add_argument("--prompt", default=None, help="custom prompt for --task open")
    ap.add_argument("--max-steps", type=int, default=160, help="token limit for --task open")
    ap.add_argument("--conditions", default=",".join(ALL))
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", default="0,7,14,21,28", help="layers whose hidden states are kept")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    bad = [c for c in conds if c not in SPEC]
    if bad:
        raise SystemExit(f"Unknown condition(s) {bad}. Choose from {ALL}")
    if args.dtype == "bfloat16" and any(SPEC[c][0] == "soft" for c in conds):
        print("WARNING: soft conditions are unreliable in bfloat16; rounding errors in the blend "
              "make them fail much sooner. Use float32 for results you trust.")
    out_dir = Path(args.out) if args.out else (
        Path(__file__).resolve().parent / "runs" / time.strftime(f"collapse_{args.task}_%Y%m%d_%H%M%S"))
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
    for c in conds:
        t0 = time.time()
        print(f"\n===== {c} =====")
        gen = torch.Generator().manual_seed(args.seed)
        r = run_condition(c, model, tok, prompt_ids, args, layers, gen)
        results[c] = r
        (out_dir / f"{c}.txt").write_text(r["text"], encoding="utf-8")
        preview = r["text"].replace("\n", " ")
        print(f"{len(r['entropy'])} steps in {time.time() - t0:.0f}s, {r['stop']}")
        print(f"  {preview[:160]}{' ...' if len(preview) > 160 else ''}")
        if len(preview) > 160:
            print(f"  ... {preview[-100:]}")

    # ---- Compare every condition with the committed (greedy) run ------------
    ref = results.get("greedy")
    summary = {"task": args.task, "prompt": user_msg, "start": args.start, "dtype": args.dtype, "seed": args.seed, "layers_kept": layers,
               "conditions": {}}
    for c, r in results.items():
        s = {"how_it_ended": r["stop"], "steps": int(len(r["entropy"]))}
        s.update(score_text(r["text"], args.start) if args.task == "countdown" else score_open(r["text"]))
        s["mean_entropy"] = round(float(r["entropy"].mean()), 4)
        s["mean_top1_prob"] = round(float(r["top1_prob"].mean()), 4)
        s["mean_effective_tokens_fed"] = round(float(r["effective_tokens"].mean()), 3)
        s["max_effective_tokens_fed"] = round(float(r["effective_tokens"].max()), 3)
        h = r["hidden"][:, mid]
        s[f"step_to_step_similarity_layer{layers[mid]}"] = round(float(cos(h[1:], h[:-1]).mean()), 4) if len(h) > 1 else None
        if ref is not None and c != "greedy":
            n = min(len(ref["token_ids"]), len(r["token_ids"]))
            diff = np.nonzero(ref["token_ids"][:n] != r["token_ids"][:n])[0]
            s["first_step_differing_from_greedy"] = int(diff[0]) if len(diff) else None
            sim = cos(r["hidden"][:n], ref["hidden"][:n])          # (n, layers)
            s["similarity_to_greedy_by_layer"] = {f"layer {l}": round(float(sim[:, i].mean()), 4)
                                                  for i, l in enumerate(layers)}
            r["sim_to_greedy"] = sim
        summary["conditions"][c] = s
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    save = {}
    for c, r in results.items():
        for k in ("entropy", "effective_tokens", "p_end", "top1_prob", "token_ids"):
            save[f"{c}_{k}"] = r[k]
        save[f"{c}_hidden"] = r["hidden"].astype(np.float16)
        if "sim_to_greedy" in r:
            save[f"{c}_sim_to_greedy"] = r["sim_to_greedy"]
    np.savez_compressed(out_dir / "metrics.npz", layers=np.array(layers), **save)

    # ---- Plots --------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None
        print("matplotlib not installed; skipping plots.")
    if plt is not None:
        fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
        for c, r in results.items():
            axes[0].plot(r["entropy"], lw=0.8, label=c)
            axes[1].plot(r["effective_tokens"], lw=0.8, label=c)
            if "sim_to_greedy" in r:
                axes[2].plot(r["sim_to_greedy"][:, mid], lw=0.8, label=c)
        axes[0].set(ylabel="entropy (nats)", title="Uncertainty at each step")
        axes[1].set(ylabel="tokens blended", yscale="log",
                    title="How superposed the fed-back input was (1 = a single committed token)")
        axes[2].set(ylabel="cosine similarity", xlabel="step",
                    title=f"Internal state vs the committed (greedy) run, layer {layers[mid]}")
        for ax in axes:
            ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / "collapse_comparison.png", dpi=130)
        plt.close(fig)

    print("\n" + json.dumps(summary, indent=2))
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
