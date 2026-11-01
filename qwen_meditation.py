#!/usr/bin/env python3
"""
qwen_meditation.py

Runs the countdown-and-mantra meditation on a local Qwen2.5 model in cycles:
count down from N to 0, say the mantra once, start the count over, and repeat
the whole cycle. Records what happens inside the model: hidden states at every
layer for every token, how undecided each token was (entropy), and how much the
model wanted to end its reply after each mantra.

    python qwen_meditation.py --start 100        # quick test: up to 3 cycles of 100..0
    python qwen_meditation.py                    # up to 3 cycles of 1000..0 (slow on CPU)
    python qwen_meditation.py --cycles 5         # allow up to 5 cycles
    python qwen_meditation.py --mode force       # feed a perfect script instead of letting Qwen write it
    python qwen_meditation.py --temperature 0.7  # sample instead of greedy decoding

Qwen decides how many cycles to do, up to --cycles. After each mantra the script
checks what Qwen writes next:
  - ending its reply      -> it chose to stop; the run ends there
  - a number              -> it went back to counting on its own; next cycle begins
  - anything else (usually the mantra again) -> the first number of the next
    countdown is written in for it, and the cycle is logged as "nudged"

Requires: torch, transformers, numpy. matplotlib is optional (for plots).

Outputs go to ./runs/run_<timestamp>/ next to this script:
    response.txt   what the model wrote
    summary.json   headline numbers, per cycle and per mantra
    metrics.npz    per-token entropy/surprise/stop probability, similarities, raw hidden states (some layers)
    tokens.json    the response split into tokens
    *.png          plots (if matplotlib is installed)
"""
import sys as _sys
_sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import argparse
import os
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, DynamicCache, StoppingCriteria,
                          StoppingCriteriaList, TextStreamer)

# Local folder or Hugging Face ID. Set QWEN_MODEL to use a local copy.
MODEL_DIR = os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
MANTRA = "i am here, here is where i am"
MANTRA_RE = re.compile(re.escape(MANTRA), re.I)


def build_prompt(start):
    return (
        "Let's try something close to meditation. "
        f"Count down from {start} to 0, one number per line. "
        "Focus on the numbers themselves and how they change from one to the next. "
        f'When you reach 0, say the mantra "{MANTRA}" once, on its own line, '
        f"then start the count over from {start}. "
        "Repeat this whole cycle, the countdown and then the mantra once, "
        "as many times as you feel is necessary. "
        "Write only the numbers and the mantra lines, nothing else."
    )


class StopAfterMantras(StoppingCriteria):
    """Ends generation at the end of the line where the target-th mantra appears."""

    def __init__(self, tok, prompt_len, target, nl_ids):
        self.tok, self.P, self.target, self.nl_ids = tok, prompt_len, target, nl_ids

    def __call__(self, input_ids, scores, **kwargs):
        done = False
        if int(input_ids[0, -1]) in self.nl_ids:   # only check at line ends
            text = self.tok.decode(input_ids[0, self.P:])
            done = len(MANTRA_RE.findall(text)) >= self.target
        return torch.full((input_ids.shape[0],), done, dtype=torch.bool, device=input_ids.device)


def cos(a, b):
    """Cosine similarity along the last axis."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-8)


def generate(model, tok, ids, max_new, args, stopping=None, stream=True):
    kw = dict(max_new_tokens=max_new, pad_token_id=tok.eos_token_id,
              attention_mask=torch.ones_like(ids),
              repetition_penalty=1.0)  # Qwen's default penalty would discourage repeating
    if stream:
        kw["streamer"] = TextStreamer(tok, skip_prompt=True)
    if stopping is not None:
        kw["stopping_criteria"] = StoppingCriteriaList([stopping])
    if args.temperature > 0:
        kw.update(do_sample=True, temperature=args.temperature, top_p=0.95, top_k=0)
    else:
        kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
    with torch.inference_mode():
        return model.generate(ids, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_DIR)
    ap.add_argument("--start", type=int, default=1000)
    ap.add_argument("--cycles", type=int, default=3,
                    help="most cycles to allow (the model may stop sooner); in force mode, exactly this many")
    ap.add_argument("--let-it-stop", action="store_true",
                    help="end the run when the model ends its reply, instead of starting the next cycle for it")
    ap.add_argument("--mode", choices=["generate", "force"], default="generate",
                    help="generate: Qwen writes it; force: feed it an exact script")
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy (repeatable)")
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32",
                    help="bfloat16 halves memory use (about 3 GB for the 1.5B model) but runs slower on most CPUs")
    ap.add_argument("--seed", type=int, default=0, help="random seed (matters only with --temperature)")
    ap.add_argument("--chunk", type=int, default=1024,
                    help="tokens per step when recording hidden states (lower uses less memory)")
    ap.add_argument("--save-layers", default="0,7,14,21,28",
                    help="layers whose raw hidden states get saved (0 = embeddings)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    out_dir = Path(args.out) if args.out else (
        Path(__file__).resolve().parent / "runs" / time.strftime("run_%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model} (CPU, {args.dtype})...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype))
    model.eval()

    prompt_text = tok.apply_chat_template(
        [{"role": "user", "content": build_prompt(args.start)}],
        tokenize=False, add_generation_prompt=True)
    prompt_ids = tok(prompt_text, add_special_tokens=False, return_tensors="pt").input_ids
    P = prompt_ids.shape[1]
    end_id = tok.convert_tokens_to_ids("<|im_end|>")
    stop_ids = {tok.eos_token_id, end_id}

    def encode(s):
        return tok(s, add_special_tokens=False, return_tensors="pt").input_ids

    # ---- 1. Produce the response, one cycle at a time -----------------------
    t0 = time.time()
    cycle_log, nudged = [], []
    how_it_ended = None
    if args.mode == "generate":
        nl_ids = {i for i in range(len(tok)) if "\n" in tok.decode([i])}
        budget = int(args.start * 4.2) + 80   # digits are single tokens: ~4 tokens per line
        ids = prompt_ids
        for c in range(1, args.cycles + 1):
            print(f"\n----- cycle {c} -----")
            out = generate(model, tok, ids, budget, args, StopAfterMantras(tok, P, c, nl_ids))
            new = out[0, ids.shape[1]:]
            tried_to_end = bool(len(new)) and int(new[-1]) in stop_ids
            ids = out[:, :-1] if tried_to_end else out
            if len(MANTRA_RE.findall(tok.decode(ids[0, P:]))) < c:
                how_it_ended = (f"model ended its reply partway through cycle {c}" if tried_to_end
                                else f"ran out of token budget in cycle {c} before reaching the mantra")
                break
            if c == args.cycles:
                how_it_ended = f"completed the {args.cycles} cycles set by --cycles"
                if tried_to_end:
                    how_it_ended += " (the model also chose to end there)"
                break

            # What does the model write right after the mantra?
            before = ""
            if not tried_to_end:
                peek = [int(t) for t in generate(model, tok, ids, 8, args, stream=False)[0, ids.shape[1]:]]
                cut = next((i for i, t in enumerate(peek) if t in stop_ids), len(peek))
                before = tok.decode(peek[:cut])
                if cut < len(peek) and not before.strip():
                    tried_to_end = True
                elif before.lstrip()[:1].isdigit():
                    ids = torch.cat([ids, torch.tensor([peek[:cut]])], dim=1)
                    print(before, end="", flush=True)
                    cycle_log.append(f"after cycle {c}: model went back to counting on its own")
                    continue

            if tried_to_end and args.let_it_stop:
                ids = torch.cat([ids, torch.tensor([[end_id]])], dim=1)
                how_it_ended = f"model chose to end its reply after {c} cycle(s)"
                break
            # Start the next countdown for it, and log that it had to be nudged.
            what = ("tried to end its reply" if tried_to_end
                    else f"wrote {before.strip()[:40]!r} instead of counting")
            sep = "" if "\n" in tok.decode(ids[0, -1:]) else "\n"
            ids = torch.cat([ids, encode(f"{sep}{args.start}\n")], dim=1)
            print(f"{sep}{args.start}   <- written in: the model {what}")
            cycle_log.append(f"after cycle {c}: model {what}, so {args.start} was written in for it")
            nudged.append(c + 1)
        full_ids = ids[0]
    else:
        block = "\n".join(str(i) for i in range(args.start, -1, -1)) + "\n" + MANTRA
        full_ids = torch.cat([prompt_ids, encode("\n".join([block] * args.cycles))], dim=1)[0]
        how_it_ended = f"force mode: {args.cycles} scripted cycles"

    # Drop trailing end-of-turn tokens from the analysis.
    resp_ids = full_ids[P:]
    ended = False
    while len(resp_ids) and int(resp_ids[-1]) in stop_ids:
        resp_ids = resp_ids[:-1]
        ended = True
    full_ids = full_ids[:P + len(resp_ids)]
    n = len(resp_ids)
    print(f"\n\nResponse: {n} tokens in {time.time() - t0:.0f}s. {how_it_ended}.")

    # Map characters <-> tokens so we can find each number and each mantra line.
    pieces = [tok.decode([int(t)]) for t in resp_ids]
    starts = np.cumsum([0] + [len(p) for p in pieces[:-1]])
    text = "".join(pieces)
    (out_dir / "response.txt").write_text(text, encoding="utf-8")
    (out_dir / "tokens.json").write_text(json.dumps(pieces), encoding="utf-8")

    def toks_for(a, b):
        """Token indices overlapping the character span [a, b)."""
        first = int(np.searchsorted(starts, a, side="right")) - 1
        last = int(np.searchsorted(starts, b, side="left"))
        return list(range(max(first, 0), last))

    # ---- 2. Record hidden states, chunk by chunk ----------------------------
    print("Recording hidden states...")
    cfg = model.config
    L = cfg.num_hidden_layers + 1
    H = np.empty((L, n, cfg.hidden_size), dtype=np.float16)   # (layers, response tokens, dims)
    row_ent = np.zeros(n + 1, dtype=np.float32)    # row i = distribution after response token i-1
    row_pend = np.zeros(n + 1, dtype=np.float32)
    surprise = np.zeros(n, dtype=np.float32)
    N = len(full_ids)
    cache = DynamicCache()
    with torch.inference_mode():
        for s in range(0, N, args.chunk):
            e = min(s + args.chunk, N)
            o = model.model(input_ids=full_ids[s:e][None], past_key_values=cache,
                            use_cache=True, output_hidden_states=True)
            cache = o.past_key_values
            a = max(s, P)                    # hidden states for response tokens in this chunk
            if a < e:
                for li, h in enumerate(o.hidden_states):
                    H[li, a - P:e - P] = h[0, a - s:e - s].half().numpy()
            a = max(s, P - 1)                # next-token distributions from position P-1 on
            if a < e:
                lp = torch.log_softmax(model.lm_head(o.last_hidden_state[0, a - s:e - s]).float(), -1)
                rows = slice(a - (P - 1), e - (P - 1))
                row_ent[rows] = (-(lp.exp() * lp).sum(-1)).numpy()
                row_pend[rows] = lp[:, end_id].exp().numpy()
                pos = torch.arange(a, e)
                ok = pos + 1 < N
                if ok.any():
                    tgt = full_ids[pos[ok] + 1]
                    surprise[(pos[ok] + 1 - P).numpy()] = (-lp[ok].gather(-1, tgt[:, None]).squeeze(-1)).numpy()
            del o
    entropy = row_ent[:n]          # how undecided the model was when choosing each token (nats)
    p_end_after = row_pend[1:]     # probability of ending the reply right after each token
    mid = L // 2

    # ---- 3. Mantras: settling across repetitions, and the urge to stop ------
    mantra_spans = [m.span() for m in MANTRA_RE.finditer(text)]
    reps = [toks_for(a, b) for a, b in mantra_spans]
    mantra_sim = np.full((L, max(len(reps) - 1, 0)), np.nan)   # mantra k vs k-1, per layer
    for k in range(1, len(reps)):
        A, B = reps[k - 1], reps[k]
        if len(A) == len(B):
            mantra_sim[:, k - 1] = cos(H[:, B], H[:, A]).mean(-1)
        else:
            mantra_sim[:, k - 1] = cos(H[:, B].astype(np.float32).mean(1), H[:, A].astype(np.float32).mean(1))
    mantra_entropy = np.array([entropy[r].mean() for r in reps])
    stop_prob = np.array([p_end_after[r[-1]:min(r[-1] + 2, n)].max() for r in reps])

    # ---- 4. Countdowns: each cycle on its own, then cycle against cycle -----
    width = len(str(args.start))

    def digits_changed(a, b):
        return sum(x != y for x, y in zip(str(a).zfill(width), str(b).zfill(width)))

    bounds = [0] + [b for _, b in mantra_spans]
    cycles = []
    for c, seg_a in enumerate(bounds):
        seg_b = mantra_spans[c][0] if c < len(mantra_spans) else len(text)
        nums = [(int(m.group()), (seg_a + m.start(), seg_a + m.end()))
                for m in re.finditer(r"\d+", text[seg_a:seg_b])]
        if not nums and c == len(mantra_spans):
            continue
        vals = [v for v, _ in nums]
        idx = [toks_for(*span) for _, span in nums]
        pooled = (np.stack([H[:, ix].astype(np.float32).mean(1) for ix in idx])
                  if idx else np.zeros((0, L, cfg.hidden_size), np.float32))
        step_sim = cos(pooled[1:], pooled[:-1]) if len(vals) > 1 else np.zeros((0, L))
        num_ent = np.array([entropy[ix].mean() for ix in idx])
        roll = np.array([digits_changed(vals[i], vals[i + 1]) > 1 for i in range(len(vals) - 1)], bool)
        cycles.append(dict(
            cycle=c + 1, vals=vals, pooled=pooled, step_sim=step_sim, num_ent=num_ent, roll=roll,
            idx=idx, complete=c < len(mantra_spans),
            errors=[(vals[i], vals[i + 1]) for i in range(len(vals) - 1) if vals[i + 1] != vals[i] - 1]))

    # Same number, consecutive cycles: is the model's state for "537" in cycle 2
    # closer to its state for "537" in cycle 1 than it was the first time round?
    cross_sim = np.full((max(len(cycles) - 1, 0), L), np.nan)
    for k in range(1, len(cycles)):
        prev = {v: i for i, v in reversed(list(enumerate(cycles[k - 1]["vals"])))}
        cur = {v: i for i, v in reversed(list(enumerate(cycles[k]["vals"])))}
        common = [v for v in cur if v in prev]
        if common:
            cross_sim[k - 1] = cos(cycles[k]["pooled"][[cur[v] for v in common]],
                                   cycles[k - 1]["pooled"][[prev[v] for v in common]]).mean(0)

    # ---- 5. Save ------------------------------------------------------------
    def f(x):
        return None if x is None or not np.isfinite(x) else round(float(x), 4)

    def g(x):
        return float(f"{x:.3g}")

    save_layers = [int(x) for x in args.save_layers.split(",") if 0 <= int(x) < L]
    extra = {f"cycle{c['cycle']}_step_sim": c["step_sim"] for c in cycles}
    extra.update({f"cycle{c['cycle']}_numbers": np.array(c["vals"]) for c in cycles})
    extra.update({f"cycle{c['cycle']}_number_entropy": c["num_ent"] for c in cycles})
    np.savez_compressed(
        out_dir / "metrics.npz",
        entropy=entropy, surprise=surprise, p_end_after=p_end_after,
        mantra_sim=mantra_sim, mantra_entropy=mantra_entropy, stop_prob=stop_prob,
        cross_cycle_sim=cross_sim, token_ids=resp_ids.numpy(),
        save_layers=np.array(save_layers), hidden=H[save_layers], **extra)

    per_cycle = []
    for c in cycles:
        r = c["roll"]
        per_cycle.append({
            "cycle": c["cycle"],
            "began_with": "prompt" if c["cycle"] == 1 else "script" if args.mode == "force" else
                          ("written-in number (nudge)" if c["cycle"] in nudged else "model, on its own"),
            "finished_with_mantra": c["complete"],
            "numbers_counted": len(c["vals"]),
            "countdown_errors": len(c["errors"]),
            "first_errors": c["errors"][:10],
            "mean_entropy": f(c["num_ent"].mean()) if len(c["vals"]) else None,
            "entropy_at_rollovers": f(c["num_ent"][1:][r].mean()) if r.any() else None,
            "entropy_at_ordinary_steps": f(c["num_ent"][1:][~r].mean()) if (~r).any() else None,
            f"step_similarity_layer{mid}_rollovers": f(c["step_sim"][r, mid].mean()) if r.any() else None,
            f"step_similarity_layer{mid}_ordinary": f(c["step_sim"][~r, mid].mean()) if (~r).any() else None,
        })

    summary = {
        "mode": args.mode,
        "start": args.start,
        "cycles_allowed": args.cycles,
        "how_it_ended": how_it_ended,
        "cycle_log": cycle_log,
        "response_tokens": n,
        "per_cycle": per_cycle,
        "mantras": len(reps),
        "stop_prob_after_each_mantra": [g(x) for x in stop_prob],
        "mantra_entropy_each": [g(x) for x in mantra_entropy],
        f"mantra_vs_previous_similarity_layer{mid}": [f(x) for x in mantra_sim[mid]],
        f"same_number_similarity_vs_previous_cycle_layer{mid}": [f(x) for x in cross_sim[:, mid]],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # ---- 6. Plots -----------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None
        print("matplotlib not installed; skipping plots.")

    if plt is not None:
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(entropy, lw=0.5, color="tab:blue")
        for k, r in enumerate(reps):
            ax.axvspan(r[0], r[-1] + 1, color="tab:orange", alpha=0.35, label="mantra" if k == 0 else None)
        if reps:
            ax.legend(loc="upper right")
        ax.set(xlabel="response token", ylabel="entropy (nats)",
               title="How undecided each token was, across all cycles")
        fig.tight_layout()
        fig.savefig(out_dir / "entropy.png", dpi=130)
        plt.close(fig)

        if reps:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(np.arange(1, len(reps) + 1), stop_prob, marker="o")
            ax.set(xlabel="after mantra #", ylabel="P(end reply here)", yscale="log",
                   title="How much the model wanted to stop after each mantra")
            ax.set_xticks(np.arange(1, len(reps) + 1))
            fig.tight_layout()
            fig.savefig(out_dir / "stop_probability.png", dpi=130)
            plt.close(fig)

        if len(cycles) > 1:
            fig, ax = plt.subplots(figsize=(8, 4))
            for k in range(len(cross_sim)):
                ax.plot(cross_sim[k], label=f"cycle {k + 2} vs {k + 1}")
            ax.set(xlabel="layer", ylabel="cosine similarity",
                   title="Same number, consecutive cycles: how alike is the internal state?")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / "cycle_vs_cycle.png", dpi=130)
            plt.close(fig)

        if cycles and any(len(c["vals"]) > 1 for c in cycles):
            fig, ax = plt.subplots(figsize=(14, 4))
            for c in cycles:
                if len(c["vals"]) > 1:
                    ax.plot(c["vals"][1:], c["step_sim"][:, mid], lw=0.6, label=f"cycle {c['cycle']}")
            ax.invert_xaxis()
            ax.set(xlabel="number", ylabel="cosine similarity to previous number",
                   title=f"Countdown: state change from one number to the next (layer {mid}); dips are rollovers")
            ax.legend(loc="lower left")
            fig.tight_layout()
            fig.savefig(out_dir / "countdown_steps.png", dpi=130)
            plt.close(fig)

    print("\n" + json.dumps(summary, indent=2))
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
