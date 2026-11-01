#!/usr/bin/env python3
"""
wake_test.py

Two follow-ups to collapse_test.py, run over several prompts so one lucky or
unlucky prompt can't decide the result.

--experiment multi   (the replication)
    For each prompt: normal generation (greedy) vs. generation with the collapse
    removed for the whole reply (soft blend fed back, as in collapse_test.py).

--experiment sweep   (dose-response) the wake test at several blend temperatures,
                     counting how many prompts went under and how many recovered.

--experiment matched (the control for the sweep) the same windows and doses, but
                     instead of a blend, the model COMMITS to one token drawn at
                     random from the very distribution the blend would be made of.
                     Same spread of possibilities, same temperature, ordinary
                     inputs; only the commitment differs from the sweep. Add
                     --compare-to <sweep run folder> to plot both arms together.

--experiment rescaled (the length control for the sweep) the same windows and doses as
                     the sweep, but each blend is rescaled before it is fed back so its
                     length matches the probability-weighted average length of the token
                     embeddings it was made from. Averaging many embeddings shrinks the
                     vector; this arm removes that shrinkage and leaves only the blending.
                     Add --compare-to <sweep run folder> to plot both arms together.

--experiment wake    ("anesthetize, then wake up")
    For each prompt, generation runs normally, then for a window of steps in the
    middle the collapse is removed, then it is restored. Does the model recover?
    Conditions:
      greedy      no window (the reference)
      wake_soft   window: soft blend at temperature 1.0 (light anesthetic)
      wake_deep   window: soft blend at temperature 1.5 (deep anesthetic)
      wake_noise  window: a random real token's embedding each step, ignoring the
                  model entirely (a "concussion" control: same disruption length,
                  but noise instead of the model's own blurred lean)
    During the window nothing the model leans toward is committed; the text shown
    for it is a readout of its top lean only. The model cannot end its reply
    during the window.

Note: the prompt and everything before the window stay in the model's memory
(its cached states) the whole time. Only the window's own inputs are blended.

    python wake_test.py --experiment multi
    python wake_test.py --experiment wake                        # window at steps 40-59
    python wake_test.py --experiment wake --window-start 60 --window-len 30
    python wake_test.py --experiment wake --prompts 3            # first 3 prompts only
    python wake_test.py --experiment sweep                       # doses 1.0-1.5: where's the tipping point?
    python wake_test.py --experiment sweep --doses 1.2,1.25,1.3  # zoom in once you know roughly where
    python wake_test.py --experiment matched --compare-to runs\\sweep_20261027_1500
    python wake_test.py --experiment rescaled --compare-to runs\\sweep
    python wake_test.py --prompts-file my_prompts.txt            # one prompt per line

Needs collapse_test.py in the same folder. Use float32 (the default).

Outputs go to ./runs/<experiment>_<timestamp>/:
    texts.md        every prompt and condition, with the window marked  [[ like this ]]
    summary.json    per-run measurements and averages per condition
    metrics.npz     per-step entropy, blend width, similarity to greedy
    <experiment>.png  plots
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

from collapse_test import MODEL_DIR, cos, score_open

PROMPTS = [
    "Describe, in one paragraph, what it is like to sit quietly by a lake at dawn.",
    "Explain how a bicycle's gears make it easier to ride up a hill.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Describe a busy outdoor market in a city you imagine, using all five senses.",
    "Write a friendly letter to an old friend about a trip you just took to the mountains.",
    "Explain to a curious ten-year-old why the sky is blue during the day and red at sunset.",
]
STOP = set("""a an the and or but of to in on at by for with from as is are was were be been it its
this that these those you your i my me we our they their he she his her them what why how when which
who into about over under than then so if do does did can could would should will just also very each
using all one write short describe explain paragraph""".split())


def content_words(text):
    return {w for w in re.findall(r"[a-z']+", text.lower()) if len(w) > 3 and w not in STOP}


def loops(text):
    s = score_open(text)
    return s["repeated_3gram_share"]


def generate(model, tok, prompt_ids, steps, layers, gen, window=None, wmode=None, T=1.0,
             can_stop_in_window=False):
    """Greedy generation; inside window=(a, b) the collapse is replaced per wmode."""
    E = model.get_input_embeddings().weight
    end_ids = {i for i in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")) if i is not None}
    real_vocab = len(tok)
    ent, eff, readout, H = [], [], [], []
    ratio = []  # blend length / weighted average length of the embeddings blended (window steps only)
    stop = "step limit"
    with torch.inference_mode():
        out = model(input_ids=prompt_ids, use_cache=True, output_hidden_states=True)
        for step in range(steps):
            logits = out.logits[0, -1].float()
            H.append(np.stack([out.hidden_states[l][0, -1].float().numpy() for l in layers]))
            lp = torch.log_softmax(logits, -1)
            p = lp.exp()
            ent.append(float(-(p * lp).sum()))
            top = int(p.argmax())
            in_window = window is not None and window[0] <= step < window[1]

            if in_window and top in end_ids and not can_stop_in_window:
                masked = logits.clone()
                masked[list(end_ids)] = -1e9
                top = int(masked.argmax())
            if in_window and wmode in ("soft", "sample", "soft_rescaled"):
                # the distribution the window works from: same for both arms
                q = torch.softmax(logits / T, -1)
                q[list(end_ids)] = 0
                keep = q > 1e-7
                q = q[keep] / q[keep].sum()
                width = float(torch.exp(-(q * q.clamp_min(1e-30).log()).sum()))
                if wmode == "sample":
                    # commit to one token drawn from that distribution; it is shown and fed back
                    top = int(keep.nonzero()[int(torch.multinomial(q, 1, generator=gen))])
            readout.append(top)
            if top in end_ids:
                stop = "model ended its reply" if not in_window else "top lean was to end the reply"
                break

            if in_window and wmode in ("soft", "soft_rescaled"):
                Ek = E[keep].float()
                x = (q[:, None] * Ek).sum(0)
                target = float((q * Ek.norm(dim=1)).sum())
                ratio.append(float(x.norm()) / target)
                if wmode == "soft_rescaled":
                    x = x * (target / x.norm())
                eff.append(width)
                out = model(inputs_embeds=x.to(E.dtype)[None, None], past_key_values=out.past_key_values,
                            use_cache=True, output_hidden_states=True)
            elif in_window and wmode == "noise":
                r = int(torch.randint(0, real_vocab, (1,), generator=gen))
                eff.append(float("nan"))
                out = model(input_ids=torch.tensor([[r]]), past_key_values=out.past_key_values,
                            use_cache=True, output_hidden_states=True)
            else:
                # for the sampled arm, record how wide the distribution it drew from was,
                # so its "went under" measure matches the blended arm's
                eff.append(width if in_window and wmode == "sample" else 1.0)
                out = model(input_ids=torch.tensor([[top]]), past_key_values=out.past_key_values,
                            use_cache=True, output_hidden_states=True)
    return dict(ids=np.array(readout), entropy=np.array(ent), eff=np.array(eff), hidden=np.stack(H), stop=stop,
                ratio=np.array(ratio))


def segment_text(tok, ids, a, b):
    end_ids = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
    clean = lambda xs: tok.decode([int(t) for t in xs if int(t) not in end_ids])
    return clean(ids[:a]), clean(ids[a:b]), clean(ids[b:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=["multi", "wake", "sweep", "matched", "rescaled"], default="wake")
    ap.add_argument("--doses", default="1.0,1.1,1.2,1.3,1.4,1.5",
                    help="sweep: blend temperatures for the window (higher = deeper anesthetic)")
    ap.add_argument("--compare-to", default=None,
                    help="matched/rescaled: a sweep run folder, to plot the blended arm alongside")
    ap.add_argument("--under-width", type=float, default=50,
                    help="sweep: a run counts as 'went under' if the blend got wider than this many tokens")
    ap.add_argument("--model", default=MODEL_DIR)
    ap.add_argument("--prompts", type=int, default=len(PROMPTS), help="how many of the built-in prompts")
    ap.add_argument("--prompts-file", default=None)
    ap.add_argument("--steps", type=int, default=160)
    ap.add_argument("--window-start", type=int, default=40)
    ap.add_argument("--window-len", type=int, default=20)
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", default="0,7,14,21,28")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    prompts = ([l.strip() for l in Path(args.prompts_file).read_text(encoding="utf-8").splitlines() if l.strip()]
               if args.prompts_file else PROMPTS[:args.prompts])
    if args.dtype == "bfloat16":
        print("WARNING: blended inputs are unreliable in bfloat16. Use float32 for results you trust.")

    out_dir = Path(args.out) if args.out else (
        Path(__file__).resolve().parent / "runs" / time.strftime(f"{args.experiment}_%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model} ({args.dtype})...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype))
    model.eval()
    L = model.config.num_hidden_layers
    layers = [int(x) for x in args.layers.split(",") if 0 <= int(x) <= L]
    mid = layers.index(min(layers, key=lambda l: abs(l - L // 2)))

    a = args.window_start
    b = a + args.window_len
    if args.experiment == "multi":
        conds = {"greedy": dict(), "soft": dict(window=(0, args.steps), wmode="soft", T=1.0, can_stop_in_window=True)}
    elif args.experiment in ("sweep", "matched", "rescaled"):
        conds = {"greedy": dict()}
        arm, prefix = {"sweep": ("soft", "dose"), "matched": ("sample", "sampled"),
                       "rescaled": ("soft_rescaled", "rescaled")}[args.experiment]
        for t in [float(x) for x in args.doses.split(",") if x.strip()]:
            conds[f"{prefix}_{t:.2f}"] = dict(window=(a, b), wmode=arm, T=t)
    else:
        conds = {"greedy": dict(),
                 "wake_soft": dict(window=(a, b), wmode="soft", T=1.0),
                 "wake_deep": dict(window=(a, b), wmode="soft", T=1.5),
                 "wake_noise": dict(window=(a, b), wmode="noise")}

    runs, md = [], [f"# {args.experiment} run\n"]
    if args.experiment != "multi":
        md.append(f"Window: steps {a}-{b - 1}, shown [[ like this ]]. " + (
            "Text inside the window is the token the model committed to, drawn at random from its "
            "distribution at that dose; it was fed back as normal.\n" if args.experiment == "matched" else
            "Text inside the window is the model's top lean only; it was never fed back.\n"))
    save = {"layers": np.array(layers)}
    for pi, prompt in enumerate(prompts):
        chat = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                       add_generation_prompt=True)
        prompt_ids = tok(chat, add_special_tokens=False, return_tensors="pt").input_ids
        md.append(f"\n## Prompt {pi + 1}: {prompt}\n")
        res = {}
        for name, kw in conds.items():
            t0 = time.time()
            gen = torch.Generator().manual_seed(args.seed + pi)
            r = generate(model, tok, prompt_ids, args.steps, layers, gen, **kw)
            res[name] = r
            print(f"[prompt {pi + 1}/{len(prompts)}] {name}: {len(r['ids'])} steps, {r['stop']} "
                  f"({time.time() - t0:.0f}s)")
            if args.experiment != "multi" and name != "greedy":
                pre, win, post = segment_text(tok, r["ids"], a, b)
                md.append(f"**{name}** ({r['stop']}):\n\n{pre}[[{win}]]{post}\n")
            else:
                md.append(f"**{name}** ({r['stop']}):\n\n{tok.decode([int(t) for t in r['ids']])}\n")
            for k in ("ids", "entropy", "eff", "ratio"):
                save[f"p{pi + 1}_{name}_{k}"] = r[k]

        g = res["greedy"]
        g_pre, _, g_post = segment_text(tok, g["ids"], a, b)
        pwords = content_words(prompt)
        for name, r in res.items():
            row = {"prompt": pi + 1, "condition": name, "steps": int(len(r["ids"])), "how_it_ended": r["stop"]}
            text = tok.decode([int(t) for t in r["ids"]])
            row.update(score_open(text))
            row["mean_entropy"] = round(float(r["entropy"].mean()), 4)
            fin = r["eff"][np.isfinite(r["eff"])]
            row["mean_blend_width"] = round(float(fin.mean()), 2) if len(fin) else None
            n = min(len(g["ids"]), len(r["ids"]))
            sim = cos(r["hidden"][:n], g["hidden"][:n])[:, mid] if n else np.array([])
            save[f"p{pi + 1}_{name}_sim_to_greedy"] = sim
            if args.experiment == "multi":
                diff = np.nonzero(g["ids"][:n] != r["ids"][:n])[0]
                row["first_step_differing_from_greedy"] = int(diff[0]) if len(diff) else None
                row["similarity_to_greedy_mid_layer"] = round(float(sim.mean()), 4) if n else None
            elif name != "greedy":
                pre, win, post = segment_text(tok, r["ids"], a, b)
                ps = score_open(post) if post.strip() else {}
                row["window_text"] = win
                row["after_window"] = {
                    "words": ps.get("words", 0),
                    "plain_english_char_share": ps.get("plain_english_char_share"),
                    "repeated_3gram_share": ps.get("repeated_3gram_share"),
                    "distinct_word_ratio": ps.get("distinct_word_ratio"),
                    # does it follow the original prompt? share of the prompt's content words it uses
                    "prompt_words_used": round(len(pwords & content_words(post)) / max(len(pwords), 1), 3),
                    "greedy_prompt_words_used": round(len(pwords & content_words(g_post)) / max(len(pwords), 1), 3),
                    # does it rejoin the original text line? overlap with greedy's own continuation
                    "content_overlap_with_greedy": round(
                        len(content_words(post) & content_words(g_post)) /
                        max(len(content_words(post) | content_words(g_post)), 1), 3),
                    "similarity_to_greedy_during_window": round(float(sim[a:b].mean()), 4) if n > a else None,
                    "similarity_to_greedy_after_window": round(float(sim[b:].mean()), 4) if n > b else None,
                    "similarity_to_greedy_last_20_steps": round(float(sim[-20:].mean()), 4) if n > b + 20 else None,
                }
                w = r["eff"][a:b]
                w = w[np.isfinite(w)]
                aw = row["after_window"]
                aw["max_blend_width_in_window"] = round(float(w.max()), 2) if len(w) else None
                aw["went_under"] = bool(len(w) and w.max() > args.under_width)
                rr = r["ratio"]
                aw["blend_length_ratio_mean"] = round(float(rr.mean()), 4) if len(rr) else None
                aw["blend_length_ratio_min"] = round(float(rr.min()), 4) if len(rr) else None
                # recovered: readable English after the window, without falling into loops
                aw["recovered"] = bool(aw["words"] >= 10 and (aw["plain_english_char_share"] or 0) >= 0.95
                                       and (aw["repeated_3gram_share"] or 0) < 0.2)
            runs.append(row)

    # ---- averages per condition ---------------------------------------------
    def avg(rows, key, sub=None):
        vals = [(r[sub] if sub else r).get(key) for r in rows]
        vals = [v for v in vals if isinstance(v, (int, float)) and np.isfinite(v)]
        return round(float(np.mean(vals)), 4) if vals else None

    averages = {}
    for name in conds:
        rows = [r for r in runs if r["condition"] == name]
        d = {"runs": len(rows),
             "ended_on_its_own": sum(r["how_it_ended"] != "step limit" for r in rows),
             "runs_with_loops (>10% repeated 3-grams)": sum((r["repeated_3gram_share"] or 0) > 0.1 for r in rows)}
        for k in ("distinct_word_ratio", "repeated_3gram_share", "plain_english_char_share", "mean_entropy",
                  "mean_blend_width", "first_step_differing_from_greedy", "similarity_to_greedy_mid_layer"):
            v = avg(rows, k)
            if v is not None:
                d[k] = v
        if args.experiment != "multi" and name != "greedy":
            for k in ("plain_english_char_share", "repeated_3gram_share", "prompt_words_used",
                      "greedy_prompt_words_used", "content_overlap_with_greedy",
                      "similarity_to_greedy_during_window", "similarity_to_greedy_after_window",
                      "similarity_to_greedy_last_20_steps", "max_blend_width_in_window",
                      "blend_length_ratio_mean", "blend_length_ratio_min"):
                d[f"after_window_{k}"] = avg(rows, k, "after_window")
            d["share_went_under"] = round(float(np.mean([r["after_window"]["went_under"] for r in rows])), 3)
            d["share_recovered"] = round(float(np.mean([r["after_window"]["recovered"] for r in rows])), 3)
            under = [r for r in rows if r["after_window"]["went_under"]]
            d["recovered_of_those_that_went_under"] = (f"{sum(r['after_window']['recovered'] for r in under)}"
                                                       f" of {len(under)}")
        averages[name] = d

    summary = {"experiment": args.experiment, "dtype": args.dtype, "steps": args.steps,
               "window": [a, b - 1] if args.experiment != "multi" else None,
               "prompts": prompts, "averages": averages, "runs": runs}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "texts.md").write_text("\n".join(md), encoding="utf-8")
    np.savez_compressed(out_dir / "metrics.npz", **save)

    # ---- plot -----------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None
        print("matplotlib not installed; skipping plots.")
    if plt is not None:
        fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
        colors = dict(zip(conds, plt.rcParams["axes.prop_cycle"].by_key()["color"]))
        for name in conds:
            if name == "greedy":
                continue
            sims = [save[f"p{i + 1}_{name}_sim_to_greedy"] for i in range(len(prompts))]
            ents = [save[f"p{i + 1}_{name}_entropy"] for i in range(len(prompts))]
            for i, (s, e) in enumerate(zip(sims, ents)):
                axes[0].plot(s, color=colors[name], lw=0.6, alpha=0.6, label=name if i == 0 else None)
                axes[1].plot(e, color=colors[name], lw=0.6, alpha=0.6, label=name if i == 0 else None)
        for i in range(len(prompts)):
            axes[1].plot(save[f"p{i + 1}_greedy_entropy"], color="gray", lw=0.6, alpha=0.6,
                         label="greedy" if i == 0 else None)
        if args.experiment != "multi":
            for ax in axes:
                ax.axvspan(a, b, color="orange", alpha=0.15)
        axes[0].set(ylabel="cosine similarity", title=f"Internal state vs greedy, layer {layers[mid]}, one line per prompt"
                    + (" (orange = collapse removed)" if args.experiment != "multi" else ""))
        axes[1].set(ylabel="entropy (nats)", xlabel="step", title="Uncertainty at each step")
        for ax in axes:
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"{args.experiment}.png", dpi=130)
        plt.close(fig)

    if plt is not None and args.experiment in ("sweep", "matched", "rescaled"):
        arms = {}
        names = [c for c in conds if c != "greedy"]
        arms[{"matched": "committed (sampled)", "rescaled": "blended, length restored",
              "sweep": "blended"}[args.experiment]] = (
            [float(c.split("_")[1]) for c in names], [averages[c] for c in names])
        if args.compare_to:
            try:
                other = json.loads((Path(args.compare_to) / "summary.json").read_text(encoding="utf-8"))["averages"]
                on = [c for c in other if c.startswith("dose_")]
                arms["blended (sweep)"] = ([float(c.split("_")[1]) for c in on], [other[c] for c in on])
            except Exception as e:
                print(f"Couldn't read --compare-to run: {e}")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
        for (label, (doses, rows)), col in zip(arms.items(), ("tab:blue", "tab:orange")):
            ax1.plot(doses, [r["share_recovered"] for r in rows], "o-", color=col, label=f"{label}: recovered")
            ax1.plot(doses, [r["share_went_under"] for r in rows], "x--", color=col, alpha=0.5,
                     label=f"{label}: went under")
            ax2.plot(doses, [r["after_window_similarity_to_greedy_last_20_steps"] for r in rows], "o-",
                     color=col, label=label)
        ax1.set(xlabel="temperature in the window (dose)", ylabel="share of prompts", ylim=(-0.05, 1.05),
                title="Recovery after the window")
        ax2.set(xlabel="temperature in the window (dose)", ylabel="cosine similarity",
                title=f"How close it returns to the normal run (last 20 steps, layer {layers[mid]})")
        for ax in (ax1, ax2):
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "dose_response.png", dpi=130)
        plt.close(fig)

    print("\n" + json.dumps(averages, indent=2))
    print(f"\nSaved to {out_dir}  (read texts.md to see every run)")


if __name__ == "__main__":
    main()
