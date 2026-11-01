#!/usr/bin/env python3
"""
paper_numbers.py

Prints every number the paper reports, computed straight from the run folders,
so nothing in the paper is copied by hand.

    python paper_numbers.py                 # reads runs/<experiment> for each experiment
    python paper_numbers.py --runs other    # a different runs folder
    python paper_numbers.py > numbers.txt   # save it

Missing run folders are skipped. Needs only numpy.
"""
import sys as _sys
_sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import argparse
import json
from pathlib import Path

import numpy as np

PROMPT_NAMES = ["lake", "bicycle", "lighthouse", "market", "letter", "sky"]


def load(d):
    p = d / "summary.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def name(i):
    return PROMPT_NAMES[i - 1] if 1 <= i <= len(PROMPT_NAMES) else f"prompt {i}"


def width_bin(w):
    if w is None:
        return "n/a"
    return "under 50" if w <= 50 else ("hundreds (50-999)" if w < 1000 else "thousands (1,000+)")


def header(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def single_prompt(s, title, ref_key):
    header(title)
    print(f"{'condition':22s} {'steps':>5s} {'ended':28s} {'distinct':>8s} {'rep3':>6s} {'plain':>6s} "
          f"{'width mean/max':>15s} {'1st diff':>8s} {'sim L14':>7s}")
    for c, d in s["conditions"].items():
        sim = (d.get(f"similarity_to_{ref_key}_by_layer") or {}).get("layer 14")
        wm, wx = d.get("mean_effective_tokens_fed"), d.get("max_effective_tokens_fed")
        w = f"{wm:.1f}/{wx:.0f}" if wm is not None else "-"
        fd = d.get(f"first_step_differing_from_{ref_key}")
        print(f"{c:22s} {d['steps']:5d} {d['how_it_ended'][:28]:28s} {d['distinct_word_ratio']:8.2f} "
              f"{d['repeated_3gram_share']:6.0%} {d['plain_english_char_share']:6.0%} {w:>15s} "
              f"{'-' if fd is None else fd:>8} {'-' if sim is None else f'{sim:.2f}':>7}")


def multi(s, d):
    header("REPLICATION: collapse removed for the whole reply, six prompts")
    m = np.load(d / "metrics.npz")
    rows = {(r["prompt"], r["condition"]): r for r in s["runs"]}
    print(f"{'prompt':11s} {'median width':>12s} {'peak':>7s} {'1st step >50':>12s} {'distinct':>8s} {'rep3':>6s} "
          f"{'plain':>6s} {'sim':>5s}  ended")
    for i in range(1, len(s["prompts"]) + 1):
        r = rows[(i, "soft")]
        eff = m[f"p{i}_soft_eff"]
        over = np.nonzero(eff > 50)[0]
        print(f"{name(i):11s} {np.median(eff):12.1f} {eff.max():7.0f} {('never' if not len(over) else int(over[0])):>12} "
              f"{r['distinct_word_ratio']:8.2f} {r['repeated_3gram_share']:6.0%} {r['plain_english_char_share']:6.0%} "
              f"{r['similarity_to_greedy_mid_layer']:5.2f}  {r['how_it_ended']}")
    for c in ("greedy", "soft"):
        n = sum(rows[(i, c)]["how_it_ended"] != "step limit" for i in range(1, len(s["prompts"]) + 1))
        print(f"{c}: ended its own reply in {n} of {len(s['prompts'])}")


def window(s, title, show_runs=True):
    header(title)
    runs = [r for r in s["runs"] if r["condition"] != "greedy"]
    conds = list(dict.fromkeys(r["condition"] for r in runs))
    print(f"{'condition':14s} {'recovered':>9s} {'went under':>10s} {'mean peak width':>15s} {'mean length ratio':>17s} "
          f"{'min plain after':>15s}")
    for c in conds:
        rs = [r["after_window"] for r in runs if r["condition"] == c]
        pk = [a.get("max_blend_width_in_window") for a in rs if a.get("max_blend_width_in_window") is not None]
        lr = [a.get("blend_length_ratio_mean") for a in rs if a.get("blend_length_ratio_mean") is not None]
        pe = [a.get("plain_english_char_share") for a in rs if a.get("plain_english_char_share") is not None]
        print(f"{c:14s} {sum(a.get('recovered', False) for a in rs):5d}/{len(rs):<3d} "
              f"{sum(a.get('went_under', False) for a in rs):6d}/{len(rs):<3d} "
              f"{(np.mean(pk) if pk else float('nan')):15.0f} {(np.mean(lr) if lr else float('nan')):17.3f} "
              f"{(min(pe) if pe else float('nan')):15.1%}")
    if show_runs:
        print(f"\n{'prompt':11s} {'condition':14s} {'peak width':>10s} {'len ratio':>9s} {'words':>5s} {'plain':>6s} "
              f"{'rep3':>5s} {'prompt words':>12s} {'overlap':>7s} {'rec':>4s}  ended")
        for r in runs:
            a = r["after_window"]
            pk, lr = a.get("max_blend_width_in_window"), a.get("blend_length_ratio_mean")
            print(f"{name(r['prompt']):11s} {r['condition']:14s} {('-' if pk is None else f'{pk:.0f}'):>10s} "
                  f"{('-' if lr is None else f'{lr:.3f}'):>9s} {a['words']:5d} "
                  f"{(a['plain_english_char_share'] or 0):6.0%} {(a['repeated_3gram_share'] or 0):5.0%} "
                  f"{a['prompt_words_used']:5.2f}/{a['greedy_prompt_words_used']:<6.2f} "
                  f"{a['content_overlap_with_greedy']:7.2f} {('yes' if a.get('recovered') else 'no'):>4s}  "
                  f"{r['how_it_ended']}")
    return runs


def bins(runs, title):
    print(f"\n{title}: recovery by peak blend width in the window")
    out = {}
    for r in runs:
        a = r["after_window"]
        b = width_bin(a.get("max_blend_width_in_window"))
        n, k = out.get(b, (0, 0))
        out[b] = (n + 1, k + int(bool(a.get("recovered"))))
    for b in ("under 50", "hundreds (50-999)", "thousands (1,000+)", "n/a"):
        if b in out:
            print(f"  {b:20s} runs {out[b][0]:3d}   recovered {out[b][1]:3d}")


def length_by_width(runs, title):
    print(f"\n{title}: blend length relative to the tokens it was made from (1.0 = no shrinkage)")
    groups = {}
    for r in runs:
        a = r["after_window"]
        if a.get("blend_length_ratio_min") is None:
            continue
        groups.setdefault(width_bin(a.get("max_blend_width_in_window")), []).append(a["blend_length_ratio_min"])
    for b, v in groups.items():
        print(f"  {b:20s} runs {len(v):3d}   lowest ratio in window: mean {np.mean(v):.3f}, min {np.min(v):.3f}")


def countdown(d, title):
    s = load(d)
    if not s:
        return
    header(title)
    print(f"how it ended: {s.get('how_it_ended')}; tokens written: {s.get('response_tokens')}; mantras: {s.get('mantras')}")
    for c in s.get("per_cycle", []):
        er, eo = c.get("entropy_at_rollovers"), c.get("entropy_at_ordinary_steps")
        print(f"cycle {c['cycle']}: numbers counted {c['numbers_counted']}, errors {c['countdown_errors']}, "
              f"entropy at rollovers {er} vs ordinary {eo}" + (f" ({er / eo:.0f}x)" if er and eo else "")
              + f", step similarity L14 rollovers {c.get('step_similarity_layer14_rollovers')} "
              f"vs ordinary {c.get('step_similarity_layer14_ordinary')}")
    for k in ("stop_prob_after_each_mantra", "mantra_vs_previous_similarity_layer14"):
        v = s.get(k)
        if v:
            print(f"{k}: first {v[0]}, peak {max(v)} at #{int(np.argmax(v)) + 1}, last {v[-1]} ({len(v)} values)")
    m = d / "metrics.npz"
    if m.exists():
        p = np.load(m)["p_end_after"]
        print(f"final-layer probability of ending at the last written token: {p[-1]:.3f}")
    lens = d / "lens.csv"
    if lens.exists():
        import csv
        rows = list(csv.DictReader(lens.open(encoding="utf-8")))
        eot = [r for r in rows if r["L21_top1"] in ("<|endoftext|>", "<|im_end|>")]
        nl = [r for r in eot if r["token"] == "\u23ce"]
        pr = [float(r["L21_top1_prob"]) for r in eot]
        print(f"logit lens: layer 21's top prediction is end-of-text at {len(eot)} of {len(rows)} positions, "
              f"{len(nl)} of them right after a line break; median probability {np.median(pr) if pr else float('nan'):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(Path(__file__).resolve().parent / "runs"))
    R = Path(ap.parse_args().runs)

    s = load(R / "collapse_open")
    if s:
        single_prompt(s, "EXPERIMENT 1: collapse removed, lake prompt (float32)", "greedy")
    s = load(R / "collapse_bf16")
    if s:
        single_prompt(s, "PRECISION CHECK: same, bfloat16", "greedy")
        m = np.load(R / "collapse_bf16" / "metrics.npz")
        eff = m["soft_effective_tokens"]
        over = np.nonzero(eff > 100)[0]
        print(f"soft (bf16): peak width {eff.max():.0f}; first step over 100 tokens: "
              f"{'never' if not len(over) else int(over[0])}")
    s = load(R / "ablate_open")
    if s:
        single_prompt(s, "CONTROL 1: silencing layers, commitment intact (lake prompt)", "intact")
    if load(R / "multi"):
        multi(load(R / "multi"), R / "multi")
    s = load(R / "wake")
    if s:
        window(s, "WAKE TEST: window steps 40-59, doses 1.0 / 1.5 / random real tokens")
    sweep = load(R / "sweep")
    if sweep:
        runs = window(sweep, "DOSE SWEEP: blended window at six doses", show_runs=False)
        bins(runs, "Sweep")
        length_by_width(runs, "Sweep")
    s = load(R / "matched")
    if s:
        runs = window(s, "CONTROL 2: matched sampling (commit to a token drawn from the same distribution)",
                      show_runs=False)
        bins(runs, "Matched")
        flagged = [f"{name(r['prompt'])} {r['condition']}" for r in runs if not r["after_window"].get("recovered")]
        print(f"  not counted as recovered: {', '.join(flagged) or 'none'}")
    s = load(R / "rescaled")
    if s:
        runs = window(s, "CONTROL 3: blended, length restored", show_runs=True)
        bins(runs, "Rescaled")
        if sweep:
            print("\nSweep vs rescaled, recovered per dose (same prompt, same dose):")
            sw = {(r["prompt"], r["condition"].split("_")[1]): r["after_window"].get("recovered")
                  for r in sweep["runs"] if r["condition"] != "greedy"}
            for r in runs:
                k = (r["prompt"], r["condition"].split("_")[1])
                if k in sw:
                    print(f"  {name(r['prompt']):11s} dose {k[1]}: blended {'yes' if sw[k] else 'no ':3s}  "
                          f"length restored {'yes' if r['after_window'].get('recovered') else 'no'}")
    countdown(R / "countdown_1000", "APPENDIX: countdown from 1,000 (+ logit lens)")
    countdown(R / "countdown_100", "APPENDIX: countdown from 100")


if __name__ == "__main__":
    main()
