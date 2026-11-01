# One Word at a Time: An Examination of Commitment and the Serial Collapse

Code and run outputs for the paper of the same name (Nathan Daniel Nixon, 2026).

At each step, a language model scores every possible next token, and one token is picked and fed back as the next input: the serial collapse. These experiments take that step away from Qwen2.5-1.5B-Instruct. Instead of one committed token, the model gets back a probability-weighted blend of all candidate tokens, either for a whole reply or for a 20-step window. Then they compare that with controls that keep commitment but disrupt the model in other ways. The network's weights are never changed in the collapse experiments.

## What's here

| File | What it does |
| --- | --- |
| `collapse_test.py` | Experiment 1: the collapse removed on one prompt, at several blend temperatures, plus the bfloat16 precision check |
| `ablate_test.py` | Control 1: silencing layers with commitment intact |
| `wake_test.py` | The six-prompt replication, the wake test, the dose sweep, Control 2 (matched sampling) and Control 3 (blend length restored) |
| `qwen_meditation.py` | Appendix: the countdown-and-mantra runs |
| `logit_lens.py` | Appendix: per-layer readout of the countdown run |
| `paper_numbers.py` | Prints every number reported in the paper, computed from the run folders |
| `make_figure.py` | Draws Figure 1 (`figures/figure1.png`) from the run folders |
| `run_all.bat` | Reruns everything in order (Windows), then writes `runs/numbers.txt` |
| `runs/` | The outputs the paper reports: generated texts, `summary.json`, plots and per-step metrics for each experiment |

## Reproducing

Everything runs on CPU. A laptop with 16 GB of RAM is enough; the full set took about 1 hour 20 minutes on the laptop used for the paper.

```
python -m pip install -r requirements.txt
run_all.bat
```

Then `python make_figure.py` redraws the figure. On macOS or Linux, run the commands listed in `run_all.bat` one at a time. The model downloads from Hugging Face the first time (about 3 GB). To use a local copy instead, set `QWEN_MODEL` to its folder first:

```
set QWEN_MODEL=C:\path\to\qwen2.5-1.5b-instruct
```

Notes:

- **Use float32** (the default). Blended inputs are not real tokens, so rounding errors compound in them; in bfloat16 they break down much sooner. The precision check shows this.
- **Greedy runs are deterministic.** The sampled arms use a fixed seed per prompt (`--seed`, default 0). Exact numbers can still shift slightly across CPUs and library versions, because the blended runs amplify tiny floating-point differences. `runs/environment.txt` and `runs/pip_freeze.txt` record the versions used for the published runs.
- `wake_test.py --help` and the docstring at the top of each script explain every option.

## License

Code: MIT (see `LICENSE`). Paper text and figures: CC BY 4.0.

## Citation

> Nixon, N. D. (2026). *One Word at a Time: An Examination of Commitment and the Serial Collapse.*
