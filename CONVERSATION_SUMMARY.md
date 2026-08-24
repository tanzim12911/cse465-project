# Color Illusion Pipeline: Conversation Record

## Objective

Improve performance on the **Color Illusion** task from ColorBench, while
preserving a clean held-out evaluation. The original generic multi-task setup
was narrowed to Color Illusion only.

## Initial diagnosis

The adaptive pipeline often reduced ColorBench held-out accuracy because it
learned noisy, generic prompt rules from very small probes.

- ColorBench contains **93** Color Illusion examples.
- Earlier 15-example evaluations were too variable: one changed answer moves
  accuracy by 6.7 percentage points.
- In the previous verifier, a candidate was accepted when it did not hurt a
  probe set (`delta >= 0`), so neutral rules were committed.
- The same VLM generated, judged, and updated rules; this made incorrect visual
  observations likely to become persistent playbook guidance.

## ColorBench run 13

Run 13 confirmed the regression:

| Held-out result | Score |
| --- | ---: |
| Baseline | 17/30 (56.7%) |
| Adaptive pipeline | 11/30 (36.7%) |
| Delta | -20.0 points |

The regression was concentrated in comparison questions:

| Subtype | Baseline | Adaptive |
| --- | ---: | ---: |
| Comparison | 12/21 | 6/21 |
| Uniformity | 5/6 | 5/6 |
| Ranking | 0/3 | 0/3 |

The comparison playbook changed six baseline-correct answers into incorrect
answers. This means ColorBench should remain a final evaluation target while
the adaptation strategy is developed elsewhere.

## Implemented pipeline changes

### Color Illusion specialization

- Removed the runnable multi-task selector.
- Kept only the Color Illusion pipeline and its three subtypes: `uniformity`,
  `comparison`, and `ranking`.
- Updated generator, reflector, curator, evaluation, and notebook wording to
  focus on colour comparison under context, shadows, gradients, and background
  contrast.

### Strict candidate validation

The verifier now:

1. Clones the current playbook.
2. Simulates the actual Curator ADD/UPDATE operation.
3. Scores the committed candidate state against the current state on a fixed,
   same-subtype probe set.
4. Accepts only a candidate that adds at least one correct probe answer.

Neutral candidates and candidates that would not actually be committed are
rejected. The Reflector also credits only playbook bullets explicitly cited by
the Generator.

### Runtime reduction

`SOLVER_MAX_NEW_TOKENS` was reduced from 512 to 64 because baseline, held-out,
and verifier solver calls only need a final option letter.

Run 13 still required approximately 428 VLM calls because candidate validation
uses both with-rule and without-rule probes. The verifier is the main reason
the run took about an hour instead of roughly 30 minutes.

### Image resolution decision

The 1,003,520-pixel cap is a Qwen visual-token budget, not a batch-size limit.
Qwen converts images to visual tokens; a full-resolution image can create many
more tokens and substantially increase T4 memory use and latency. Targeted
high-resolution crops are preferable to sending every full image at raw
resolution, but neither ColorBench nor public RCID provides target-region
coordinates, so no generic cropper was added.

## RCID support

The pipeline can now select either dataset:

```bash
python run_pipeline.py --dataset colorbench --mode both --num_adaptation 60 --num_eval 30 --seed 42
python run_pipeline.py --dataset rcid --mode both --num_adaptation 60 --num_eval 30 --seed 42
```

The public Hugging Face RCID release provides images plus `same`/`different`
class labels, not the image-specific VQA questions from the full research
dataset. The adapter converts each label to a binary comparison question:

> Do the two target regions appear to have different colors?

ColorBench and RCID have separate split-cache names and dataset-prefixed output
files, preventing accidental mixing or logger resume collisions.

## RCID run 14

Run 14 was encouraging:

| RCID held-out result | Score |
| --- | ---: |
| Baseline | 11/30 (36.7%) |
| Adaptive pipeline | 15/30 (50.0%) |
| Delta | +13.3 points |

The adaptive system corrected seven baseline mistakes and introduced three new
mistakes, a net gain of four. It accepted three of 25 candidate rules and
rejected 22.

The strongest retained rule was:

> Compare the color histograms of the target regions while accounting for
> background contrast and lighting variations.

This is currently prompt guidance only: the code does **not** calculate actual
colour histograms. The RCID result should be repeated over several fixed seeds
before treating it as reliable.

## Colab notebook

`ColorBench_Adaptive_Skills.ipynb` was updated with:

```python
RUN_TAG = "run-15-rcid"
DATASET = "rcid"  # or "colorbench"
```

It writes results to seed-specific folders such as
`results/<RUN_TAG>/seed_42/`, preventing overlap between independent runs.

## RCID download limitation

The current RCID loader uses normal Hugging Face loading and therefore downloads
the entire public split before selecting the requested 60 adaptation and 30
evaluation samples. This is an implementation limitation, not an experimental
requirement. The public RCID split is approximately 22.8 GB.

The next data-loading improvement is RCID streaming: deterministically choose
the requested examples before image decoding so a 60/30 experiment downloads
only its selected images.

## Recommended next steps

1. Implement streaming RCID subset selection.
2. Repeat RCID baseline versus adaptive evaluation over seeds 7, 13, and 42.
3. Report mean and spread, not one 30-example split.
4. Add an actual pixel-measurement tool only after establishing how target
   regions can be localized reliably.
5. Keep ColorBench Color Illusion as the separate final evaluation target.
