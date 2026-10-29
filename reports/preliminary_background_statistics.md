# Preliminary background results: inter-frame and across-repeat variability

The generator reads the fixed main paper cohort: 7 ego intervals and the
original 20 final intervals, five treatments and three seed-42 repeats each
(405 trainings). Restored-depth batches are deliberately not incorporated
piecemeal. W203 60-107 and W975 0-68, 69-99 and 99-162 remain labelled pending
in this preliminary snapshot; W324 96-161 is separately excluded for resources.
Historical PCA/AR3/depth-proxy ablations, upright poles and double-resolution
treatments are not part of this five-treatment main table.

Run with the reporting Python (NumPy, PyYAML and PyMuPDF installed), not with
an environment modification to the frozen NAG training interpreter:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 \
  reports/generate_preliminary_background_statistics.py
```

Default artifacts are on D:, outside Git:

```text
D:\CGProjectData\waymo\diagnostics\autowork_preliminary_background_statistics_2026-10-29\
  preliminary_background_results.txt
  preliminary_background_results.pdf
  runs.csv
  summary.json
```

Choose a new `--output-root /.../autowork_NAME` for a repeat export. Existing
reports are never overwritten. This is not a training, rendering or fitting
script and does not require images to be decoded or checkpoints to be loaded.

## Statistical conventions

The source vectors are the epoch-80 `tracker/train_epoch_{MPSNR,SSIM,LPIPS}.csv`
arrays. Official means remain the corresponding `train_epoch_Mean*.csv`
scalars. Each vector must have exactly one finite value per selected frame,
and its mean must agree with the official scalar within rounding tolerance.

For each run, frame SD uses all its frames and the population divisor F.
For each interval/treatment, average its three official means and separately
average its three frame SDs. Thus main tables show **mean +/- averaged
inter-frame SD**. Across-repeat SD is the sample SD of the three official
run means (divisor 2), in explicitly separate tables. No pooled frame/run SD,
SD of interval means, confidence interval or standard error is substituted.

Primary aggregates give intervals equal weight; scene rows average intervals
within a parent, and the scene-balanced alternative gives those parent rows
equal weight. This follows the inter-frame concept in NAG supplementary C.1
and Tables 3/11, with our normalization and repeated-run extension explicitly
stated rather than claiming an undocumented denominator matches the paper.

The text and PDF include both main groups, scene-balanced robustness,
individual parent scenes, every individual interval/treatment, separate repeat
SDs and a complete 405-run appendix. `summary.json` preserves SHA256 source
evidence and logical/resolved archive paths; `runs.csv` is machine-readable.

Frame SD measures variation of reconstruction quality across frames, not
flicker directly. Same-seed repeats measure nondeterminism, not the full
independent-seed distribution. Intervals from the same parent are dependent.

## Validation of the preliminary snapshot

The 2026-10-29 export verified all 405 selected runs and all 1,215 final
per-frame metric vectors. Its official aggregate means reproduce the previous
comparison within 2e-14. The largest absolute scalar/vector-mean differences
from logger rounding were 5.015e-6 dB, 1.624e-7 SSIM and 5.588e-8 LPIPS.
The provenance inventory contains 3,387 source-file SHA256 entries.

All 37 PDF pages carry the preliminary marker. All 4,109 text spans are black
and inside their page boundaries; aggregate, dense interval and run-appendix
pages were also visually inspected. No training or checkpoint loading was run.

The 11 reporting tests pass with the reporting Python:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/tests" \
  python3 -m unittest test_preliminary_background_statistics
```
