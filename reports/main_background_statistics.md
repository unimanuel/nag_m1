# Completed main background benchmark

The 1 November 2026 exporter joins the original 405-run main cohort to the
60 restored W203/W975 jobs, without changing historical manifests, fitting
code, training code or preliminary report artifacts. It requires exactly
465 completed jobs: 7 ego and 24 final intervals, five treatments and three
fresh seed-42 repeats each. W324 96–161 remains a resource/runtime exclusion.

Run with the reporting Python (NumPy, PyYAML and PyMuPDF):

```bash
PYTHONDONTWRITEBYTECODE=1 python3 reports/generate_main_background_statistics.py
```

The default export is separate from the historical preliminary snapshot:

```text
D:\CGProjectData\waymo\diagnostics\autowork_main_background_statistics_2026-11-01\
  main_background_results.pdf
  main_background_results.txt
  main_results_tables.md
  interval_means_wide.csv
  parent_scene_means_wide.csv
  interval_statistics.csv
  paired_comparisons.csv
  runs.csv
  summary.json
  ablation_plan.md
```

Use `--output-root /.../autowork_NEW --report-date YYYY-MM-DD` for another
export. Existing output roots are rejected; no report, source run or failed
attempt is overwritten. Standard YAML saved-config parsing can use PyYAML's
C safe loader for reporting speed; no numerical fitting or training interpreter
is changed.

Compact interval tables put `PSNR / SSIM / LPIPS` in each treatment column.
The PDF/text additionally provide interval-balanced and parent-scene-balanced
aggregates, every parent/interval with averaged frame SD, separate sample
across-repeat SDs, paired win/delta summaries and all run-level results.
The tidy CSV preserves separate metric/mean/frame-SD/repeat-SD fields for
typesetting. JSON pins source/report SHA256 evidence and logical/resolved
archive paths. Every page says the main benchmark is complete and ablations
are planned; no formerly missing-depth interval is still labelled pending.

Means and SD definitions are unchanged from the preliminary report. Frame SD
uses divisor F within each run and is averaged over the three repeats;
repeat SD uses divisor 2 on the three official final tracker means. Primary
aggregates weight intervals equally; scene-balanced rows weight parents
equally. Neither is a confidence interval, an independence claim, a direct
flicker measurement or a held-out novel-view evaluation.

The [ablation proposal](../docs/paper_ablation_matrix_2026-11-01.md) is copied
into the artifact folder and summarized in the PDF. It proposes a matched
regular-mapping depth-only/reprojection/floor ladder on five predeclared
development intervals. It does not implement or run new fitting/training;
in particular, zero-weight D0 needs an ablation-only interface because the
released fitter currently requires positive reprojection/depth weights.

Regression tests cover the preserved preliminary defaults, new manifest join,
strict complete-cohort checks, metric order, LPIPS comparison direction,
standard safe-parser equivalence, protected existing outputs and every PDF
page's status, black text and page bounds:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/tests" \
  python3 -m unittest test_preliminary_background_statistics test_main_background_statistics
```

## Export verification

The completed export contains 465 unique runs and all 1,395 finite final
per-frame metric vectors, with 3,887 source-file provenance entries. All
statistics for the original 405 runs, including both original aggregation
schemes, reproduce the preliminary snapshot exactly. Restored results add
60 jobs without changing those historical values.

All 19 reporting tests pass. The 51-page PDF has 5,080 black text spans,
all within page boundaries. Aggregate, compact individual and ablation-plan
pages were visually inspected. Every page distinguishes the complete main
benchmark from planned ablations; no pending-depth label remains. Generated
artifacts stay on D:, outside Git; no fitting or training was run.
