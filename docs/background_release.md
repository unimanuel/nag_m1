# Background extension release review

The release-cleanup branch keeps the sphere and ellipsoid implementations,
Visible-UV mapping, preparation pipeline, matched configurations, diagnostics,
and experiment launchers. It does not change the numerical recipe or repair
the geometries used by completed experiments. In particular, camera enclosure
checks do not guarantee foreground-atlas enclosure; the known foreground
clipping limitation remains explicit rather than changing past results.

## Git scope

Review `release/backgrounds-main-cleanup` against `main`. The original feature
branch remains available. Generated PDFs, one-off PDF scripts, and saved result
summaries have recoverable external copies and are removed from the cleanup
branch, not from earlier history. Executable analysis tools and provenance
inputs remain under `reports/`; they are required by several launchers.

Do not include unrelated local scripts, notes, monitoring output, or existing
working-tree edits in the release commits. This cleanup does not authorize
merging into `main`, rewriting existing history, or publishing any repository.

## Submodules

The `tools` submodule pins commit
`e8c5b21716cd76386de37a2a0e7f8c396b08d3f0` in
`https://github.com/unimanuel/tools_m1`. Its configured tracking branch is
`main`, which contains that commit. Normal submodule checkout uses the pinned
commit, not the tracking branch; avoid `git submodule update --remote` when
reproducing experiments.

The Waymo submodule pins commit
`82493a93c83ecc4e53adf4b050fd7d350a55b0dd`. Its configured repository
`https://github.com/jp-schneider/nag-waymo-processing.git` was not accessible
during release review, and no accessible `unimanuel/nag-waymo-processing` fork
was available. A local checkout is not proof that a fresh GitHub clone works.

Before publication, make that exact Waymo commit available in an accessible
repository, update `.gitmodules` if the URL changes, and test a clone without
local URL overrides. Publishing the Waymo repository must precede publishing
the parent pointer. Fork creation and pushing remain a separate approval step;
the cleanup does not silently vendor the dependency or point at a nonexistent
fork. The optional nested segmentation dependency has its own setup and is not
needed by the point-cloud/ellipsoid unit tests.

## Verification

Run the complete repository and Waymo tests using [the testing guide](testing.md).
The optional reporting dependencies can be supplied from a temporary directory
without updating the historical training or raw-v4 environments.

Check shell syntax and source whitespace without treating generated run
artifacts as implementation files:

```bash
git diff --check main...HEAD -- . ':!runs/**'
git diff --name-only -z main...HEAD -- '*.sh' |
  while IFS= read -r -d '' script; do bash -n "$script" || exit; done
```

The release certificate must still accept all seven ego intervals in both
verification layers. The exact compatibility ledgers account only for reviewed
non-numerical changes; they do not waive future fitting, model, or observation
changes. The comparison-video allowance preserves the old certificate and
manifests without pretending a numerical replay was rerun.

The fixed, hash-backed historical line-ending ledger also permits Git's
LF-only checkout of 14 older mixed-ending source files. Its tests reconstruct
the exact historical bytes from the recorded CRLF ranges. This is not a
general training-code exemption or an on-disk source rewrite.

A temporary local clone can verify the committed tree and submodule objects
using local source repositories. Such a test must be labeled local-only: it
cannot discharge the separate remote-publication check above.

After review and publication approval, integrate the approved branch into the
fork's `main` and push only to `unimanuel/nag_m1`, not to the upstream project.
