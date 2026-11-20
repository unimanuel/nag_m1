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
working-tree edits in the release commits. The initial cleanup did not
authorize publication. The owner separately approved publication on
20 November 2026; that approval covers the owner's repositories, not upstream
pushes or rewriting existing history.

## Submodules

The `tools` submodule pins commit
`e8c5b21716cd76386de37a2a0e7f8c396b08d3f0` in
`https://github.com/unimanuel/tools_m1`. Its configured tracking branch is
`main`, which contains that commit. Normal submodule checkout uses the pinned
commit, not the tracking branch; avoid `git submodule update --remote` when
reproducing experiments.

The Waymo submodule pins commit
`82493a93c83ecc4e53adf4b050fd7d350a55b0dd` in
`https://github.com/unimanuel/nag-waymo-processing.git`, on the existing
`feature/waymo-pointcloud-extraction` branch. The owner created this repository
after the original upstream URL proved inaccessible. The exact pinned commit
was published there before the parent release branch, and `.gitmodules` now
uses that accessible URL. The existing local submodule retains the author's
repository as `upstream`; no upstream push is part of this release.

A fresh GitHub checkout must retrieve those exact pinned objects without
local source repositories or URL overrides. The nested segmentation dependency
remains pinned to `29800f32e546141409c721cd01a0f990036e4867` in
`https://github.com/jp-schneider/sam2.git`. It has separate optional setup and
is not needed by the point-cloud/ellipsoid unit tests, but its remote checkout
is included in publication validation.

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

## Local validation results

The cleanup was validated on 20 November 2026 in the existing NAG environment,
with optional PDF-test packages supplied from an isolated temporary directory.
The training and raw-v4 environments were not upgraded.

| Check | Result |
| --- | --- |
| Complete repository suite in the research checkout | 356 passed |
| Separate local source checkout with locally supplied submodules | 353 passed, 3 expected integration skips |
| Waymo point-cloud and ellipsoid tests | 15 passed |
| Saved preparation certificate in both checkout forms | Accepted, seven cached and seven end-to-end records |
| Source whitespace and changed shell-script syntax | Passed |
| Generated-artifact archive | 18 exact copies verified before removal |

The local source checkout omitted upstream static media and notebooks, and did
not contain the private `.env`, datasets, run archives, or submodule virtual
environment. Its three skips concern two completed-cohort integration checks
and one historical processing-interpreter check. The saved preparation
certificate was checked against the existing external fixture records; the
cleanup did not rerun numerical fits or training.

Those initial local results did not establish that recursive cloning from
GitHub worked. Publication received separate approval; the remote validation
below closes that dependency-availability gap without changing the numerical
recipe, saved geometries, or original feature branch.

## GitHub publication validation

The release branch was cloned directly from `unimanuel/nag_m1`, with Git's
system/global configuration disabled and no local URL overrides, object
alternates, or reference repositories. The parent used a partial, sparse
source checkout omitting static media, notebooks, and run archives. Both
submodules and the nested SAM2 submodule were initialized recursively from
their published GitHub URLs at the exact commits listed above.

Historical source blobs needed by the controlled-replay tests were fetched
before the offline test rerun: partial clones defer those blobs, and a
network-restricted test process cannot fetch them lazily. No test expectation,
source code, compatibility tolerance, or dependency version was changed to
accommodate this checkout.

| Check on remote source commit `8d8a462` | Result |
| --- | --- |
| Repository suite | 353 passed, 3 expected integration skips |
| Waymo point-cloud and ellipsoid suite | 15 passed |
| Saved preparation certificate | Accepted, seven cached and seven end-to-end records |
| Recursive dependency checkout | Waymo, tools, and nested SAM2 at their exact pins |

Only this publication record is updated after that test commit; the numerical
source, submodule pins, and compatibility signatures are unchanged.

The fresh source checkout has no private `.env`, downloaded datasets, completed
training folders, or historical processing virtual environment. Its three
expected integration skips concern two completed-cohort checks and one
historical-interpreter check. Verification reuses the established Python
environments; this is not a dependency-installation or training benchmark.
