# Background extension testing

Run validation in the established NAG environment. The tests use synthetic
geometry, temporary datasets, and mocked training calls; they do not launch
full training. CUDA/model imports still need NAG's normal dependencies.

## Optional test dependencies

The optional Poetry `test` group supplies pytest, PyMuPDF, and ReportLab. The
PDF packages are needed by reporting tests, not by the sphere or ellipsoid
renderer. They should not be installed into the historical fitting environment
as an incidental effect of validation.

For a new development environment, include the test group when installing
NAG's normal dependencies:

```bash
poetry install --with dev,test --extras depth --extras torch --extras post-torch
```

For an existing training environment, install optional test packages into an
isolated directory and add that directory to `PYTHONPATH` only for testing.
This avoids replacing NumPy, SciPy, PyTorch, or the raw-v4 interpreter used by
recorded experiments.

## Test commands

From the repository root in the NAG environment:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest tests -q -p no:cacheprovider
```

In the Waymo processing environment:

```bash
PYTHONDONTWRITEBYTECODE=1 libs/nag-waymo-processing/.venv/bin/python \
  -m unittest discover -s libs/nag-waymo-processing/tests -v
```

The generalized-preparation worker regression uses a temporary dataset,
calibration, and ellipsoid fixture. A moved archive or unavailable training
folder must not change whether that worker check passes.

The saved seven-interval preparation certificate is a separate regression
record. Unit tests do not replace it. Its production release gate must accept
the current code identities, recipe, fixture hash, and numerical policy before
static preparation or training can proceed.
