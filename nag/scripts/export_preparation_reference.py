"""Native sphere initialization only: deliberately no Trainer/build/train calls."""
import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'nag'))
sys.path.insert(1, str(REPO/'nag/scripts'))
from preparation.jobs import PreparationJob
from preparation.reference import export_reference
from preparation.storage import write_json


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job-json', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    d = json.loads(args.job_json.read_text())
    job = PreparationJob(**{f.name: d[f.name] for f in fields(PreparationJob)})
    write_json(args.output/'sphere_reference.json', export_reference(job, args.output))
