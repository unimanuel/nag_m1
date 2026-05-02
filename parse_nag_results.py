from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo


METRIC_CANDIDATES = {
    "mpsnr": (
        "test/epoch/MPSNR",
        "train/epoch/MPSNR",
        "eval/epoch/MPSNR",
        "test/epoch/PSNR",
        "train/epoch/PSNR",
        "eval/epoch/PSNR",
    ),
    "ssim": (
        "test/epoch/SSIM",
        "train/epoch/SSIM",
        "eval/epoch/SSIM",
    ),
    "lpips": (
        "test/epoch/LPIPS",
        "train/epoch/LPIPS",
        "eval/epoch/LPIPS",
    ),
}

MEAN_METRIC_CANDIDATES = {
    "mpsnr": (
        "test/epoch/MeanMPSNR",
        "train/epoch/MeanMPSNR",
        "eval/epoch/MeanMPSNR",
        "test/epoch/MeanPSNR",
        "train/epoch/MeanPSNR",
        "eval/epoch/MeanPSNR",
    ),
    "ssim": (
        "test/epoch/MeanSSIM",
        "train/epoch/MeanSSIM",
        "eval/epoch/MeanSSIM",
    ),
    "lpips": (
        "test/epoch/MeanLPIPS",
        "train/epoch/MeanLPIPS",
        "eval/epoch/MeanLPIPS",
    ),
}

WINDOWS_PATH_RE = re.compile(r"^(?P<drive>[A-Za-z]):[\\/](?P<rest>.*)$")
TIMESTAMP_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an Excel summary of NAG mean quality metrics for all result "
            "folders inside a parent directory."
        )
    )
    parser.add_argument(
        "results_dir",
        help="Directory containing multiple NAG result folders.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output .xlsx path. Defaults to <results_dir>/nag_results_summary.xlsx.",
    )
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.exists():
        return path.resolve()

    match = WINDOWS_PATH_RE.match(path_str)
    if match and os.name == "posix":
        drive = match.group("drive").lower()
        rest = match.group("rest").replace("\\", "/")
        converted = Path("/mnt") / drive / rest
        return converted.resolve()

    return path.resolve()


def load_training_config(config_path: Path) -> dict[str, Any]:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    config = data.get("NAGConfig", {})
    return config if isinstance(config, dict) else {}


def load_tracker_metrics(tracker_path: Path) -> dict[str, Any]:
    data = json.loads(tracker_path.read_text(encoding="utf-8"))
    metrics = data.get("metrics", {})
    return metrics if isinstance(metrics, dict) else {}


def unwrap_path_value(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        nested_value = value.get("value")
        if isinstance(nested_value, str):
            return nested_value
        if isinstance(nested_value, dict):
            return unwrap_path_value(nested_value)
    return None


def extract_metric_csv_path(
    run_dir: Path,
    metric_name: str,
    metrics: dict[str, Any],
    candidate_map: dict[str, tuple[str, ...]],
    fallback_name: str,
) -> Path:
    for tag in candidate_map[metric_name]:
        metric_entry = metrics.get(tag)
        if not isinstance(metric_entry, dict):
            continue

        values = metric_entry.get("values", {})
        if not isinstance(values, dict):
            continue

        csv_path = unwrap_path_value(values.get("file_path"))
        if not csv_path:
            continue

        resolved_path = Path(csv_path)
        if resolved_path.exists():
            return resolved_path

    tracker_dir = run_dir / "tracker"
    return tracker_dir / fallback_name


def read_scalar_metric_series(csv_path: Path) -> pd.Series | None:
    if not csv_path.exists():
        return None

    frame = pd.read_csv(csv_path, sep=";")
    if "value" not in frame.columns:
        return None

    values = pd.to_numeric(frame["value"], errors="coerce").dropna()
    if values.empty:
        return None

    return values


def parse_metric_array(value: Any) -> np.ndarray | None:
    if isinstance(value, (list, tuple, np.ndarray)):
        array = np.asarray(value, dtype=float)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None

        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            normalized = text.strip("[]").replace("\n", " ")
            array = np.fromstring(normalized, sep=" ")
        else:
            array = np.asarray(parsed, dtype=float)
    elif pd.isna(value):
        return None
    else:
        array = np.asarray([value], dtype=float)

    array = np.asarray(array, dtype=float).reshape(-1)
    array = array[~np.isnan(array)]
    return array if array.size else None


def read_metric_distribution(csv_path: Path) -> np.ndarray | None:
    if not csv_path.exists():
        return None

    frame = pd.read_csv(csv_path, sep=";")
    if "value" not in frame.columns or frame.empty:
        return None

    for value in reversed(frame["value"].tolist()):
        parsed = parse_metric_array(value)
        if parsed is not None:
            return parsed

    return None


def extract_metric_stats(run_dir: Path, metric_name: str, metrics: dict[str, Any]) -> dict[str, float | None]:
    mean_csv = extract_metric_csv_path(
        run_dir,
        metric_name,
        metrics,
        MEAN_METRIC_CANDIDATES,
        f"train_epoch_Mean{metric_name.upper()}.csv",
    )
    scalar_values = read_scalar_metric_series(mean_csv)

    distribution_csv = extract_metric_csv_path(
        run_dir,
        metric_name,
        metrics,
        METRIC_CANDIDATES,
        f"train_epoch_{metric_name.upper()}.csv",
    )
    distribution = read_metric_distribution(distribution_csv)

    if scalar_values is None and distribution is None:
        return {"mean": None, "median": None, "min": None, "max": None, "source": None}

    mean_value = float(scalar_values.iloc[-1]) if scalar_values is not None else float(distribution.mean())
    if distribution is None:
        return {
            "mean": mean_value,
            "median": mean_value,
            "min": mean_value,
            "max": mean_value,
            "source": str(mean_csv),
        }

    return {
        "mean": mean_value,
        "median": float(np.median(distribution)),
        "min": float(distribution.min()),
        "max": float(distribution.max()),
        "source": str(distribution_csv),
    }


def classify_background(config: dict[str, Any], run_dir_name: str) -> str:
    background_type = str(config.get("background_plane_type", "")).lower()
    if "background_image_plane" in background_type:
        return "plane"
    if "background_image_sphere" in background_type:
        return "sphere"

    lowered_name = run_dir_name.lower()
    if "bgplane" in lowered_name or lowered_name.endswith("_plane"):
        return "plane"
    return "sphere"


def strip_timestamp_prefix(name: str) -> str:
    return TIMESTAMP_PREFIX_RE.sub("", name)


def sortable_name(name: str) -> str:
    return re.sub(r"\d+", lambda match: f"{int(match.group()):010d}", name)


def derive_dataset_label(config: dict[str, Any], run_dir_name: str, background: str) -> str:
    name = str(config.get("name") or strip_timestamp_prefix(run_dir_name))
    if background == "plane":
        name = re.sub(r"_(bgplane|plane)$", "", name)
    return name


def extract_data_path(config: dict[str, Any]) -> str | None:
    data_path = config.get("data_path")
    if isinstance(data_path, dict):
        return unwrap_path_value(data_path)
    if isinstance(data_path, str):
        return data_path
    return None


def collect_run_summary(run_dir: Path) -> dict[str, Any] | None:
    config_path = run_dir / "training_config.yaml"
    tracker_path = run_dir / "tracker" / "tracker.json"

    if not config_path.exists():
        return None

    config = load_training_config(config_path)
    background = classify_background(config, run_dir.name)

    metrics: dict[str, Any] = {}
    if tracker_path.exists():
        metrics = load_tracker_metrics(tracker_path)

    mpsnr = extract_metric_stats(run_dir, "mpsnr", metrics)
    ssim = extract_metric_stats(run_dir, "ssim", metrics)
    lpips = extract_metric_stats(run_dir, "lpips", metrics)

    return {
        "dataset": derive_dataset_label(config, run_dir.name, background),
        "background": background,
        "result_folder": run_dir.name,
        "data_path": extract_data_path(config),
        "mean_mpsnr": mpsnr["mean"],
        "median_mpsnr": mpsnr["median"],
        "min_mpsnr": mpsnr["min"],
        "max_mpsnr": mpsnr["max"],
        "mean_ssim": ssim["mean"],
        "median_ssim": ssim["median"],
        "min_ssim": ssim["min"],
        "max_ssim": ssim["max"],
        "mean_lpips": lpips["mean"],
        "median_lpips": lpips["median"],
        "min_lpips": lpips["min"],
        "max_lpips": lpips["max"],
    }


def build_summary(results_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for child in sorted(results_dir.iterdir()):
        if not child.is_dir():
            continue

        row = collect_run_summary(child)
        if row is not None:
            rows.append(row)

    if not rows:
        raise SystemExit(f"No NAG result folders found in: {results_dir}")

    frame = pd.DataFrame(rows)
    background_order = {"sphere": 0, "plane": 1}
    frame["_background_order"] = frame["background"].map(background_order).fillna(99)
    frame["_dataset_order"] = frame["dataset"].map(sortable_name)
    frame["_result_folder_order"] = frame["result_folder"].map(sortable_name)
    frame = frame.sort_values(
        by=["_dataset_order", "_background_order", "_result_folder_order"],
        kind="stable",
    ).drop(columns=["_background_order", "_dataset_order", "_result_folder_order"])
    return frame[
        [
            "dataset",
            "background",
            "result_folder",
            "data_path",
            "mean_mpsnr",
            "median_mpsnr",
            "min_mpsnr",
            "max_mpsnr",
            "mean_ssim",
            "median_ssim",
            "min_ssim",
            "max_ssim",
            "mean_lpips",
            "median_lpips",
            "min_lpips",
            "max_lpips",
        ]
    ]


def autosize_columns(worksheet) -> None:
    for column_cells in worksheet.columns:
        max_length = 0
        column_letter = get_column_letter(column_cells[0].column)
        for cell in column_cells:
            value = "" if cell.value is None else str(cell.value)
            max_length = max(max_length, len(value))
        worksheet.column_dimensions[column_letter].width = min(max_length + 2, 60)


def write_excel(summary: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary.to_excel(writer, index=False, sheet_name="summary")

        worksheet = writer.book["summary"]
        worksheet.freeze_panes = "A2"

        for cell in worksheet[1]:
            cell.font = Font(bold=True)

        table_ref = f"A1:{get_column_letter(worksheet.max_column)}{worksheet.max_row}"
        table = Table(displayName="NAGResultsSummary", ref=table_ref)
        table.tableStyleInfo = TableStyleInfo(
            name="TableStyleMedium2",
            showFirstColumn=False,
            showLastColumn=False,
            showRowStripes=True,
            showColumnStripes=False,
        )
        worksheet.add_table(table)
        autosize_columns(worksheet)


def main() -> None:
    args = parse_args()
    results_dir = resolve_path(args.results_dir)

    if not results_dir.exists():
        raise SystemExit(f"Results directory does not exist: {results_dir}")
    if not results_dir.is_dir():
        raise SystemExit(f"Results path is not a directory: {results_dir}")

    output_path = (
        resolve_path(args.output)
        if args.output
        else (results_dir / "nag_results_summary.xlsx").resolve()
    )

    summary = build_summary(results_dir)
    write_excel(summary, output_path)
    print(f"Wrote {len(summary)} rows to {output_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
