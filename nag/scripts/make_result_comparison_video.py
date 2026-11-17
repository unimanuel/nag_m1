#!/usr/bin/env python3
"""Make a synchronized, labeled side-by-side MP4 from two completed NAG runs.

CPU-only; requires Pillow, PyYAML, FFmpeg and FFprobe, not Torch or checkpoints.
The displayed PSNR is the selected run's final scalar, not an aggregate score.
"""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from PIL import Image, ImageDraw, ImageFont
import yaml


def executable(name: str, explicit: str | None = None, neighbor: str | None = None) -> str:
    if explicit:
        resolved = shutil.which(explicit)
        if not resolved:
            raise ValueError(f'Explicit {name} executable is not available: {explicit}')
        return str(Path(resolved).resolve())
    candidates = []
    if neighbor:
        candidates.append(str(Path(neighbor).parent / name))
    candidates += [shutil.which(name), str(Path(sys.executable).parent / name),
                   str(Path.home() / 'anaconda3/envs/nag1/bin' / name)]
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return str(Path(candidate).resolve())
    raise ValueError(f'{name} not found. Activate an environment containing it or pass --{name}.')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unwrap(value):
    while isinstance(value, dict):
        if 'value' not in value:
            raise ValueError('Cannot resolve saved dataset path')
        value = value['value']
    return str(value)


def final_psnr(path: Path, epoch: int) -> float:
    with path.open(newline='') as handle:
        rows = list(csv.DictReader(handle, delimiter=';'))
    matching = [row for row in rows if float(row['step']) == epoch]
    if len(matching) != 1:
        raise ValueError(f'Expected exactly one epoch-{epoch} scalar PSNR in {path}')
    value = float(matching[0]['value'])
    if not math.isfinite(value):
        raise ValueError(f'Nonfinite PSNR in {path}')
    return value


def read_run(directory: Path, title: str) -> dict:
    directory = directory.resolve()
    config_path = directory / 'training_config.yaml'
    config = yaml.safe_load(config_path.read_text())['NAGConfig']
    exit_path = directory / 'exit_000.txt'
    if not exit_path.read_text().startswith('Exit code: 0.'):
        raise ValueError(f'Run did not complete successfully: {directory}')
    selection = config['frame_indices_filter']
    start, stop, step = selection['start'], selection['stop'], selection.get('step') or 1
    if not isinstance(start, int) or not isinstance(stop, int) or step <= 0 or stop <= start:
        raise ValueError(f'Unsupported/empty frame slice: {selection}')
    dataset = unwrap(config['data_path']).replace('\\', '/')
    segment = Path(dataset).name
    if not segment.startswith('segment-') or not segment[8:].isdigit():
        raise ValueError(f'Cannot identify Waymo segment from {dataset}')
    frames = list(range(start, stop, step))
    video = directory / 'final/complete_scene.mp4'
    if not video.is_file():
        raise ValueError(f'Missing complete-scene video: {video}')
    tracker = directory / 'tracker/train_epoch_MeanMPSNR.csv'
    epoch = int(config['max_epochs'])
    uv = bool(config.get('background_plane_kwargs', {}).get('sphere_visible_uv_texture', {}).get('enabled', False))
    return {'run': str(directory), 'video': str(video), 'segment': segment,
            'frames': frames, 'title': title, 'seed': config.get('seed'), 'epoch': epoch,
            'psnr': final_psnr(tracker, epoch), 'psnr_tracker': str(tracker),
            'background_class': config['background_plane_type'], 'visible_uv': uv,
            'config_sha256': sha(config_path), 'tracker_sha256': sha(tracker),
            'image_source': unwrap(config['images_path']).replace('\\', '/'),
            'render_resolution_factor': config.get('final_plot_resolution_factor')}


def probe(video: Path, ffprobe: str) -> dict:
    command = [ffprobe, '-v', 'error', '-select_streams', 'v:0', '-count_frames',
               '-show_entries', 'stream=width,height,avg_frame_rate,nb_read_frames',
               '-of', 'json', str(video)]
    data = json.loads(subprocess.check_output(command, text=True))
    if len(data['streams']) != 1:
        raise ValueError(f'Expected one primary video stream: {video}')
    stream = data['streams'][0]
    fps = Fraction(stream['avg_frame_rate'])
    if fps <= 0:
        raise ValueError(f'Invalid FPS in {video}')
    return {'width': int(stream['width']), 'height': int(stream['height']),
            'frames': int(stream['nb_read_frames']), 'fps': str(fps)}


def validate_pair(left: dict, right: dict, fps: float | None) -> Fraction:
    for field in ('segment', 'frames', 'image_source', 'render_resolution_factor', 'epoch'):
        if left[field] != right[field]:
            raise ValueError(f'Inputs differ in {field}; refusing an unaligned comparison')
    for item in (left, right):
        if item['probe']['frames'] != len(item['frames']):
            raise ValueError(f"Video does not contain exactly one frame per selected index: {item['video']}")
    if (left['probe']['width'], left['probe']['height']) != (right['probe']['width'], right['probe']['height']):
        raise ValueError('Input video dimensions differ')
    if fps is not None:
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError('--fps must be finite and positive')
        return Fraction(str(fps)).limit_denominator(100000)
    if left['probe']['fps'] != right['probe']['fps']:
        raise ValueError('Source frame rates differ; provide --fps for explicit frame-index retiming')
    return Fraction(left['probe']['fps'])


def choose_font(explicit: Path | None) -> Path:
    candidates = [explicit] if explicit else []
    candidates += [Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'),
                   Path('/mnt/c/Windows/Fonts/arialbd.ttf')]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ValueError('No suitable font found; pass --font /path/to/font.ttf')


def make_header(path: Path, labels: list[str], width: int, height: int, font: Path):
    image = Image.new('RGB', (width * 2, height), 'black')
    draw = ImageDraw.Draw(image)
    size = min(32, height // 2)
    while size >= 10:
        face = ImageFont.truetype(str(font), size=size)
        if all(draw.textbbox((0, 0), label, font=face)[2] <= width - 24 for label in labels):
            break
        size -= 1
    else:
        raise ValueError('Labels are too long for this panel width; shorten titles or enlarge --panel-width')
    for index, label in enumerate(labels):
        box = draw.textbbox((0, 0), label, font=face)
        x = index * width + (width - (box[2] - box[0])) / 2 - box[0]
        y = (height - (box[3] - box[1])) / 2 - box[1]
        draw.text((x, y), label, fill='white', font=face)
    image.save(path)


def encode_command(ffmpeg, left, right, header, target, fps, width, height, threads, crf):
    # Rebuild time from frame indices so corresponding frames stay paired.
    # No shortest-input truncation: matching frame counts are validated first.
    retime = f'settb=AVTB,setpts=N/(({fps})*TB),scale={width}:{height}:flags=lanczos,setsar=1'
    graph = (f'[0:v]{retime}[l];[1:v]{retime}[r];'
             '[l][r]hstack=inputs=2[panels];'
             '[2:v]setsar=1[header];[header][panels]vstack=inputs=2:shortest=1[out]')
    return [ffmpeg, '-hide_banner', '-loglevel', 'warning', '-stats', '-nostdin',
            '-threads', str(threads), '-i', left['video'], '-threads', str(threads), '-i', right['video'],
            '-loop', '1', '-framerate', str(fps), '-i', str(header),
            '-filter_complex_threads', '1', '-filter_complex', graph, '-map', '[out]',
            '-an', '-frames:v', str(len(left['frames'])), '-r', str(fps),
            '-c:v', 'libx264', '-preset', 'medium', '-crf', str(crf), '-pix_fmt', 'yuv420p',
            '-threads', str(threads), '-movflags', '+faststart', '-n', str(target)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--left-run', type=Path, required=True)
    parser.add_argument('--right-run', type=Path, required=True)
    parser.add_argument('--left-title', default='PLANE')
    parser.add_argument('--right-title', default='ELLIPSOID')
    parser.add_argument('--sequence-label', help='Default: derived from the full segment ID and frame range')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fps', type=float, help='Retime playback by frame index; default: preserve source FPS')
    parser.add_argument('--panel-width', type=int, help='Default: native source width; both panels retain aspect ratio')
    parser.add_argument('--header-height', type=int, default=64)
    parser.add_argument('--threads', type=int, default=2, help='CPU encoder threads (default: 2)')
    parser.add_argument('--crf', type=int, default=18, help='H.264 quality, lower is higher quality (default: 18)')
    parser.add_argument('--font', type=Path)
    parser.add_argument('--ffmpeg', default=os.environ.get('FFMPEG_BIN'))
    parser.add_argument('--ffprobe', default=os.environ.get('FFPROBE_BIN'))
    parser.add_argument('--dry-run', action='store_true', help='Read/validate inputs and show plan; write nothing')
    args = parser.parse_args(argv)
    if args.threads < 1 or args.header_height < 24 or args.header_height % 2 or not 0 <= args.crf <= 51:
        raise ValueError('Require positive threads, even header height ≥24, and CRF in [0,51]')
    output = args.output.absolute()
    if output.suffix.lower() != '.mp4':
        raise ValueError('--output must end in .mp4')
    metadata = output.with_suffix('.json')
    preview = output.with_suffix('.preview.png')
    for path in (output, metadata, preview):
        if path.exists():
            raise ValueError(f'Refusing to overwrite {path}; choose a new --output filename')
    ffmpeg = executable('ffmpeg', args.ffmpeg)
    ffprobe = executable('ffprobe', args.ffprobe, ffmpeg)
    left, right = (read_run(p, t) for p, t in ((args.left_run, args.left_title), (args.right_run, args.right_title)))
    for item in (left, right):
        item['probe'] = probe(Path(item['video']), ffprobe)
    fps = validate_pair(left, right, args.fps)
    width = args.panel_width or left['probe']['width']
    if width < 160 or width % 2:
        raise ValueError('Panel width must be even and at least 160')
    height = max(2, round(left['probe']['height'] * width / left['probe']['width'] / 2) * 2)
    font = choose_font(args.font)
    sequence = args.sequence_label or f"W{left['segment'][8:11]} {left['frames'][0]}–{left['frames'][-1]}"
    labels = [f"{sequence} | {item['title']} | PSNR: {item['psnr']:.4f} dB" for item in (left, right)]
    plan = {'schema': 1, 'left': left, 'right': right, 'labels': labels,
            'psnr_definition': 'Selected run final-epoch mean PSNR from train_epoch_MeanMPSNR.csv; not across-repeat or per-frame PSNR.',
            'alignment': 'Same segment, source images, explicit frame indices and frame count; source order preserved.',
            'fps': str(fps), 'playback_seconds': len(left['frames']) / float(fps),
            'output_dimensions': [width * 2, height + args.header_height],
            'cpu_only': True, 'output': str(output), 'preview': str(preview), 'font': str(font)}
    for side, item, label in zip(('left', 'right'), (left, right), labels):
        print(f'[{side}] {label}\n  {item["run"]}', flush=True)
    print(f'[plan] {len(left["frames"])} synchronized frames | {fps} FPS | '
          f'{plan["playback_seconds"]:.2f} s | {width * 2}×{height + args.header_height}\n'
          f'[output] {output}', flush=True)
    if args.dry_run:
        print('[dry-run] No files or directories written.', flush=True)
        return plan
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='autowork_comparison_', dir=output.parent) as tmp:
        tmp = Path(tmp)
        header, video = tmp / 'header.png', tmp / 'comparison.mp4'
        make_header(header, labels, width, args.header_height, font)
        command = encode_command(ffmpeg, left, right, header, video, fps, width, height,
                                 args.threads, args.crf)
        print('[encode] CPU-only FFmpeg; no training or checkpoint loading', flush=True)
        subprocess.run(command, check=True)
        result = probe(video, ffprobe)
        if result != {'width': width * 2, 'height': height + args.header_height,
                      'frames': len(left['frames']), 'fps': str(fps)}:
            raise ValueError(f'Encoded output does not match the plan: {result}')
        thumbnail = tmp / 'preview.png'
        subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin', '-i', str(video),
                        '-frames:v', '1', '-threads', '1', str(thumbnail)], check=True)
        for item in (left, right):
            item['video_sha256'] = sha(Path(item['video']))
        plan['output_probe'] = result
        plan['output_sha256'] = sha(video)
        sidecar = tmp / 'comparison.json'
        sidecar.write_text(json.dumps(plan, indent=2) + '\n')
        # Exclusive publication: never replace existing files, even if another
        # invocation created a target after preflight. Inputs are never edited.
        for source, destination in ((video, output), (thumbnail, preview), (sidecar, metadata)):
            with destination.open('xb') as handle, source.open('rb') as reader:
                shutil.copyfileobj(reader, handle)
    print(f'[done] {output}\n[preview] {preview}\n[provenance] {metadata}', flush=True)
    return plan


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)
