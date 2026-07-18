#!/usr/bin/env python3
import argparse
import logging  # noqa
import os
import sys
from typing import Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TOOLS_ROOT = os.path.join(REPO_ROOT, "tools")

# Force the in-repo `nag` and companion `tools` package to win import resolution.
# Without this, `python nag/scripts/run_nag.py ...` can pick up an older installed
# `site-packages/nag`, which makes local debugging confusing because edits in this
# checkout would not affect the code path the script actually executes.
for path in (REPO_ROOT, TOOLS_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import matplotlib
import matplotlib.pyplot as plt
from tools.logger.logging import basic_config, logger

from nag.config.nag_config import NAGConfig

from tools.util.package_tools import set_module_path
from tools.context.script_execution import ScriptExecution
plt.ioff()
matplotlib.use('agg')


def config():
    from tools.mixin.argparser_mixin import set_warning_on_unsupported_type
    set_warning_on_unsupported_type(False)
    set_module_path(os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")))
    basic_config()


def get_config() -> NAGConfig:

    parser = argparse.ArgumentParser(
        description='Performs a SceneNSF training run.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argv = sys.argv[1:]
    cli_sphere_radius_from_visible_plane_corners = \
        _parse_cli_sphere_radius_from_visible_plane_corners(argv)
    cli_sphere_radius_from_visible_plane_centers = \
        _parse_cli_sphere_radius_from_visible_plane_centers(argv)
    parser_argv = _strip_cli_only_sphere_radius_flags(argv)
    original_sys_argv = sys.argv
    try:
        sys.argv = [sys.argv[0], *parser_argv]
        config: NAGConfig = NAGConfig.parse_args(parser)
    finally:
        sys.argv = original_sys_argv

    cli_sphere_radius_override = _parse_cli_sphere_radius_override(argv)
    cli_sphere_radius_mode_count = sum([
        cli_sphere_radius_from_visible_plane_corners,
        cli_sphere_radius_from_visible_plane_centers,
        cli_sphere_radius_override is not None or config.background_sphere_radius_override is not None,
    ])
    if cli_sphere_radius_mode_count > 1:
        raise ValueError(
            "Only one spherical background radius mode can be active. Do not "
            "combine --background-sphere-radius-from-visible-plane-corners, "
            "--background-sphere-radius-from-visible-plane-centers, "
            "background_sphere_radius_override, or "
            "--background-sphere-radius-override."
        )
    if cli_sphere_radius_override is not None:
        config.background_sphere_radius_override = cli_sphere_radius_override
    if cli_sphere_radius_from_visible_plane_corners:
        setattr(config, "background_sphere_radius_from_visible_plane_corners", True)
    if cli_sphere_radius_from_visible_plane_centers:
        setattr(config, "background_sphere_radius_from_visible_plane_centers", True)

    return config


def _parse_cli_sphere_radius_override(argv) -> Optional[float]:
    flag = "--background-sphere-radius-override"
    for idx, arg in enumerate(argv):
        if arg == flag:
            if idx + 1 >= len(argv):
                raise ValueError(f"{flag} requires a float value.")
            return float(argv[idx + 1])
        if arg.startswith(flag + "="):
            return float(arg.split("=", 1)[1])
    return None


def _parse_cli_sphere_radius_from_visible_plane_corners(argv) -> bool:
    flag = "--background-sphere-radius-from-visible-plane-corners"
    for arg in argv:
        if arg == flag:
            return True
        if arg.startswith(flag + "="):
            raise ValueError(f"{flag} is a boolean flag and does not take a value.")
    return False


def _parse_cli_sphere_radius_from_visible_plane_centers(argv) -> bool:
    flag = "--background-sphere-radius-from-visible-plane-centers"
    for arg in argv:
        if arg == flag:
            return True
        if arg.startswith(flag + "="):
            raise ValueError(f"{flag} is a boolean flag and does not take a value.")
    return False


def _strip_cli_only_sphere_radius_flags(argv):
    flags = {
        "--background-sphere-radius-from-visible-plane-corners",
        "--background-sphere-radius-from-visible-plane-centers",
    }
    return [arg for arg in argv if arg not in flags]


def main(config: NAGConfig):
    logging.info(f"Setup: {config.name}")
    logger.info(f"Using output directory: {str(config.output_path)}")
    # Setup
    from nag.run.nag_runner import NAGRunner
    runner = NAGRunner(config)
    runner.build()

    # Save config and log it
    cfg_file = runner.store_config(str(config.output_path))

    logger.info(f"Stored config in: {cfg_file}")
    # Training
    logger.info(f"Start training of: {config.name}")
    with plt.ioff():
        runner.train()


if __name__ == "__main__":
    config()
    cfg = get_config()
    with ScriptExecution(cfg):
        main(cfg)
