# Fixed-geometry ellipsoid texture-orientation experiment

The experiment reuses the v4 Moderate-Y ellipsoids for W254 37–81 and 86–137.
Only the texture coordinate mapping changes. The source JSON and PLY files,
runtime center, radii, physical rotation, ray intersections, surface normals,
view-dependent tangent-ray basis, texture budget, and training YAML are retained.
These are regular-layout runs; the view-dependent appearance network remains
enabled as in the earlier regular runs. No LiDAR extraction or fitting is needed.

## Run in WSL

Activate the existing `nag1` Python environment, then:

```bash
cd /mnt/c/Coding/CGProject/nag1
DRY_RUN=1 bash run_W254_fixed_geometry_texture_orientation.sh
bash run_W254_fixed_geometry_texture_orientation.sh
```

The default produces two new runs ending in
`_ellipsoid_v4_moderate_y_texture_world_y_auto_seam_regular`.
Use the ordinary NAG output folder for the current date. Both JSON files are
validated before training starts, and their center and radii are printed.

For a six-run ablation, run the following **instead of** the default command:

```bash
TEXTURE_MODES="legacy world_y world_y_auto_seam" \
  bash run_W254_fixed_geometry_texture_orientation.sh
```

This runs each of the following on both intervals:

| Mode | Polar axis | Longitude seam |
| --- | --- | --- |
| `legacy` | Second normalized local ellipsoid axis | Existing mapping |
| `world_y` | Physical world Y | Fixed world-Z-based texture frame |
| `world_y_auto_seam` | Physical world Y | Automatic from all sampled frames |

`world_y` includes the horizontal reference change needed to define the new
coordinate frame. Comparing it with `world_y_auto_seam` isolates the additional
longitude offset. The same seed and other settings come from the existing YAML;
those settings do not guarantee bitwise deterministic GPU training.

## Coordinate mapping

For world position p, retain physical center c, rotation R, and radii r. The
normalized local surface coordinate is s = diag(1/r) R^T (p-c). Texture basis B
is separate from R. Its vertical column is the normalized vector
diag(1/r) R^T [0,1,0]. Its forward column is obtained from world Z in the same
space, orthogonalized against the vertical column, with world X as fallback.
The right column completes a right-handed orthonormal frame.

Map B^T s to equirectangular coordinates and add a periodic longitude offset.
The inverse first removes that offset, reconstructs the unit direction d, then
returns c + R diag(r) B d. Thus remapping a hit to UV and back returns the same
physical hit. Moving a UV by flow still produces a point on the same ellipsoid.
World Y designates the vertical axis; the name does not assume that +Y is up.

Auto seam samples a 73-by-49 grid covering the image footprint at every selected
camera time, including image borders. It intersects those rays with the existing
ellipsoid and bins longitudes into 360 angular bins. The seam is placed in the
middle of the largest empty circular arc. With complete angular coverage, it
uses the least occupied five-bin neighborhood. The rule is deterministic and
does not depend on scene identity, turn count, or manual street labels. Sampled
gaps are estimates, not guarantees that every rendered pixel avoids the seam.

The same forward/inverse mapping is used for runtime texture queries, flowed
surface reconstruction, image-to-texture initialization, and visible-UV layout
calculation. The latter is supported but disabled in this test's regular YAMLs.
Legacy is the configuration default, preserving the previous mapping and buffer
layout for old experiments. New texture basis buffers are stored in new-mode
checkpoints; construction derives the seam offset from saved configuration and
camera inputs. Changing texture modes while resuming a trained checkpoint is not
part of this experiment: start a fresh training run for each mode.

## Inspect after running

Each run's `initial/background_diagnostics/ellipsoid_geometry.txt` includes
`texture_orientation`, with the selected mode, texture basis, longitude offset,
sample count, and sampled fractions within 10 degrees of a pole or 5 degrees of
the seam for both the legacy and selected mappings. Physical geometry fields
should match the previous Moderate-Y run. No change should appear in the PLY.

Compare final PSNR, SSIM, and LPIPS with the existing regular Moderate-Y runs in
`runs/2026-09-02_depth_proxy_v4`. Also inspect image regions with high error:
lower pole/seam exposure alone is not proof of better reconstruction. If the
geometry is fixed and quality improves, that supports texture parameterization
as one source of difficulty. A single run cannot establish the size of the gain
independently of training variation.
