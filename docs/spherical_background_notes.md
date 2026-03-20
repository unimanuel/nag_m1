# Spherical Background Notes

This file exists to preserve context across chat sessions for the spherical
background work in NAG, especially the blur-reduction improvements.

## Scope

These notes are about replacing the planar background with a spherical
background and improving the resulting image quality without changing the main
training/rendering behavior unnecessarily.

## Blur Reduction Suggestions

The main suggestions that were identified for reducing spherical background
blurriness were:

1. Increase sphere init resolution.
2. Stop uniform averaging across all frames during sphere initialization.
3. Use more robust stitching/fusion than plain averaging.
4. Respect lens-distortion handling explicitly in sphere init.
5. Super-sample sphere texels during init.
6. Longer-term: consider a different spherical storage/parameterization, such
   as cubemap-style storage.

## Current Status

### Implemented

1. Higher default sphere init resolution.
   - The sphere texture resolution is no longer fixed at `512x1024`.
   - It now scales from the available image resolution, while keeping the old
     minimum floor.

2. Better sphere color initialization.
   - The sphere no longer initializes from one flat mean color.
   - It now builds a stitched equirectangular initialization by projecting
     sphere texels into the source camera images.

3. Stop uniform averaging across all frames.
   - The sphere init now uses best-view selection/weighting instead of simply
     averaging all valid frame samples per texel.
   - Current weighting prefers samples near image center and with better
     incidence angle.

4. Sphere flow export.
   - The exported sphere background flow is no longer a hard-coded zero tensor.
   - The current implementation exports learned sphere texture-space flow and
     reprojects it to image-space flow in the modality path.

### Not Yet Implemented

1. Super-sampling sphere texels during init.
   - Proposed implementation: sample several subpoints inside each texel’s UV
     cell, convert them to sphere directions/points, sample colors from source
     images, and combine them into one final texel color.
   - Reason this may help: one texel represents an area patch on the sphere,
     not a single exact point. Center-only sampling can alias detail,
     especially near edges and mask boundaries.

2. More robust fusion than current best-view selection.
   - Possible options:
     - weighted top-k blending
     - robust mean / trimmed mean
     - confidence-weighted splatting

3. Explicit lens-distortion-aware handling in sphere init.
   - The planar init has more direct handling for this path than the current
     spherical init.

4. Alternative storage/parameterization.
   - Equirectangular is simple and consistent with the current code, but has
     distortion.
   - Cubemap-style storage is a longer-term architectural option, not a small
     patch.

## Super-Sampling Clarification

The key point is:

- A stored texel still ends with exactly one RGB value.
- Super-sampling only changes how that one RGB value is computed during init.

For example, with 4 sub-samples:

1. Pick 4 subpoints inside one equirectangular texel.
2. Convert each subpoint to a sphere direction / 3D point.
3. Reproject each one into source images.
4. Sample 4 colors.
5. Combine them into one final texel color.

This is not only about view direction. It also affects sampled RGB, mask
validity, and which view/source sample is chosen for a texel.

## Files Most Relevant To This Work

- `nag/model/background_image_sphere_scene_node_3d.py`
- `nag/model/view_dependent_background_image_sphere_scene_node_3d.py`
- `nag/model/sphere_utils.py`
- `nag/model/nag_functional_model.py`

## Useful Prompt For Future Sessions

If starting a new chat, point Codex to this file and ask it to read it before
making more spherical background changes.
