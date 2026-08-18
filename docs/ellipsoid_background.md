# Waymo ellipsoid background

Milestone 3 uses the PCA ellipsoid produced for one YAML frame interval as
NAG's actual background surface. It does not approximate the ellipsoid with a
sphere: rays are intersected analytically with the oriented ellipsoid, texture
coordinates are evaluated on the normalized local unit sphere, and
view-dependent normals use the true ellipsoid gradient.

The original sphere and plane YAML files remain unchanged so they can continue
to serve as experiment baselines. An ellipsoid run selects the new background
class and matching geometry JSON through CLI overrides.

## Required inputs

The four scene-115 geometry files must exist under:

```text
/mnt/d/CGProjectData/waymo/ellipsoids/segment-12511696717465549299/
```

Every run validates that the JSON sequence and exclusive frame slice match the
selected YAML. The fitted radii are expanded uniformly by 2% by default. NAG
then checks that every initial camera center is strictly inside the effective
ellipsoid before creating the background.

## Training commands

Run these commands from `/mnt/c/Coding/CGProject/nag1` in the normal `nag1`
environment. Each command starts a full training run; they are intentionally
not part of the automated validation suite.

Frames 0--40:

```bash
python nag/scripts/run_nag.py \
  --config-path config/waymo/final/segment-12511696717465549299_0_40_large-undistorted.yaml \
  --background-plane-type nag.model.view_dependent_background_image_ellipsoid_scene_node_3d.ViewDependentBackgroundImageEllipsoidSceneNode3D \
  --background-ellipsoid-path /mnt/d/CGProjectData/waymo/ellipsoids/segment-12511696717465549299/frames_000_040/ellipsoid.json \
  --background-ellipsoid-scale-margin 0.02 \
  --name W125_0_40_large-undistorted_ellipsoid
```

Frames 40--93:

```bash
python nag/scripts/run_nag.py \
  --config-path config/waymo/final/segment-12511696717465549299_40_93_large-undistorted.yaml \
  --background-plane-type nag.model.view_dependent_background_image_ellipsoid_scene_node_3d.ViewDependentBackgroundImageEllipsoidSceneNode3D \
  --background-ellipsoid-path /mnt/d/CGProjectData/waymo/ellipsoids/segment-12511696717465549299/frames_040_093/ellipsoid.json \
  --background-ellipsoid-scale-margin 0.02 \
  --name W125_40_93_large-undistorted_ellipsoid
```

Frames 93--124:

```bash
python nag/scripts/run_nag.py \
  --config-path config/waymo/final/segment-12511696717465549299_93_124_large-undistorted.yaml \
  --background-plane-type nag.model.view_dependent_background_image_ellipsoid_scene_node_3d.ViewDependentBackgroundImageEllipsoidSceneNode3D \
  --background-ellipsoid-path /mnt/d/CGProjectData/waymo/ellipsoids/segment-12511696717465549299/frames_093_124/ellipsoid.json \
  --background-ellipsoid-scale-margin 0.02 \
  --name W125_93_124_large-undistorted_ellipsoid
```

Frames 124--149:

```bash
python nag/scripts/run_nag.py \
  --config-path config/waymo/final/segment-12511696717465549299_124_149_large-undistorted.yaml \
  --background-plane-type nag.model.view_dependent_background_image_ellipsoid_scene_node_3d.ViewDependentBackgroundImageEllipsoidSceneNode3D \
  --background-ellipsoid-path /mnt/d/CGProjectData/waymo/ellipsoids/segment-12511696717465549299/frames_124_149/ellipsoid.json \
  --background-ellipsoid-scale-margin 0.02 \
  --name W125_124_149_large-undistorted_ellipsoid
```

## Diagnostics

At initialization, the run writes
`initial/background_diagnostics/ellipsoid_geometry.txt` inside its output
directory. It records the source JSON, original and effective radii, center,
rotation, scale margin, and the maximum normalized camera radius. A failure
before training normally means that the wrong interval JSON was paired with a
YAML or that the safety margin is too small to contain the camera trajectory.

The ellipsoid class supports the same learned equirectangular texture flow and
view-dependent appearance path as the sphere. The native UV flow is mapped
back onto the ellipsoid surface and reprojected through the active camera when
NAG exports image-space flow modalities.
