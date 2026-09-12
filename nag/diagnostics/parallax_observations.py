"""Shared observation-pair cache with explicit sparse visibility evidence."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .parallax_core import (DepthEvidence, angle_degrees, balanced_indices, inside,
                            intrinsics, mask_at, project)
from .parallax_io import (SEQUENCES, code_signature, digest, input_files, load_mask,
                          snapshot, write_json)


def targets_for(source, poses, count):
    """Select nearby and distant frames plus heading diversity, independent of geometry."""
    n=len(poses)
    candidates=set()
    for offset in (1,3,8,16,32):
        candidates.update(j for j in (source-offset,source+offset) if 0<=j<n)
    baseline=np.linalg.norm(poses[:,:3,3]-poses[source,:3,3],axis=1)
    heading=angle_degrees(poses[:,:3,2],poses[source,:3,2])
    candidates.update([int(np.argmax(baseline)),int(np.argmax(heading))])
    candidates.discard(source)
    ordered=sorted(candidates,key=lambda j:(baseline[j],heading[j],j))
    if len(ordered)>count:
        ordered=[ordered[i] for i in np.linspace(0,len(ordered)-1,count).round().astype(int)]
    return ordered


def alignment_errors(raw_uv, projected_uv, raw_camera, size):
    """Convert Waymo's distorted pixel labels before checking NAG projection."""
    k=intrinsics(raw_camera, size)
    full=raw_camera['resolution']['value']
    pixels=raw_uv.astype(float)*np.asarray(size)/full
    k1,k2,k3,p1,p2=raw_camera['lens_distortion']
    undistorted=cv2.undistortPoints(pixels.reshape(-1,1,2),k,np.array([k1,k2,p1,p2,k3]),P=k).reshape(-1,2)
    return np.linalg.norm(undistorted-projected_uv,axis=1)


def check_prepared(pointer, repo):
    meta=json.loads(Path(pointer).read_text())
    if meta['code']!=code_signature(repo) or meta['inputs']!=snapshot([x['path'] for x in meta['inputs']]):
        raise ValueError('Observation inputs or diagnostic code changed; rerun prepare_ego_parallax_diagnostics.sh')
    return meta


def prepare_sequence(args, inv, label):
    segment,start,stop=SEQUENCES[label]
    data_dir=args.data_root/f'segment-{segment}'
    cloud_dir=args.cloud_root/f'segment-{segment}'
    run=next(r for r in inv['runs'] if r['sequence']==label)
    size=tuple(run['size'])
    frames=np.arange(start,stop)
    if args.frame_limit:
        frames=frames[np.unique(np.linspace(0,len(frames)-1,min(args.frame_limit,len(frames))).round().astype(int))]
    settings={k:getattr(args,k) for k in ('rays_per_frame','targets_per_frame','frame_limit','visibility_radius',
              'depth_absolute','depth_relative','alignment_p90_limit','seed')}
    inputs=snapshot(input_files(data_dir,cloud_dir,frames))
    meta={'schema':1,'sequence':label,'settings':settings,'inputs':inputs,'code':code_signature(args.repo),
          'frames':frames.tolist(),'size':size,'camera_policy':inv['camera_policy'],
          'mask_policy':'union of saved NAG mask layers in training-image coordinates; no semantic/static tracking',
          'scope':'smoke_subset' if len(frames)<stop-start else 'full_interval'}
    cache_key=digest(meta)
    out=args.output/label/'observations'/cache_key[:16]
    pointer=args.output/label/'prepared.json'
    if (out/'summary.json').exists() and not args.force:
        write_json(pointer,json.loads((out/'summary.json').read_text()))
        print(f'[reuse] {label} observations {cache_key[:16]}',flush=True)
        return
    out.mkdir(parents=True,exist_ok=True)
    # Reuse the established coordinate/pose alignment, but perform mask selection
    # at the reprojected undistorted pixels below instead of raw Waymo pixels.
    sys.path.insert(0,str(args.repo/'libs/nag-waymo-processing'))
    from nag_waymo_processing.ellipsoid import EllipsoidJob, load_front_ray_observations
    # Loading frames one at a time preserves the existing per-frame correction
    # convention while bounding memory and avoiding unused frames in smoke mode.
    camera=json.loads((data_dir/'waymo_pose_config_no_distortion.json').read_text())
    raw_camera=json.loads((data_dir/'waymo_pose_config.json').read_text())
    poses=np.asarray(camera['position'])[frames]
    k=intrinsics(camera,size)
    per_frame=[]
    counts=Counter()
    alignment=[]
    rng=np.random.default_rng(args.seed)
    for fi,frame in enumerate(frames):
        job=EllipsoidJob(f'segment-{segment}',int(frame),int(frame)+1,1,'waymo_pose_config_no_distortion.json',())
        obs=load_front_ray_observations(job,str(args.cloud_root),str(args.data_root),
                                       correct_pose_mismatches=True,background_only=False)
        points=obs.points
        uv,z=project(points,poses[fi],k)
        in_image=inside(uv,z,size)
        mask=load_mask(data_dir,int(frame),size)
        evidence=DepthEvidence(uv,z,size)
        errors=alignment_errors(obs.pixels[in_image],uv[in_image],raw_camera,size)
        p90=float(np.quantile(errors,.90)) if len(errors) else float('inf')
        alignment.append({'frame':int(frame),'median_px':float(np.median(errors)) if len(errors) else None,'p90_px':p90})
        if p90>args.alignment_p90_limit:
            write_json(out/'alignment_failure.json',alignment)
            raise ValueError(f'{label} frame {frame}: projected-vs-Waymo pixel P90={p90:.2f}px exceeds '
                             f'{args.alignment_p90_limit}px. Inspect calibration before changing the threshold.')
        foreground=mask_at(mask,uv)
        visible=evidence.classify(uv,z,args.visibility_radius,args.depth_absolute,args.depth_relative)==1
        eligible=in_image & ~foreground & visible
        candidates=np.flatnonzero(eligible)
        chosen=candidates[balanced_indices(uv[candidates],size,args.rays_per_frame,rng)]
        per_frame.append({'points':points[chosen],'uv':uv[chosen], 'mask':mask,'evidence':evidence})
        counts.update({'front_points':len(points),'outside_source_image':int((~in_image).sum()),
                       'source_foreground':int((in_image&foreground).sum()),
                       'source_depth_inconsistent':int((in_image&~foreground&~visible).sum()),
                       'eligible_source_points':int(eligible.sum()),'sampled_source_points':len(chosen)})
        if fi in (0,len(frames)//2,len(frames)-1):
            with Image.open(data_dir/'images_undistorted'/f'{frame:03d}.png') as image:
                image=image.convert('RGB').resize(size)
            arr=np.asarray(image).copy()
            arr[mask]=(arr[mask]*.65+np.array([220,40,180])*.35).astype(np.uint8)
            image=Image.fromarray(arr);draw=ImageDraw.Draw(image)
            for x,y in uv[chosen]:draw.ellipse((x-1,y-1,x+1,y+1),fill=(0,255,100))
            image.save(out/f'alignment_frame_{frame:03d}.jpg')
        print(f'[prepare] {label} frame {frame}: {len(chosen)} rays, alignment P90={p90:.2f}px',flush=True)
    arrays={key:[] for key in ('points','source','target','visibility','baseline','heading')}
    for i,source in enumerate(per_frame):
        p=source['points']
        for j in targets_for(i,poses,args.targets_per_frame):
            uv,z=project(p,poses[j],k)
            valid=inside(uv,z,size)
            foreground=mask_at(per_frame[j]['mask'],uv)
            counts['attempted_pairs']+=len(p)
            counts['target_invalid_or_outside']+=int((~valid).sum())
            counts['target_foreground']+=int((valid&foreground).sum())
            valid &= ~foreground
            status=per_frame[j]['evidence'].classify(uv,z,args.visibility_radius,args.depth_absolute,args.depth_relative)
            counts['target_occluded']+=int((valid&(status==2)).sum())
            counts['target_inconsistent']+=int((valid&(status==3)).sum())
            keep=valid & (status<=1)
            n=int(keep.sum())
            counts['supported_pairs']+=int((keep&(status==1)).sum())
            counts['unknown_pairs']+=int((keep&(status==0)).sum())
            for key,value in [('points',p[keep]),('source',np.full(n,i)),('target',np.full(n,j)),
                               ('visibility',status[keep]),('baseline',np.full(n,np.linalg.norm(poses[i,:3,3]-poses[j,:3,3]))),
                               ('heading',np.full(n,angle_degrees(poses[i,:3,2],poses[j,:3,2])))]:
                arrays[key].append(value)
    if not counts['supported_pairs']:
        raise ValueError(f'No supported cross-frame pairs for {label}; inspect alignment and sampling')
    arrays={key:np.concatenate(value) for key,value in arrays.items()}
    for key in ('source','target','visibility'):arrays[key]=arrays[key].astype(np.int32)
    source_points=np.concatenate([f['points'] for f in per_frame])
    source_ids=np.concatenate([np.full(len(f['points']),i,dtype=np.int32) for i,f in enumerate(per_frame)])
    np.savez_compressed(out/'pairs.npz',**arrays,poses=poses,intrinsics=k,frames=frames,
                        source_points=source_points,source_ids=source_ids)
    meta.update({'key':cache_key,
                 'directory':str(out),'counts':dict(counts),'alignment':alignment})
    write_json(out/'summary.json',meta)
    write_json(pointer,meta)
    print(f'[prepared] {label}: {counts["supported_pairs"]} supported, {counts["unknown_pairs"]} unknown pairs',flush=True)
