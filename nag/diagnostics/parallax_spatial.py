"""Spatial MAE comparisons from saved renders, independent of pixel_error folders."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .parallax_core import image_statistics, regions, signed_error_rgb
from .parallax_io import (SEQUENCES, code_signature, digest, load_mask, mask_paths,
                          snapshot, write_csv, write_json)


def spatial_inputs(args,runs,label,frames):
    data_dir=args.data_root/f'segment-{SEQUENCES[label][0]}'
    files=[data_dir/'masks',data_dir/'images_undistorted']
    for frame in frames:
        files.append(data_dir/'images_undistorted'/f'{frame:03d}.png')
        files+=mask_paths(data_dir,frame)
        for run in runs:
            index=run['frames'].index(frame)
            files.append(Path(run['path'])/'final/complete'/f'{index:02d}_t.png')
    return snapshot(files)


def spatial_key(args,runs,label,frames):
    return {'runs':runs,'frames':frames,'inputs':spatial_inputs(args,runs,label,frames),
            'settings':{'boundary_width':args.boundary_width,'display_cap':args.display_cap},
            'code':code_signature(args.repo)}


def rgb(path,size):
    with Image.open(path) as im:
        return np.asarray(im.convert('RGB').resize(size,Image.Resampling.BILINEAR))


def region_metrics(count,mae_sum,mse_sum):
    mae=mae_sum/count if count else None
    mse=mse_sum/count if count else None
    return {'pixels':count,'mae':mae,'rmse':float(np.sqrt(mse)) if count else None,
            'psnr':float(-10*np.log10(mse)) if count and mse>0 else None,
            'perfect_match':bool(count and mse==0)}


def spatial_sequence(args,inv,label):
    runs=[r for r in inv['runs'] if r['sequence']==label]
    frames=runs[0]['frames']
    if args.frame_limit:
        frames=[frames[i] for i in np.unique(np.linspace(0,len(frames)-1,min(args.frame_limit,len(frames))).round().astype(int))]
    provenance=spatial_key(args,runs,label,frames)
    key=digest(provenance)
    out=args.output/label/'spatial'/key[:16]
    if (out/'summary.json').exists() and not args.force:
        write_json(args.output/label/'spatial.json',json.loads((out/'summary.json').read_text()))
        print(f'[reuse] {label} spatial',flush=True);return
    out.mkdir(parents=True,exist_ok=True)
    data_dir=args.data_root/f'segment-{SEQUENCES[label][0]}'
    size=tuple(runs[0]['size'])
    comparisons=[];missing=[]
    for run in runs:
        if run['cohort']=='benchmark' and run['family']!='sphere':
            reference=next((r for r in runs if r['family']=='sphere' and r['mode']==run['mode'] and r['cohort']=='benchmark'),None)
            if reference:
                comparisons.append((run['id'],reference['id']))
            else:missing.append({'run':run['id'],'reason':'matching regular/Visible-UV sphere baseline absent'})
        elif run['cohort'].startswith('seam_') and run['cohort']!='seam_legacy':
            reference=next(r for r in runs if r['cohort']=='seam_legacy')
            comparisons.append((run['id'],reference['id']))
    totals={(r['id'],region):[0,0.,0.] for r in runs for region in ('all','background','foreground','boundary')}
    rows=[];difference_rows=[]
    for frame in frames:
        target=rgb(data_dir/'images_undistorted'/f'{frame:03d}.png',size)
        foreground=load_mask(data_dir,frame,size)
        selections=regions(foreground,args.boundary_width)
        selections['all']=np.ones_like(foreground)
        region_image=target.copy()
        for region,color in [('background',[40,120,220]),('foreground',[240,180,30]),('boundary',[210,30,160])]:
            choose=selections[region]
            region_image[choose]=(target[choose]*.5+np.array(color)*.5).astype(np.uint8)
        mask_dir=out/'regions';mask_dir.mkdir(exist_ok=True)
        Image.fromarray(region_image).save(mask_dir/f'frame_{frame:03d}.png')
        errors={}
        for run in runs:
            index=run['frames'].index(frame)
            render=rgb(Path(run['path'])/'final/complete'/f'{index:02d}_t.png',size)
            stats,mae=image_statistics(render,target,selections)
            errors[run['id']]=mae
            for region,s in stats.items():
                n=s['pixels'];a=0 if not n else s['mae']*n;b=0 if not n else s['mse']*n
                row={'sequence':label,'run':run['id'],'frame':frame,'region':region,**region_metrics(n,a,b)}
                rows.append(row)
                t=totals[run['id'],region];t[0]+=n;t[1]+=a;t[2]+=b
        for candidate,reference in comparisons:
            delta=errors[candidate]-errors[reference]
            folder=out/(candidate+'__vs__'+reference);folder.mkdir(exist_ok=True)
            color=signed_error_rgb(delta,args.display_cap)
            Image.fromarray(color).save(folder/f'frame_{frame:03d}_signed_mae.png')
            strength=np.minimum(np.abs(delta)/args.display_cap,1)[...,None]*.8
            overlay=np.rint(target*(1-strength)+color*strength).astype(np.uint8)
            Image.fromarray(overlay).save(folder/f'frame_{frame:03d}_overlay.jpg',quality=92)
            for region,mask in selections.items():
                if not mask.any():continue
                difference_rows.append({'sequence':label,'candidate':candidate,'reference':reference,
                    'frame':frame,'region':region,'pixels':int(mask.sum()),
                    'mean_mae_difference':float(delta[mask].mean()),
                    'improved_pixel_fraction':float((delta[mask]<0).mean()),
                    'worsened_pixel_fraction':float((delta[mask]>0).mean())})
        print(f'[spatial] {label} frame {frame}',flush=True)
    aggregate=[]
    for (run_id,region),(n,a,b) in totals.items():
        aggregate.append({'sequence':label,'run':run_id,'region':region,**region_metrics(n,a,b)})
    summary={'key':key,'directory':str(out),'sequence':label,'provenance':provenance,
             'scope':'smoke_subset' if len(frames)<len(runs[0]['frames']) else 'full_interval',
             'frames':frames,'statistics':aggregate,'comparisons':comparisons,'missing_baselines':missing,
             'mask_policy':'all saved foreground layers, nearest-neighbor resized to render size; excludes no semantic class',
             'region_colors':{'background':'blue','foreground':'yellow','boundary':'magenta'},
             'difference_colors':{'negative':'blue = candidate improves','zero':'white','positive':'red = candidate worsens'},
             'display_cap':args.display_cap,'normalization':'RGB mean absolute difference / 255; region statistics pixel weighted',
             'perfect_psnr_policy':'PSNR null and perfect_match true when MSE=0; null for empty regions'}
    write_csv(out/'frame_region_metrics.csv',rows)
    write_csv(out/'region_metrics.csv',aggregate)
    write_csv(out/'comparison_metrics.csv',difference_rows)
    write_json(out/'summary.json',summary);write_json(args.output/label/'spatial.json',summary)
