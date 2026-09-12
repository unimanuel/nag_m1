"""Evaluate every saved geometry on identical observation pairs."""
from __future__ import annotations

import json
import numpy as np

from .parallax_core import (angle_degrees, inside, intersect, surface_coordinates,
                            surface_uv, unit, uv_errors)
from .parallax_io import digest, write_csv, write_json
from .parallax_observations import check_prepared


def batched_project(points, ids, poses, k):
    local=np.einsum('ni,nij->nj',points-poses[ids,:3,3],poses[ids,:3,:3])
    h=local@k.T
    with np.errstate(divide='ignore',invalid='ignore'):
        return h[:,:2]/h[:,2:3],local[:,2]


def distribution(values):
    values=np.asarray(values,dtype=float)
    finite=values[np.isfinite(values)]
    return {'count':len(values),'finite_count':len(finite),'invalid_count':len(values)-len(finite),
            'mean':float(np.mean(finite)) if len(finite) else None,
            'median':float(np.median(finite)) if len(finite) else None,
            'p90':float(np.quantile(finite,.90)) if len(finite) else None,
            'p95':float(np.quantile(finite,.95)) if len(finite) else None}


def evaluate_pairs(pairs, geometry, mapping, size):
    p=pairs['points'];si=pairs['source'];ti=pairs['target'];poses=pairs['poses'];k=pairs['intrinsics']
    ci, cj=poses[si,:3,3],poses[ti,:3,3]
    qi,_,vi=intersect(ci,p-ci,geometry)
    qj,_,vj=intersect(cj,p-cj,geometry)
    p_i,_=batched_project(p,si,poses,k);p_j,_=batched_project(p,ti,poses,k)
    q_i_in_j,z_ij=batched_project(qi,ti,poses,k)
    q_j_in_i,z_ji=batched_project(qj,si,poses,k)
    valid=vi & vj & (z_ij>1e-6) & (z_ji>1e-6)
    eij=np.linalg.norm(q_i_in_j-p_j,axis=1);eji=np.linalg.norm(q_j_in_i-p_i,axis=1)
    eij[~valid]=np.nan;eji[~valid]=np.nan
    theta=(angle_degrees(qi-cj,p-cj)+angle_degrees(qj-ci,p-ci))/2
    theta[~valid]=np.nan
    ui=surface_coordinates(qi,geometry);uj=surface_coordinates(qj,geometry)
    surface_angle=angle_degrees(ui,uj)
    a,b=surface_uv(ui,mapping),surface_uv(uj,mapping)
    duv,seam,texels,covered=uv_errors(a,b,mapping)
    # Projection failure is common to all headline metrics, so comparisons
    # cannot silently score a different subset for each metric.
    for v in (surface_angle,duv,texels):v[~valid]=np.nan
    return {'pixel_error':(eij+eji)/2,'pixel_forward':eij,'pixel_reverse':eji,
            'camera_angle_deg':theta,'surface_angle_deg':surface_angle,'uv_distance':duv,
            'texture_texel_distance':texels,'seam_crossing':seam & valid,
            'valid':valid,'tile_covered':covered & valid,
            'proxy_outside_image':valid & (~inside(q_i_in_j,z_ij,size)|~inside(q_j_in_i,z_ji,size))}


def evaluation_key(inv,meta):
    return digest({'runs':inv,'observations':meta['key']})


def evaluate_sequence(args,inv,label):
    meta=check_prepared(args.output/label/'prepared.json',args.repo)
    runs=[r for r in inv['runs'] if r['sequence']==label]
    key=evaluation_key(runs,meta)
    out=args.output/label/'geometry'/key[:16]
    pointer=args.output/label/'evaluated.json'
    if (out/'summary.json').exists() and not args.force:
        write_json(pointer,json.loads((out/'summary.json').read_text()))
        print(f'[reuse] {label} geometry',flush=True);return
    out.mkdir(parents=True,exist_ok=True)
    with np.load(meta['directory']+'/pairs.npz',allow_pickle=False) as f:
        pairs=dict(f)
    rows=[];details=[]
    for run in runs:
        g=run['geometry']; m=run['mapping']
        values=evaluate_pairs(pairs,g,m,run['size'])
        source=pairs['source_points'];ids=pairs['source_ids']
        cameras=pairs['poses'][:,:3,3]
        _,depth,hit=intersect(cameras[ids],source-cameras[ids],g)
        observed=np.linalg.norm(source-cameras[ids],axis=1)
        relative=np.abs(depth/observed-1)
        log=np.abs(np.log(depth/observed))
        camera_norm=np.linalg.norm(((cameras-g['center'])@np.asarray(g['rotation']))/g['radii'],axis=1)
        record={'id':run['id'],'geometry_id':run['geometry_id'],'geometry':g,'mapping':m,
                'depth_absolute_relative_error':distribution(relative), 'depth_absolute_log_error':distribution(log),
                'depth_intersection_failures':int((~hit).sum()),
                'max_camera_normalized_radius':float(camera_norm.max()),
                'conservative_camera_clearance_m':float(np.min(g['radii'])*(1-camera_norm.max()))}
        detail=[]
        for visibility,selection in [('supported',pairs['visibility']==1),('supported_plus_unknown',pairs['visibility']<=1)]:
            groups=[('all','all',selection)]
            for lo,hi in [(0,.1),(.1,1),(1,5),(5,15),(15,np.inf)]:
                groups.append(('baseline_m',f'{lo:g}-{hi:g}',selection&(pairs['baseline']>=lo)&(pairs['baseline']<hi)))
            for lo,hi in [(0,5),(5,20),(20,45),(45,181)]:
                groups.append(('heading_deg',f'{lo}-{hi}',selection&(pairs['heading']>=lo)&(pairs['heading']<hi)))
            for i,frame in enumerate(pairs['frames']):
                groups.append(('source_frame',str(frame),selection&(pairs['source']==i)))
            for group,bin_name,choose in groups:
                n=int(choose.sum())
                if not n:continue
                valid=choose&values['valid']
                row={'sequence':label,'run':run['id'],'family':run['family'],'mode':run['mode'],
                     'cohort':run['cohort'],'visibility':visibility,'group':group,'bin':bin_name,
                     'pairs':n,'invalid_pairs':int((choose&~values['valid']).sum()),
                     'proxy_outside_image_pairs':int(values['proxy_outside_image'][choose].sum()),
                     'seam_crossing_fraction':float(values['seam_crossing'][valid].mean()) if valid.any() else None,
                     'tile_covered_fraction':float(values['tile_covered'][valid].mean()) if valid.any() else None}
                for metric in ('pixel_error','camera_angle_deg','surface_angle_deg','uv_distance','texture_texel_distance'):
                    for name,value in distribution(values[metric][choose]).items():row[metric+'_'+name]=value
                # Bounded score retains a denominator even for failed projections.
                # This is a reporting score, not the proposed fitting objective.
                cap=float(np.hypot(*run['size']))
                row['capped_pixel_error_mean_with_failures']=float(np.minimum(np.nan_to_num(values['pixel_error'][choose],nan=cap),cap).mean())
                detail.append(row);rows.append(row)
        np.savez_compressed(out/(run['id']+'.npz'),**values)
        record['statistics']=detail;details.append(record)
        print(f'[evaluate] {label} {run["id"]}',flush=True)
    summary={'key':key,'sequence':label,'scope':meta['scope'],'observation_key':meta['key'],'directory':str(out),
             'camera_policy':meta['camera_policy'],'runs':runs,'results':details,
             'missing':[r for r in inv['missing'] if r['sequence']==label],
             'metric_policy':'mean of forward and reverse pixel/angular errors; finite percentiles accompanied by invalid counts; outside-image proxies retained'}
    write_csv(out/'metrics.csv',rows)
    write_json(out/'summary.json',summary);write_json(pointer,summary)
