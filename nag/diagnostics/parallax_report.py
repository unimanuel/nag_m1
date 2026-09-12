"""Standalone HTML/CSV/plot report; no browser or network services required."""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from .parallax_evaluate import evaluation_key
from .parallax_io import code_signature, digest, snapshot, write_csv, write_json
from .parallax_observations import check_prepared


def table(rows,columns):
    def cell(value):
        if value is None:return '—'
        if isinstance(value,float):return f'{value:.5g}'
        return html.escape(str(value))
    head=''.join(f'<th>{html.escape(label)}</th>' for _,label in columns)
    body=''.join('<tr>'+''.join('<td>'+cell(row.get(k))+'</td>' for k,_ in columns)+'</tr>' for row in rows)
    return '<div class="scroll"><table><thead><tr>'+head+'</tr></thead><tbody>'+body+'</tbody></table></div>'


def report(args,inv):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    joined=[];pending=[];sections=[];pair_counts=[]
    for label in args.sequences:
        root=args.output/label
        if not all((root/name).exists() for name in ('prepared.json','evaluated.json','spatial.json')):
            pending.append(label);continue
        meta=check_prepared(root/'prepared.json',args.repo)
        evaluation=json.loads((root/'evaluated.json').read_text())
        spatial=json.loads((root/'spatial.json').read_text())
        runs=[r for r in inv['runs'] if r['sequence']==label]
        if evaluation['key']!=evaluation_key(runs,meta):
            raise ValueError(f'{label}: run inventory changed; rerun evaluation')
        provenance=spatial['provenance']
        if (provenance['runs']!=runs or provenance['code']!=code_signature(args.repo) or
            provenance['inputs']!=snapshot([p['path'] for p in provenance['inputs']])):
            raise ValueError(f'{label}: spatial inputs changed; rerun spatial export')
        pair_counts.append({'sequence':label,'scope':meta['scope'],**meta['counts']})
        for run in runs:
            result=next(r for r in evaluation['results'] if r['id']==run['id'])
            for visibility in ('supported','supported_plus_unknown'):
                stats=next(s for s in result['statistics'] if s['visibility']==visibility and s['group']=='all')
                row={**stats,'geometry_id':run['geometry_id'],'observation_scope':meta['scope'],
                     'spatial_scope':spatial['scope'],'training_PSNR':run['metrics']['MPSNR'],
                     'training_SSIM':run['metrics']['SSIM'],'training_LPIPS':run['metrics']['LPIPS']}
                for region in ('all','background','foreground','boundary'):
                    regional=next(s for s in spatial['statistics'] if s['run']==run['id'] and s['region']==region)
                    for k in ('mae','rmse','psnr'):row[region+'_'+k]=regional[k]
                row['max_camera_normalized_radius']=result['max_camera_normalized_radius']
                row['conservative_camera_clearance_m']=result['conservative_camera_clearance_m']
                row['ray_depth_relative_median']=result['depth_absolute_relative_error']['median']
                joined.append(row)
        local=[r for r in joined if r['sequence']==label and r['visibility']=='supported']
        sections.append(f'<h2>{html.escape(label)}</h2><p>Geometry scope: {meta["scope"]}; spatial scope: {spatial["scope"]}.</p>')
        sections.append(table(local,[('run','Run'),('pairs','Supported pairs'),('invalid_pairs','Invalid'),
             ('pixel_error_median','Pixel median'),('pixel_error_p90','Pixel P90'),
             ('camera_angle_deg_median','Camera angle °'),('surface_angle_deg_median','Surface angle °'),
             ('uv_distance_median','UV distance'),('texture_texel_distance_median','Texel distance'),
             ('seam_crossing_fraction','Seam fraction'),('tile_covered_fraction','Tile coverage')]))
        sections.append(table(local,[('run','Run'),('background_mae','Background MAE'),('foreground_mae','Foreground MAE'),
               ('boundary_mae','Boundary MAE'),('training_PSNR','Saved PSNR ↑'),('training_SSIM','Saved SSIM ↑'),('training_LPIPS','Saved LPIPS ↓')]))
        for title,path in [('Geometry parameters and layout metadata',Path(evaluation['directory'])/'summary.json'),
                            ('Observation selection and alignment metadata',Path(meta['directory'])/'summary.json'),
                            ('Detailed geometry statistics',Path(evaluation['directory'])/'metrics.csv'),
                            ('Per-frame region statistics',Path(spatial['directory'])/'frame_region_metrics.csv'),
                            ('Image comparison statistics',Path(spatial['directory'])/'comparison_metrics.csv')]:
            url=path.relative_to(args.output).as_posix()
            sections.append(f'<p><a href="{html.escape(url)}">{title}</a></p>')
        sections.append('<p>Alignment checks: ')
        for path in sorted(Path(meta['directory']).glob('alignment_frame_*.jpg')):
            url=path.relative_to(args.output).as_posix()
            sections.append(f'<a href="{url}">{path.name}</a> ')
        sections.append('</p>')
        # Show the middle frame for every signed comparison; all frame files
        # remain available alongside these linked examples.
        middle=spatial['frames'][len(spatial['frames'])//2]
        for candidate,reference in spatial['comparisons']:
            folder=Path(spatial['directory'])/(candidate+'__vs__'+reference)
            image=(folder/f'frame_{middle:03d}_overlay.jpg').relative_to(args.output).as_posix()
            sections.append(f'<details><summary>{html.escape(candidate)} vs {html.escape(reference)}</summary>'
                            f'<p>Frame {middle}; blue improves, red worsens.</p><a href="{image}"><img src="{image}" width="720"></a></details>')
    args.output.mkdir(parents=True,exist_ok=True)
    write_csv(args.output/'comparison.csv',joined)
    write_csv(args.output/'observation_counts.csv',pair_counts)
    metrics=['pixel_error_median','camera_angle_deg_median','surface_angle_deg_median','uv_distance_median','texture_texel_distance_median']
    correlations=[];deltas=[]
    for label in args.sequences:
        for mode in ('regular','visuv'):
            rows=[r for r in joined if r['sequence']==label and r['mode']==mode and r['cohort']=='benchmark' and r['visibility']=='supported']
            # No texture reruns in the geometry correlation; invalid geometry
            # is excluded from finite-only correlations, but still reported.
            for metric in metrics:
                usable=[r for r in rows if r['invalid_pairs']==0 and r[metric] is not None and r['background_mae'] is not None]
                rho=None
                if len(usable)>=3:
                    x=[r[metric] for r in usable];y=[r['background_mae'] for r in usable]
                    if np.ptp(x)>0 and np.ptp(y)>0:rho=float(spearmanr(x,y).statistic)
                correlations.append({'sequence':label,'mode':mode,'metric':metric,'geometries':len(usable),
                                     'spearman_with_background_mae':rho})
            sphere=next((r for r in rows if r['family']=='sphere'),None)
            if sphere:
                for r in rows:
                    if r['family']=='sphere':continue
                    row={'sequence':label,'mode':mode,'family':r['family'],'run':r['run'],
                         'delta_background_mae':r['background_mae']-sphere['background_mae'],
                         'delta_PSNR':r['training_PSNR']-sphere['training_PSNR'],
                         'delta_SSIM':r['training_SSIM']-sphere['training_SSIM'],
                         'delta_LPIPS':r['training_LPIPS']-sphere['training_LPIPS']}
                    for metric in metrics:
                        row['delta_'+metric]=(r[metric]-sphere[metric]) if r[metric] is not None and sphere[metric] is not None else None
                    deltas.append(row)
    write_csv(args.output/'within_sequence_correlations.csv',correlations)
    write_csv(args.output/'deltas_vs_matching_sphere.csv',deltas)
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for ax,metric,title in zip(axes,metrics[:3],['Pixel reprojection','Camera angle','Ellipsoid angle']):
        for mode,marker in [('regular','o'),('visuv','x')]:
            rows=[r for r in deltas if r['mode']==mode and r['delta_'+metric] is not None]
            ax.scatter([r['delta_'+metric] for r in rows],[r['delta_background_mae'] for r in rows],label=mode,marker=marker)
            for r in rows:ax.annotate(r['sequence'].replace('W254_','254:').replace('W191_','191:')+' '+r['family'],
                                      (r['delta_'+metric],r['delta_background_mae']),fontsize=5)
        ax.axhline(0,color='gray',lw=.6);ax.axvline(0,color='gray',lw=.6)
        ax.set_xlabel('Change vs same-mode sphere: '+title);ax.set_ylabel('Change in background MAE')
        ax.legend(fontsize=8)
    fig.suptitle('Within-sequence changes; lower-left improves both measures')
    fig.tight_layout();fig.savefig(args.output/'metric_vs_background_error.png',dpi=160);plt.close(fig)
    intro='''<h1>Existing-geometry parallax diagnostics</h1>
<p>These are diagnostics, not a new fit. Shared original camera poses are used;
learned camera corrections, learned flow and view-dependent appearance are not simulated.
Saved PSNR/SSIM/LPIPS describe the full training interval, even when a diagnostic is labelled smoke_subset.</p>
<p>Primary visibility is sparse LiDAR-supported background correspondence. Unknown visibility is included
only in the separate sensitivity rows in comparison.csv. All methods receive identical pairs.
Finite error percentiles must be read with invalid-pair counts. Proxies outside the target image remain
in the score if in front of the camera. The bounded pixel score assigns an image-diagonal penalty to invalid pairs.</p>
<p>Surface angles are measured on the normalized unit sphere, not ellipsoid geodesics. Texel distance is
a coordinate-allocation diagnostic, not measured neural flow stress. Layout coverage and seam crossings
are reported separately. Camera clearance is a conservative bound. Sphere geometry is rounded to 0.0001 m
in historical diagnostics. Region masks are the saved NAG masks, not validated semantic labels.</p>
<p>Correlations are descriptive: few geometries, shared cameras and points, and one training run per setting.
Positive within-sequence rank correlation means lower geometric error accompanies lower background MAE.
Regular and Visible-UV are separate; missing baselines are never substituted.</p>
<p><a href="comparison.csv">All metrics + visibility sensitivity</a> |
<a href="deltas_vs_matching_sphere.csv">Changes vs matching sphere</a> |
<a href="within_sequence_correlations.csv">Within-sequence correlations</a> |
<a href="observation_counts.csv">Observation selection counts</a></p>'''
    missing=table(inv['missing'],[('sequence','Sequence'),('family','Missing family'),('mode','Missing mode')])
    html_text='<!doctype html><meta charset="utf-8"><title>Parallax diagnostics</title><style>body{font:15px sans-serif;color:#171717;background:white;margin:32px;line-height:1.5}table{border-collapse:collapse;font-size:12px}th,td{padding:7px;border:1px solid #ccc;color:#111}th{background:#ddd}tr:nth-child(even){background:#f3f3f3}.scroll{overflow-x:auto}img{max-width:100%;height:auto}a{color:#164ba0}</style>'+intro
    html_text+='<p>Pending sequences: '+html.escape(', '.join(pending) or 'none')+'</p>'+missing
    html_text+='<img src="metric_vs_background_error.png">'+''.join(sections)
    (args.output/'report.html').write_text(html_text)
    write_json(args.output/'report_manifest.json',{'sequences':args.sequences,'pending':pending,'missing':inv['missing'],
               'code':code_signature(args.repo),'row_count':len(joined)})
    print(f'[report] {args.output / "report.html"}',flush=True)
