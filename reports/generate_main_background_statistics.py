#!/usr/bin/env python3
"""Completed 465-run main benchmark, restored intervals and planned ablations.

Read-only reporting. The historical 405-run exporter and artifacts retain
their original cohort and preliminary labels. No training or model imports.
"""
import argparse
import csv
from datetime import date
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_preliminary_background_statistics as base

EXCLUSION = 'W324_96_161'
RESTORED = tuple(base.PENDING)
PLAN = base.REPO/'docs/paper_ablation_matrix_2026-11-01.md'
ABLATION_INTERVALS = ('W191_0_40', 'W254_37_81', 'W254_86_137', 'W203_60_107', 'W975_99_162')
ABLATION_ROWS = (
    ('D0: depth only', '0', '0.15', 'Off', 'Up to 15 new'),
    ('D1: reprojection + depth', '1', '0.15', 'Off', 'Up to 15; audit reuse'),
    ('D2: paper ellipsoid', '1', '0.15', '5 / 10 / 20', 'All 15 completed'),
)


def main_manifests(repo=base.REPO):
    return base.manifests(repo) + [('final', family, repo/'runs'/root) for family, root in (
        ('baseline', 'autowork_restored_baselines_W203'),
        ('baseline', 'autowork_restored_baselines_W975'),
        ('ellipsoid', 'autowork_restored_ellipsoids_W203'),
        ('ellipsoid', 'autowork_restored_ellipsoids_W975'))]


def safe_yaml(text):
    # Standard PyYAML scalar/sequence semantics; C acceleration affects only
    # saved configuration parsing, never fitting, training or numerical arrays.
    return yaml.load(text, Loader=getattr(yaml, 'CSafeLoader', yaml.SafeLoader))


def collect_main():
    return base.collect(manifest_rows=main_manifests(), omitted={EXCLUSION}, yaml_loader=safe_yaml)


def aggregate_main(records):
    groups = base.aggregate(records, {'ego': 7, 'final': 24})
    if len(records) != 465 or not set(RESTORED) <= set(groups['final']['intervals']):
        raise ValueError('Missing restored-interval main results')
    if any(EXCLUSION in g['intervals'] for g in groups.values()):
        raise ValueError('Resource-excluded interval entered the main cohort')
    return groups


def triple(item):
    return f'{item["mean"][0]:.3f} / {item["mean"][1]:.5f} / {item["mean"][2]:.5f}'


def comparisons(groups):
    """Paired differences/wins on interval repeat-means; not significance tests."""
    rows=[]
    for group,g in groups.items():
        for candidate in base.TREATMENTS:
            for reference in ('plane','sphere_regular','sphere_visuv'):
                if candidate==reference:continue
                diff=np.asarray([v[candidate]['mean'] for v in g['intervals'].values()])-np.asarray(
                    [v[reference]['mean'] for v in g['intervals'].values()])
                for k,metric in enumerate(('PSNR','SSIM','LPIPS')):
                    signed=diff[:,k]*(1 if k<2 else -1)
                    rows.append(dict(group=group,candidate=candidate,reference=reference,metric=metric,
                        intervals=len(diff),mean_delta=float(diff[:,k].mean()),median_delta=float(np.median(diff[:,k])),
                        wins=int((signed>0).sum()),ties=int((signed==0).sum()),losses=int((signed<0).sum())))
    return rows


def compact_rows(groups,scope='intervals'):
    return [dict(group=group,identity=label,**{t:triple(v[t]) for t in base.TREATMENTS})
            for group,g in groups.items() for label,v in g[scope].items()]


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def text_report(groups,records,provenance,report_date):
    lines=['MAIN BENCHMARK COMPLETE - ABLATIONS PLANNED', 'Report date: '+report_date,
        '465 completed matched trainings; 7 ego and 24 included final intervals; 9 parent scenes.',
        'Five treatments; three fresh seed-42 repeats per interval. Same-seed nondeterminism, not three independent seeds.',
        'All four restored-depth intervals are included: '+', '.join(RESTORED),
        'W324_96_161 remains excluded for resource/runtime reasons, not reconstruction quality.',
        'No pending-depth intervals in the selected main cohort. This is not a completed-ablation report.', '',
        'STATISTICS',
        'Official mean: final epoch-80 scalar, averaged over the three repeats.',
        'Frame SD: population SD within each run (divisor F), then averaged over its three repeats.',
        'Displayed +/- is averaged frame SD, NOT confidence interval or training-repeat uncertainty.',
        'Repeat SD: sample SD of the three official run means (divisor 2), reported separately.',
        'Primary aggregates weight intervals equally; scene-balanced aggregates weight parent scenes equally.',
        'No frame pooling, held-out evaluation or direct temporal-flicker measurement is implied.',
        'PSNR (dB) and SSIM: higher is better. LPIPS: lower is better.',
        'Inter-frame concept follows NAG supplementary C.1/Tables 3,11; our SD divisor/repeat extension is explicit.', '',
        'COMPACT INDIVIDUAL TABLES: each treatment column is PSNR / SSIM / LPIPS',
        'Group | Interval | '+' | '.join(base.NAMES)]
    for r in compact_rows(groups):lines.append(' | '.join(r.values()))
    for group,g in groups.items():
        for key,title in [('interval_macro','EQUAL INTERVALS'),('scene_macro','EQUAL PARENT SCENES')]:
            lines+=['',group.upper()+' - '+title,'Treatment | PSNR +/- frame SD | SSIM +/- frame SD | LPIPS +/- frame SD']
            for t,name in zip(base.TREATMENTS,base.NAMES):
                lines.append(name+' | '+' | '.join(base.cell(g[key][t],k,True) for k in range(3)))
        for scope in ('scenes','intervals'):
            lines+=['',group.upper()+' - '+scope.upper(),'Identity | Treatment | PSNR +/- frame SD | SSIM +/- frame SD | LPIPS +/- frame SD']
            for label,v in g[scope].items():
                for t,name in zip(base.TREATMENTS,base.NAMES):
                    lines.append(label+' | '+name+' | '+' | '.join(base.cell(v[t],k,True) for k in range(3)))
        lines+=['',group.upper()+' - ACROSS-REPEAT SAMPLE SD','Interval | Treatment | PSNR repeat SD | SSIM repeat SD | LPIPS repeat SD']
        for label,v in g['intervals'].items():
            for t,name in zip(base.TREATMENTS,base.NAMES):
                lines.append(label+' | '+name+' | '+' | '.join(f'{x:.9g}' for x in v[t]['repeat_sd']))
    lines+=['','PAIRED COMPARISONS: differences candidate minus reference, wins are descriptive counts',
            'Group | Candidate | Reference | Metric | N | Mean delta | Median delta | Wins | Ties | Losses']
    for row in comparisons(groups):lines.append(' | '.join(str(v) for v in row.values()))
    lines+=['','COMPLETE RUN APPENDIX',
            'Group | Interval | Treatment | Repeat | Frames | PSNR mean/frame SD | SSIM mean/frame SD | LPIPS mean/frame SD']
    for r in sorted(records,key=lambda x:(base.GROUPS.index(x['group']),base.label_key(x['label']),base.TREATMENTS.index(x['treatment']),x['repeat'])):
        lines.append(' | '.join(str(r[k]) for k in ('group','label','treatment','repeat','frames'))+' | '+
                     ' | '.join(f'{m:.9g} / {s:.9g}' for m,s in zip(r['mean'],r['frame_sd'])))
        lines.append('  Source: '+r['path'])
        if r['path']!=r['physical_path']:lines.append('  Archive target: '+r['physical_path'])
    lines+=['','INCOMPLETE/FAILED ATTEMPTS - NOT COUNTED']+provenance['incomplete_attempts']
    lines+=['','SOURCE MANIFEST SHA256 (full file inventory in summary.json)']
    lines += [h+'  '+p for p,h in provenance['files_sha256'].items() if p.endswith('/manifest.json')]
    lines+=['','ABLATION PLAN - NOT EXECUTED BY THIS REPORT',PLAN.read_text(),
            'No training, fitting, checkpoint loading or rendering performed by this reporting script.']
    return '\n'.join(lines)+'\n'


def pdf_report(path,groups,records,report_date):
    p=base.PDF(report_date,banner='MAIN BENCHMARK COMPLETE | 465 trainings | ablations planned',footer='MAIN BENCHMARK',
        title='Completed main background benchmark and proposed ablations',subject='31 included intervals; W324 96-161 resource exclusion')
    p.page('Plane, sphere and ellipsoid: completed main benchmark',
           '7 ego + 24 included final intervals | five treatments | three seed-42 repeats | no pending-depth intervals')
    for group in base.GROUPS:
        p.para(group.upper()+': interval-balanced mean +/- averaged inter-frame SD',11)
        v=groups[group]['interval_macro']
        p.table(['Treatment','PSNR (dB), higher','SSIM, higher','LPIPS, lower'],
            [[n]+[base.cell(v[t],k) for k in range(3)] for t,n in zip(base.TREATMENTS,base.NAMES)],
            [206,192,192,192],height=24)
    p.para('The +/- values describe within-run frame variation, not uncertainty of training-repeat means.',10)
    p.para('All restored-depth intervals are included. W324 96-161 remains excluded for resources/runtime. Ablations are not complete.',9)

    p.page('Scope, statistical definitions and treatment recipes')
    for text in [
        'Official final epoch-80 means are averaged over three fresh seed-42 repeats. Frame SD uses population divisor F separately within each run, then averages those three SDs. Across-repeat sample SD uses divisor 2 on the three official run means and is shown separately.',
        'Primary results weight intervals equally. Parent-scene rows average their intervals; the scene-balanced table gives parents equal weight. We do not pool frames, report SD across interval means as frame SD, or interpret averaged SDs as confidence intervals.',
        'The inter-frame convention follows NAG supplementary C.1, Tables 3 and 11; the population divisor and three-repeat extension are explicit reporting choices. Same-seed repeats measure nondeterminism, not independent-seed variation. Adjacent/overlapping intervals are not independent scenes.',
        'Plane: existing middle-trajectory planar background. Sphere: visible-foreground-plane-center radius policy. Ellipsoid: corrected-Y-up raw-v4 initializer, original reprojection/depth fit, and frozen legacy regular-mapping sampling-floor v2 (including its original-fit fallback). Zero runtime margin; no sphere clipping, compactness or radius-scale penalty.',
        'Visible-UV changes texture allocation, not the fitted ellipsoid geometry. All modes share the nominal 960 x 1920 texture budget, epoch count, foreground model and matched settings within each interval. Final uses the approved common sampler; ego sampler/update counts vary by interval. All selected runs use ten workers.',
        'These are training-sequence reconstruction measurements, not held-out novel-view or direct parallax/flicker evaluation. A lateral-major ellipsoid is not silently discarded or repaired after seeing its quality.',
    ]:p.para(text,10)
    p.table(['Coverage','Status'],[['Ego','7/7 intervals; 105 trainings'],['Final','24/25 intervals; 360 trainings'],
        ['Restored intervals','W203 60-107; W975 0-68, 69-99, 99-162: all included'],
        ['Only exclusion','W324 96-161: resource/runtime, not quality selection']],[180,602],height=23)

    p.page('Parent-scene-balanced robustness')
    for group in base.GROUPS:
        p.para(group.upper()+': equal weight per parent scene',11);v=groups[group]['scene_macro']
        p.table(['Treatment','PSNR mean +/- frame SD','SSIM mean +/- frame SD','LPIPS mean +/- frame SD'],
            [[n]+[base.cell(v[t],k) for k in range(3)] for t,n in zip(base.TREATMENTS,base.NAMES)],
            [206,192,192,192],height=24)
    p.para('Frame SD is still averaged within this hierarchy, not recomputed across parent-scene means.',10)

    for group in base.GROUPS:
        labels=list(groups[group]['intervals'])
        for start in range(0,len(labels),12):
            chunk=labels[start:start+12]
            p.page(f'{group.title()}: compact individual results ({start+1}-{start+len(chunk)})',
                'Each cell: PSNR / SSIM / LPIPS, averaged over all three repeats. Higher / higher / lower is better.')
            p.table(['Interval']+list(base.NAMES),[[label]+[triple(groups[group]['intervals'][label][t])
                    for t in base.TREATMENTS] for label in chunk],size=8.5,height=28)

    for k,metric in enumerate(('PSNR','SSIM','LPIPS')):
        p.page('Parent-scene comparison: '+metric,'Mean +/- averaged inter-frame SD; intervals averaged within each parent.')
        p.table(['Parent scene']+list(base.NAMES),[[g.title()+' '+label]+[base.cell(v[t],k) for t in base.TREATMENTS]
            for g in base.GROUPS for label,v in groups[g]['scenes'].items()],height=28)
    for group in base.GROUPS:
        labels=list(groups[group]['intervals']);v=groups[group]['intervals']
        for k,metric in enumerate(('PSNR','SSIM','LPIPS')):
            for start in range(0,len(labels),10):
                chunk=labels[start:start+10]
                p.page(f'{group.title()}: {metric} with variability ({start+1}-{start+len(chunk)})',
                    'TOP: mean +/- averaged inter-frame SD. BOTTOM: across-repeat SAMPLE SD, a different statistic.')
                p.table(['Interval']+list(base.NAMES),[[label]+[base.cell(v[label][t],k) for t in base.TREATMENTS]
                    for label in chunk],height=20)
                p.para('Across-repeat sample SD of the three run means (not inter-frame SD):',9)
                precision=4 if k==0 else 6
                p.table(['Interval']+list(base.NAMES),[[label]+[f'{v[label][t]["repeat_sd"][k]:.{precision}f}'
                    for t in base.TREATMENTS] for label in chunk],height=18)

    comparison=comparisons(groups)
    for group in base.GROUPS:
        p.page(group.title()+': paired interval comparisons',
            'Repeat means first; equal interval weights. Deltas are candidate minus reference. Wins are descriptive, not significance tests.')
        rows=[]
        for cand,ref in [('sphere_regular','plane'),('sphere_visuv','plane'),('ellipsoid_regular','plane'),
                         ('ellipsoid_visuv','plane'),('ellipsoid_regular','sphere_regular'),('ellipsoid_visuv','sphere_visuv')]:
            take=[next(x for x in comparison if (x['group'],x['candidate'],x['reference'],x['metric'])==
                       (group,cand,ref,m)) for m in ('PSNR','SSIM','LPIPS')]
            names=dict(zip(base.TREATMENTS,base.NAMES))
            rows.append([names[cand]+' vs '+names[ref]]+[f'{x["mean_delta"]:+.4f}' for x in take]+
                        [' / '.join(str(x['wins']) for x in take)])
        p.table(['Candidate vs reference','PSNR delta','SSIM delta','LPIPS delta','Wins: P / S / L'],rows,
                [270,110,110,110,182],height=34)
        p.para('Positive PSNR/SSIM deltas and negative LPIPS deltas favor the candidate. '+
               'Each win count has denominator '+str(len(groups[group]['intervals']))+'. Exact ties are not wins.',10)

    p.page('Restored intervals: included without geometry selection',
           'All 60 restored trainings are included. Poor-looking or lateral-major fits are not removed.')
    v=groups['final']['intervals']
    p.table(['Interval']+list(base.NAMES),[[label]+[triple(v[label][t]) for t in base.TREATMENTS]
        for label in RESTORED],height=32,size=8.5)
    p.para('All five treatments and all three repeats are included for each interval. The new rows are joined to historical scene manifests by interval, treatment and repeat; failed attempts are never counted twice.',11)
    p.para('The earlier preliminary PDF remains a separate 405-run snapshot and should not be used as the updated main table.',11)

    p.page('Proposed ablations: one controlled loss ladder', 'PLAN ONLY: no new fitting or training was run for this report.')
    p.table(['Treatment','Reprojection weight','Depth weight','Floor continuation','New jobs before reuse audit'],
            [list(x) for x in ABLATION_ROWS],[190,135,95,140,222],height=34)
    p.para('D0 is NOT the old thin raw-v4 mesh: refit with the SAME optimizer, observations, initializer, orientation, bounds and safety as D1, changing only reprojection weight to zero. D1 is the original reprojection/depth recipe with floor disabled. D2 is the completed paper recipe. Use regular mapping only for this loss ladder.',11)
    p.para('Freeze all other settings: legacy poles, zero runtime margin, no clipping, no compactness/radius-scale penalty; same seeds, workers, texture budget and per-interval training configuration. Keep the common camera/surface safety settings. Evaluate all candidates, including failed floor improvements, without post-hoc best-geometry selection.',11)
    p.para('Five predeclared development intervals: '+', '.join(ABLATION_INTERVALS)+'. Three fresh seed-42 repeats per cell. These intervals have informed development; do not call this an unseen generalization test.',11)
    p.para('45 ellipsoid cells total: 15 D2 cells already complete; at most 30 new trainings before audited D1/identical-geometry reuse. Plane and regular-sphere controls already exist. Audit reuse first; do not start duplicate jobs.',11)

    p.page('Ablation scope, reuse and stopping rule', 'Core plan: at most 30 new trainings; optional experiments require a separate decision.')
    p.table(['Interval','Reason for selecting it'],[
        ['W191 0-40','Historically lateral-major ego failure case'],['W254 37-81','Turning trajectory; depth-proxy strong case'],
        ['W254 86-137','Well-studied successful reprojection/control case'],['W203 60-107','Nearly stationary camera; weak parallax evidence'],
        ['W975 99-162','Restored final interval; lateral-major / floor-fallback case']],[190,592],height=28)
    for text in [
        'Already covered: plane/sphere/ellipsoid comparison and regular versus Visible-UV on all 31 included intervals; three repeats each. Eligible pole-orientation experiments may be supplemental after provenance/config checks. Do not automatically rerun upright fitting, initialization variants or every historical geometry.',
        'Reuse an existing D1 training only after checking the exact objective/regularizer, effective geometry, interval, calibration, full training settings and repeat identity. Historical raw-v4 and mixed-setting exploratory runs are not automatically controlled ablations. No best-repeat or quality-based geometry selection.',
        'Report PSNR/SSIM/LPIPS, repeat variability and paired per-interval differences. Export fit loss, radii, camera clearance and achieved sampling-floor violations. Use matched-support pixel/flow diagnostics on selected checkpoints, not as new training treatments or proof of flow causality.',
        'After the core matrix and final validity checks, freeze the training study. Threshold sweeps or independent-seed repetitions are optional, claim-driven additions, not automatic requirements. Parallax claims need appropriate evaluation beyond reconstruction metrics.',
    ]:p.para(text,10)

    ordered=sorted(records,key=lambda r:(base.GROUPS.index(r['group']),base.label_key(r['label']),base.TREATMENTS.index(r['treatment']),r['repeat']))
    for group in base.GROUPS:
        subset=[r for r in ordered if r['group']==group]
        for first in range(0,len(subset),20):
            p.page(f'Run appendix: {group} ({first+1}-{min(first+20,len(subset))})',
                'Every completed run; metric cells show its mean +/- its own inter-frame SD. No best-repeat selection.')
            p.table(['Interval','Treatment','Repeat','PSNR','SSIM','LPIPS'],[
                [r['label'],base.NAMES[base.TREATMENTS.index(r['treatment'])],str(r['repeat'])]+
                [base.cell(r,k) for k in range(3)] for r in subset[first:first+20]],
                [124,112,44,166,168,168],height=21,size=8.5)
    p.page('Provenance and limitations')
    for text in [
        'The selected cohort has 465 unique completed jobs and 1,395 final per-frame vectors. The exporter checks exit status, exact final-frame coverage, epoch-80 finite scalars/vectors, source YAML/geometry hashes and matched effective configurations. Retained failed attempts are listed but excluded.',
        'summary.json contains logical and resolved archive paths, SHA256 source evidence, every run and both aggregation schemes. CSVs provide compact interval/parent tables, tidy mean/frame-SD/repeat-SD rows and paired comparisons. The plain-text report and PDF include complete run appendices.',
        'Logger vectors were rounded; their means must agree with the official scalar within tolerance. Official scalars remain authoritative. Native training validators separately checked all 60 restored runs in the preceding audit, including calibration and the geometry actually used at runtime.',
        'The finite benchmark contains two ego parent scenes and seven final parent scenes. Intervals overlap and are not independent experimental units. No confidence interval or significance claim is inferred from averaged frame SD or exact win counts.',
        'No numerical fitting, training, checkpoint loading or rendering was done to generate this report. The proposed ablation matrix has not been executed. Existing exploratory runs are not silently relabelled as matched ablation results.',
    ]:p.para(text,11)
    p.save(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,default=Path('/mnt/d/CGProjectData/waymo/diagnostics/autowork_main_background_statistics_2026-11-01'))
    parser.add_argument('--report-date',default='2026-11-01')
    args=parser.parse_args();date.fromisoformat(args.report_date)
    root=args.output_root
    if root.exists() or not root.name.startswith('autowork_'):parser.error('Choose a new autowork_ output directory')
    records,provenance=collect_main();groups=aggregate_main(records)
    provenance['report_sources_sha256']={str(p):base.sha(p) for p in (Path(__file__),Path(base.__file__),PLAN)}
    content=text_report(groups,records,provenance,args.report_date)
    root.mkdir(parents=True,exist_ok=False)
    (root/'main_background_results.txt').write_text(content,encoding='utf-8')
    summary=dict(status='MAIN_BENCHMARK_COMPLETE_ABLATIONS_PLANNED',date=args.report_date,groups=groups,runs=records,
        pending=[],restored_included=list(RESTORED),resource_exclusion=EXCLUSION,provenance=provenance,
        frame_sd_ddof=0,repeat_sd_ddof=1,comparisons=comparisons(groups))
    (root/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    write_csv(root/'interval_means_wide.csv',compact_rows(groups))
    write_csv(root/'parent_scene_means_wide.csv',compact_rows(groups,'scenes'))
    write_csv(root/'paired_comparisons.csv',summary['comparisons'])
    write_csv(root/'interval_statistics.csv',[dict(group=g,interval=l,treatment=t,metric=m,
        mean=v[t]['mean'][k],frame_sd=v[t]['frame_sd'][k],repeat_sd=v[t]['repeat_sd'][k])
        for g,x in groups.items() for l,v in x['intervals'].items() for t in base.TREATMENTS for k,m in enumerate(('PSNR','SSIM','LPIPS'))])
    write_csv(root/'runs.csv',[dict(group=r['group'],interval=r['label'],segment=r['segment'],treatment=r['treatment'],repeat=r['repeat'],
        frames=r['frames'],path=r['path'],physical_path=r['physical_path'],
        **{m+'_'+field:r[field][k] for k,m in enumerate(('PSNR','SSIM','LPIPS')) for field in ('mean','frame_sd')}) for r in records])
    markdown=['# Completed main benchmark: per-interval results','',
        'Each cell: PSNR / SSIM / LPIPS. Three-repeat means; no pending-depth intervals. W324 96-161 excluded for resources.','',
        '| Group | Interval | '+' | '.join(base.NAMES)+' |','| --- | --- | '+' | '.join(['---']*5)+' |']
    markdown+=['| '+' | '.join(r.values())+' |' for r in compact_rows(groups)]
    (root/'main_results_tables.md').write_text('\n'.join(markdown)+'\n')
    shutil.copyfile(PLAN,root/'ablation_plan.md')
    pdf_report(root/'main_background_results.pdf',groups,records,args.report_date)
    print('REPORTS:',root,flush=True)
    print('AGGREGATES:',json.dumps({g:v['interval_macro'] for g,v in groups.items()}),flush=True)


if __name__=='__main__':main()
