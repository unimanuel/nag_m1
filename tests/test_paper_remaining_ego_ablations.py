"""Remaining-cohort selection/provenance tests; no real fitting or training."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'reports'), str(REPO / 'nag')]
import paper_remaining_ego_ablations as extension

driver = extension.driver


def plan():
    return dict(intervals=[dict(label=label, D1_geometry='/missing/d1.json',
        D2_geometry='/missing/d2.json', D2_runs=['/main/one', '/main/two', '/main/three'],
        D1_same_as_D2=label.startswith('W191_'), estimated_training_seconds=1200, pins={})
        for label in extension.INTERVALS], diagnostic_code={}, recipes={'D0': {}})


class RemainingEgoAblationTests(unittest.TestCase):
    def test_cohort_scope_restores_original_selection_even_on_error(self):
        original = driver.ABLATION_INTERVALS
        with self.assertRaisesRegex(RuntimeError, 'test failure'):
            with extension.remaining_intervals():
                self.assertEqual(driver.ABLATION_INTERVALS, extension.INTERVALS)
                raise RuntimeError('test failure')
        self.assertIs(driver.ABLATION_INTERVALS, original)

    def test_inventory_uses_original_audits_and_adds_exact_extension_pins(self):
        original = driver.ABLATION_INTERVALS
        def inventory(catalog):
            self.assertEqual(catalog, Path('/catalog.json'))
            self.assertEqual(driver.ABLATION_INTERVALS, extension.INTERVALS)
            p = plan(); p['diagnostic_code'] = {'/original.py': 'original-pin'}
            return p
        with patch.object(driver, 'inventory', side_effect=inventory) as audit:
            result = extension.inventory(Path('/catalog.json'))
        audit.assert_called_once()
        self.assertIs(driver.ABLATION_INTERVALS, original)
        self.assertEqual(result['diagnostic_code']['/original.py'], 'original-pin')
        self.assertEqual(result['cohort'], 'remaining_ego_loss_ablations_v1')
        paths = [Path(extension.__file__).resolve()] + [REPO / name for name in extension.LAUNCHERS]
        for path in paths:
            self.assertEqual(result['diagnostic_code'][str(path)], driver.sha(path))

    def test_inventory_rejects_wrong_or_incomplete_cohort(self):
        for intervals in ([], list(reversed(plan()['intervals']))):
            p = plan(); p['intervals'] = intervals
            with patch.object(driver, 'inventory', return_value=p), self.assertRaises(ValueError):
                extension.inventory(Path('/unused'))

    def test_matrix_has_eighteen_new_and_eighteen_reused_cells(self):
        with tempfile.TemporaryDirectory(prefix='autowork_remaining_ablation_test_') as d:
            cells = driver.matrix(plan(), Path(d))
        self.assertEqual(len(cells), 36)
        new = [c for c in cells if c['reused_run'] is None]
        self.assertEqual(len(new), 18)
        self.assertEqual(sum(c['recipe'] == 'D0' for c in new), 12)
        self.assertEqual(sum(c['recipe'] == 'D1' for c in new), 6)
        for cell in cells:
            reuse = cell['recipe'] == 'D2' or (cell['recipe'] == 'D1' and cell['label'].startswith('W191_'))
            self.assertEqual(cell['reused_run'] is not None, reuse)
        self.assertEqual(sum(c['label'] in extension.BATCHES['part1of2'] for c in new), 6)
        self.assertEqual(sum(c['label'] in extension.BATCHES['part2of2'] for c in new), 12)

    def test_roots_and_selection_cannot_target_original_batch(self):
        args = extension.parse_args(['train'])
        self.assertNotEqual(args.preparation_root, driver.DEFAULT_PREP)
        self.assertNotEqual(args.training_root, driver.DEFAULT_TRAIN)
        self.assertEqual(args.max_hours, 8.)
        bad = [
            ['prepare', '--preparation-root', str(driver.DEFAULT_PREP)],
            ['train', '--training-root', str(driver.DEFAULT_TRAIN / 'autowork_child')],
            ['train', '--intervals', 'W191_0_40'],
            ['prepare', '--intervals', 'W191_40_128'],
            ['train', '--max-hours', 'nan'],
            ['train', '--max-hours', '0'],
            ['train', '--preparation-root', '/tmp/autowork_same', '--training-root', '/tmp/autowork_same'],
        ]
        for argv in bad:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                extension.parse_args(argv)

    def test_prepare_dry_run_creates_nothing_and_never_fits_or_trains(self):
        with tempfile.TemporaryDirectory(prefix='autowork_remaining_ablation_test_') as d:
            prep = Path(d) / 'autowork_prep'; train = Path(d) / 'autowork_train'
            with patch.object(extension, 'inventory', return_value=plan()), \
                    patch.object(driver, 'prepare') as prepare, \
                    patch.object(driver, 'train_or_summary') as training, \
                    contextlib.redirect_stdout(io.StringIO()):
                extension.main(['prepare', '--dry-run', '--preparation-root', str(prep), '--training-root', str(train)])
            prepare.assert_not_called(); training.assert_not_called()
            self.assertFalse(prep.exists()); self.assertFalse(train.exists())

    def test_prepare_dispatch_keeps_original_pipeline(self):
        with tempfile.TemporaryDirectory(prefix='autowork_remaining_ablation_test_') as d:
            prep = Path(d) / 'autowork_prep'; train = Path(d) / 'autowork_train'
            with patch.object(extension, 'inventory', return_value=plan()), \
                    patch.object(driver, 'prepare') as prepare, \
                    patch.object(driver, 'train_or_summary') as training, \
                    contextlib.redirect_stdout(io.StringIO()):
                extension.main(['prepare', '--preparation-root', str(prep), '--training-root', str(train)])
            prepare.assert_called_once(); training.assert_not_called()
            self.assertEqual(prepare.call_args.args[1], prep)
            self.assertFalse(prep.exists()); self.assertFalse(train.exists())

    def test_summary_alias_and_selected_training_use_scoped_original_driver(self):
        for stage in ('train', 'summary'):
            with tempfile.TemporaryDirectory(prefix='autowork_remaining_ablation_test_') as d:
                prep = Path(d) / 'autowork_prep'; train = Path(d) / 'autowork_train'
                selected = list(extension.BATCHES['part2of2'])
                def dispatch(args, p, cells):
                    self.assertEqual(driver.ABLATION_INTERVALS, extension.INTERVALS)
                    self.assertEqual(args.stage, stage)
                    self.assertEqual(args.intervals, selected)
                    self.assertEqual(len(cells), 36)
                argv = ['train', '--dry-run', '--intervals', *selected,
                        '--preparation-root', str(prep), '--training-root', str(train)]
                if stage == 'summary':
                    argv.append('--summary')
                with patch.object(extension, 'inventory', return_value=plan()), \
                        patch.object(driver, 'matrix', return_value=[{}] * 36) as matrix, \
                        patch.object(driver, 'show'), patch.object(driver, 'prepare') as prepare, \
                        patch.object(driver, 'train_or_summary', side_effect=dispatch) as training:
                    extension.main(argv)
                training.assert_called_once(); prepare.assert_not_called()
                self.assertTrue(matrix.call_args.kwargs['require_prepared'])
                self.assertFalse(prep.exists()); self.assertFalse(train.exists())

    def test_batch_scripts_match_declared_nonoverlapping_intervals(self):
        labels = []
        for batch, intervals in extension.BATCHES.items():
            text = (REPO / f'run_remaining_ego_loss_ablations_{batch}.sh').read_text()
            self.assertIn('--intervals ' + ' '.join(intervals), text)
            self.assertIn('"$@"', text)
            labels.extend(intervals)
        self.assertEqual(tuple(labels), extension.INTERVALS)
        self.assertEqual(len(set(labels)), 4)


if __name__ == '__main__':
    unittest.main()
