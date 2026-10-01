"""Subset run roots must not collide with the full upright training batch."""
from argparse import Namespace
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'nag/scripts'))
from autowork_upright_training import DEFAULT_SEQUENCES, RECIPES, default_output_root


class UprightTrainingOutputTests(unittest.TestCase):
    def test_full_and_staged_subsets_have_distinct_prefixed_roots(self):
        full = Namespace(sequences=list(DEFAULT_SEQUENCES), recipes=list(RECIPES),
                         modes=['regular', 'visuv'])
        first = Namespace(sequences=list(DEFAULT_SEQUENCES), recipes=['strong_floor'],
                          modes=['regular'])
        second = Namespace(sequences=list(DEFAULT_SEQUENCES), recipes=['compact'],
                           modes=['regular'])
        roots = [default_output_root(args) for args in (full, first, second)]
        self.assertEqual(len(set(roots)), 3)
        self.assertTrue(all(path.name.startswith('autowork_') for path in roots))
        self.assertEqual(default_output_root(first), roots[1])


if __name__ == '__main__':
    unittest.main()
