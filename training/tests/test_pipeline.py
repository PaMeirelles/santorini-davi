"""Checks for observation compatibility, phase targets, ownership, and D4 identity."""
import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import raven_pipeline as run
from board_symmetry import canonical_key, transform_position
from game.board import Board
from game.move import ApolloMove, DemeterMove, PrometheusMove

spec = importlib.util.spec_from_file_location("original_translation", HERE / "board_translation.py")
translation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(translation)


def position(gray=(6, 18), blue=(8, 16), turn=0, gods="00", athena=0):
    heights = [0, 1, 0, 2, 4, 0, 1, 1, 1, 0, 2, 1, 2, 0, 1, 0, 0, 1, 0, 2, 3, 0, 1, 0, 0]
    cells = "".join(str(h) + ("G" if i in gray else "B" if i in blue else "N") for i, h in enumerate(heights))
    p = cells + str(turn) + gods + str(athena)
    canonical_key(p)
    return p


class PipelineTests(unittest.TestCase):
    def test_vectorized_inputs_match_original_for_all_gods_sides_phases(self):
        rows, expected = [], []
        for god in range(10):
            for side in (0, 1):
                for phase in (0, 1, 2):
                    p = position(turn=side, gods=f"{god}{9-god}", athena=god % 2)
                    for t in range(8):
                        rotated = transform_position(p, t)
                        key = canonical_key(rotated)
                        self.assertEqual(key, canonical_key(p))
                        rows.append((np.frombuffer(rotated.encode(), np.uint8), key.encode(), phase, 1, 1, 1, 0))
                        expected.append(translation.board_to_tensor(Board(rotated), phase))
        data = np.array(rows, dtype=run.RECORD)
        x, value, policy = run.tensorize(data, "cpu")
        np.testing.assert_array_equal(x.numpy(), np.stack(expected))
        self.assertEqual(tuple(x.shape), (480, 31, 5, 5))

    def test_blue_apollo_swap_preserves_ownership(self):
        p = position(gray=(7, 23), blue=(6, 18), turn=1)
        board = Board(p)
        move = ApolloMove(6, 7, 1)
        self.assertTrue(board.move_is_valid(move))
        samples = list(run.phase_samples(board, move))
        after = Board(samples[1][0])
        self.assertEqual(set(after.workers[:2]), {6, 23})
        self.assertEqual(set(after.workers[2:]), {7, 18})
        self.assertEqual(board.position_to_text(), p)

    def test_demeter_phase_build_sequence(self):
        p = position(gods="40")
        samples = list(run.phase_samples(Board(p), DemeterMove(6, 7, 1, 2)))
        self.assertEqual([(phase, action) for _, phase, action in samples], [(0, 157), (1, 626), (2, 652)])
        self.assertEqual(Board(samples[2][0]).blocks[1], Board(p).blocks[1] + 1)

    def test_prometheus_prebuild_sequence(self):
        p = position(gods="90")
        samples = list(run.phase_samples(Board(p), PrometheusMove(6, 7, 2, 1)))
        self.assertEqual([(phase, action) for _, phase, action in samples], [(2, 651), (0, 157), (1, 627)])
        self.assertEqual(Board(samples[1][0]).blocks[1], Board(p).blocks[1] + 1)

    def test_metadata_is_part_of_symmetry_identity(self):
        p = position()
        self.assertNotEqual(canonical_key(p), canonical_key(p[:50] + "1000"))
        self.assertNotEqual(canonical_key(p), canonical_key(p[:50] + "0010"))
        self.assertNotEqual(canonical_key(p), canonical_key(p[:53] + "1"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
