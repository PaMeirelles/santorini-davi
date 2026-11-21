import importlib.util
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import subprocess
import tempfile
import unittest
import numpy as np
import torch
from context import (BASE, HERE, ENGINE_SRC, RECORD, CONTEXT_RECORD, Context, PLANE_NAMES, baseline,
                     canonical_context_key, canonical_key, context_planes, context_records,
                     load_model, tensorize, transform_context, transform_position, TRANSFORMS)

import raven_pipeline as pipeline
from game.board import Board
from game.move import ApolloMove, DemeterMove, HephaestusMove, AtlasMove, PrometheusMove


def position(god=0, side=0):
    occupants = {6: 'G', 18: 'G', 8: 'B', 16: 'B'}
    result = ''.join('0' + occupants.get(i, 'N') for i in range(25)) + f'{side}{god}{god}0'
    canonical_key(result)
    return result


def rows_for(god, move, side=0, game=1):
    return np.array([(np.frombuffer(p.encode(), np.uint8), canonical_key(p).encode(),
                      phase, action, 1, game, 0)
                     for p, phase, action in pipeline.phase_samples(Board(position(god, side)), move)], dtype=RECORD)


class ContextTests(unittest.TestCase):
    def test_movement_then_build_has_actor_and_origin(self):
        rows = rows_for(0, ApolloMove(6, 7, 2))
        contexts = context_records(rows)
        self.assertEqual(tuple(contexts[0]), (-1, -1, -1, 0))
        self.assertEqual(tuple(contexts[1]), (7, 6, -1, 0))

    def test_demeter_hephaestus_atlas_remember_first_build(self):
        for god, move in [(4, DemeterMove(6, 7, 2, 3)),
                          (5, HephaestusMove(6, 7, 2, 2)), (3, AtlasMove(6, 7, 2, True))]:
            with self.subTest(god=god):
                contexts = context_records(rows_for(god, move))
                self.assertEqual(tuple(contexts[2]), (7, 6, 2, 0))

    def test_prometheus_prebuild_and_pass_no_future_worker_leakage(self):
        for prebuild in (None, 2):
            with self.subTest(prebuild=prebuild):
                contexts = context_records(rows_for(9, PrometheusMove(6, 7, 3, prebuild)))
                self.assertEqual(tuple(contexts[0]), (-1, -1, -1, 0))
                self.assertEqual(tuple(contexts[1]), (-1, -1, -1 if prebuild is None else 2, int(prebuild is not None)))
                self.assertEqual(tuple(contexts[2]), (7, 6, -1 if prebuild is None else 2, int(prebuild is not None)))

    def test_turn_and_game_boundaries_reset_all_context(self):
        rows = np.concatenate([rows_for(9, PrometheusMove(6, 7, 3, 2)),
                               rows_for(9, PrometheusMove(8, 7, 3, None), side=1),
                               rows_for(9, PrometheusMove(6, 7, 3, None), game=2)])
        contexts = context_records(rows)
        for index in (0, 3, 6):
            self.assertEqual(tuple(contexts[index]), (-1, -1, -1, 0))

    def test_truncated_or_shuffled_phase_stream_rejected(self):
        rows = rows_for(4, DemeterMove(6, 7, 2, 3))
        with self.assertRaises(ValueError): context_records(rows[1:])
        with self.assertRaises(ValueError): context_records(rows[:1])

    def test_original_planes_and_targets_are_bit_identical(self):
        rows = rows_for(9, PrometheusMove(6, 7, 3, 2))
        new, value, policy = tensorize(rows, context_records(rows))
        old, old_value, old_policy = baseline.tensorize(rows, 'cpu')
        self.assertEqual(tuple(new.shape), (3, 35, 5, 5))
        torch.testing.assert_close(new[:, :31], old, rtol=0, atol=0)
        torch.testing.assert_close(value, old_value, rtol=0, atol=0)
        torch.testing.assert_close(policy, old_policy, rtol=0, atol=0)
        self.assertEqual(new[0, 31:].sum(), 0)
        self.assertEqual(new[1, 34].sum(), 25)

    def test_all_eight_symmetries_transform_context_with_board(self):
        p = position()
        c = Context(6, 0, 4, True)
        key = canonical_context_key(p, 1, c)
        original = context_planes(np.array([c.record()], dtype=CONTEXT_RECORD)).reshape(4, 25)
        for t in range(8):
            transformed = transform_context(c, t)
            self.assertEqual(key, canonical_context_key(transform_position(p, t), 1, transformed))
            actual = context_planes(np.array([transformed.record()], dtype=CONTEXT_RECORD)).reshape(4, 25)
            np.testing.assert_array_equal(actual, original[:, TRANSFORMS[t]])
        self.assertNotEqual(key, canonical_context_key(p, 1, Context(6, 0, 4, False)))
        self.assertNotEqual(key, canonical_context_key(p, 2, c))
        self.assertNotEqual(key, canonical_context_key(p, 1, Context(18, 0, 4, True)))

    def test_native_encoder_matches_python_all_squares_and_absent(self):
        # Compile and execute the same C++ helper used by the runtime adapter.
        source = '''#include <cstdio>
#include "context_planes.h"
int main() { float out[100];
for (int s=-1;s<25;++s) for(int flag=0;flag<2;++flag) {
 encode_context_planes(out,s,s,s,flag); fwrite(out,sizeof(float),100,stdout);
} }'''
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory) / 'parity.cpp'
            exe = Path(directory) / 'parity.exe'
            src.write_text(source)
            subprocess.run(['g++', '-std=c++17', '-O2', '-static', '-I', str(ENGINE_SRC), str(src), '-o', str(exe)], check=True, capture_output=True)
            actual = np.frombuffer(subprocess.check_output([str(exe)]), dtype=np.float32).reshape(52, 4, 5, 5)
        records = np.array([(s, s, s, flag) for s in range(-1, 25) for flag in (0, 1)], dtype=CONTEXT_RECORD)
        np.testing.assert_array_equal(actual, context_planes(records))

    def test_new_network_forward_backward_and_torchscript_reload(self):
        torch.set_num_threads(2)
        model = load_model()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 4608646)
        inputs = torch.zeros(2, 35, 5, 5)
        inputs[:, 31, 1, 2] = 1
        policy, value = model(inputs)
        loss = torch.nn.functional.cross_entropy(policy, torch.tensor([1, 2])) + torch.nn.functional.binary_cross_entropy_with_logits(value, torch.ones(2, 1))
        loss.backward()
        self.assertGreater(model.shared_conv[0].weight.grad[:, 31].abs().sum().item(), 0)
        model.eval()
        with tempfile.TemporaryDirectory() as directory, torch.inference_mode():
            path = str(Path(directory) / 'test.pt')
            torch.jit.trace(model, inputs[:1]).save(path)
            loaded = torch.jit.load(path)
            for expected, actual in zip(model(inputs), loaded(inputs)):
                torch.testing.assert_close(actual, expected)

    def test_runtime_board_trackers_match_record_context(self):
        runtime = ENGINE_SRC
        source = '''#include <iostream>
#include "board.h"
int main(int argc,char** argv) { Santorini::Board b(argv[1]);
for(int i=2;i<argc;++i) {
 std::cout << b.get_active_worker() << ' ' << b.get_movement_origin() << ' '
 << b.get_previous_build() << ' ' << b.get_prometheus_prebuilt() << '\\n';
 b.make_action(std::stoi(argv[i]));
} }'''
        with tempfile.TemporaryDirectory() as directory:
            src, exe = Path(directory) / 'runtime.cpp', Path(directory) / 'runtime.exe'
            src.write_text(source)
            subprocess.run(['g++', '-std=c++17', '-O2', '-static', '-I', str(runtime),
                            str(src), str(runtime / 'board.cpp'), '-o', str(exe)], check=True, capture_output=True)
            for god, move in [(0, ApolloMove(6, 7, 2)), (4, DemeterMove(6, 7, 2, 3)),
                              (5, HephaestusMove(6, 7, 2, 2)), (3, AtlasMove(6, 7, 2, True))]:
                rows = rows_for(god, move)
                output = subprocess.check_output([str(exe), position(god), *map(str, rows['policy'])], text=True)
                actual = [tuple(map(int, line.split())) for line in output.splitlines()]
                self.assertEqual(actual, [tuple(row) for row in context_records(rows)])


if __name__ == '__main__':
    unittest.main(verbosity=2)
