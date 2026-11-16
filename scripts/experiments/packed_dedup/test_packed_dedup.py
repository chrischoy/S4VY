"""Exact contracts against the evaluated CPU reference, including packed tails."""
import unittest

import numpy as np

from packed_dedup import dedup_keep, warmup
from reference_dedup import dedup_keep as reference


class PackedDedupTests(unittest.TestCase):
    def assert_parity(self, masks, scores, threshold, weights=None, priority=None):
        expected = reference(masks, scores, threshold, weights=weights,
                             priority=priority, return_keeper=True)
        actual = dedup_keep(masks, scores, threshold, weights=weights,
                            priority=priority, return_keeper=True)
        for got, want in zip(actual, expected):
            np.testing.assert_array_equal(got, want)
        np.testing.assert_array_equal(dedup_keep(masks, scores, threshold, weights=weights,
                                               priority=priority), expected[0])

    def test_random_tubes_and_frames_exact_parity(self):
        rng = np.random.default_rng(7301)
        for shape in ((0, 3, 5), (1, 5, 13), (19, 5, 13), (11, 4, 9, 17)):
            masks = rng.random(shape) > .4
            scores = rng.choice([.2, .3, .5, .8], shape[0])
            if len(masks) > 2:
                masks[1] = masks[0]
                scores[1] = scores[0]
                masks[-1] = False
            for weights in (None, np.ones(shape[1:], np.int64),
                            rng.integers(0, 11, shape[1:])):
                for threshold in (0, .3, .5, .7, 1):
                    for priority in (None, rng.integers(0, 3, shape[0])):
                        self.assert_parity(masks, scores, threshold, weights, priority)

    def test_exact_threshold_and_greedy_chain(self):
        masks = np.array([[1,1,0], [1,1,1], [0,1,1]], bool)
        for threshold in (1/3, .5, 2/3, np.float32(2/3).item(),
                          np.float32(1/3), np.nextafter(np.float32(1/3), np.float32(1)),
                          np.nextafter(np.float32(1/3), np.float32(0))):
            self.assert_parity(masks, [.9,.8,.7], threshold, np.ones(3, np.int64))
            self.assert_parity(masks, [.9,.8,.7], threshold)
        np.testing.assert_array_equal(dedup_keep(masks, [.9,.8,.7], .5), [True,False,True])

    def test_nearest_weights_noninteger_expansion_and_broadcast_tubes(self):
        rng = np.random.default_rng(82)
        masks = rng.random((21, 5, 7, 11)) > .6
        masks[3:6] = masks[0]
        y = np.arange(29) * 7 // 29
        x = np.arange(43) * 11 // 43
        weights = np.bincount(y, minlength=7)[:,None] * np.bincount(x, minlength=11)[None,:]
        weights = np.broadcast_to(weights, masks.shape[1:])
        self.assert_parity(masks, rng.random(21), .7, weights)
        expanded = masks[:, :, y, :][:, :, :, x]
        scores = rng.random(21)
        np.testing.assert_array_equal(dedup_keep(masks, scores, .7, weights),
                                      reference(expanded, scores, .7, weights=np.ones(expanded.shape[1:])))

    def test_zero_weights_empty_masks_and_tail_bits(self):
        for pixels in (1, 15, 16, 17, 63, 64, 65, 127, 128, 129):
            masks = np.ones((3,pixels), bool)
            self.assert_parity(masks, [.8,.8,.7], .7, np.zeros(pixels, np.int64))
            self.assert_parity(masks, [.8,.8,.7], .7, np.full(pixels, 17, np.int64))
            self.assert_parity(masks, [.8,.8,.7], .7)

    def test_invalid_weights_and_threshold(self):
        masks = np.ones((2,3), bool)
        for weights in ([.5,1,2], [-1,2,3], [np.nan,1,2], [1,2], [2**52]*3):
            with self.assertRaises(ValueError):
                dedup_keep(masks, [.8,.7], .7, weights)
        for threshold in (-.1, 1.1, np.nan):
            with self.assertRaises(ValueError):
                dedup_keep(masks, [.8,.7], threshold)

    def test_noncanonical_bool_storage(self):
        masks = np.full((3, 65), 255, dtype=np.uint8).view(np.bool_)
        masks[2, :32] = False
        self.assert_parity(masks, [.9, .8, .7], .7, np.ones(65, np.int64))

    def test_cache_reused(self):
        self.assertEqual(warmup(), warmup())


if __name__ == '__main__':
    unittest.main()
