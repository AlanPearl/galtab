import unittest

import numpy as np

from . import obs


class TestFuzzyHist(unittest.TestCase):
    def test_fuzzy_histogram(self):
        x = [-np.inf, 2.9, 5, 7.5]
        weights = [1, 10, 2, 4]
        centroids = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

        result = obs.fuzzy_histogram(x, centroids, weights=weights)
        known_result = list(reversed([1, 0, 1, 9, 0, 2, 0, 2, 2, 0]))
        assert np.allclose(result, known_result), (
            f"result: {result} != known result: {known_result}")
