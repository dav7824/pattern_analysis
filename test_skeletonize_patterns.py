import unittest

import numpy as np
from scipy import ndimage as ndi

from skeletonize_patterns import extract_skeleton, prune_spurs


class SkeletonTests(unittest.TestCase):
    def test_reflection_keeps_clipped_stripe_straight(self):
        image = np.zeros((100, 120), dtype=np.uint8)
        image[25:75, :] = 255
        mask, skeleton, distances = extract_skeleton(image)
        self.assertTrue(skeleton[:, 0].any())
        self.assertTrue(skeleton[:, -1].any())
        self.assertLessEqual(np.ptp(np.nonzero(skeleton)[0]), 1)
        self.assertGreater(distances[50, 0], 20)
        self.assertTrue(np.all(mask[skeleton]))

    def test_short_spur_removed_without_splitting_main_path(self):
        skeleton = np.zeros((60, 60), dtype=bool)
        skeleton[30, 5:55] = True
        skeleton[26:30, 30] = True
        result = prune_spurs(skeleton, np.full(skeleton.shape, 10.0))
        self.assertFalse(result[26, 30])
        self.assertTrue(result[30, 5])
        self.assertTrue(result[30, 54])
        self.assertEqual(ndi.label(result, np.ones((3, 3)))[1], 1)

    def test_long_branch_and_border_endpoint_preserved(self):
        skeleton = np.zeros((60, 60), dtype=bool)
        skeleton[30, 0:55] = True
        skeleton[5:30, 30] = True
        result = prune_spurs(skeleton, np.full(skeleton.shape, 10.0))
        np.testing.assert_array_equal(result, skeleton)

    def test_empty_image(self):
        for method in ("thinning", "medial"):
            _, skeleton, _ = extract_skeleton(np.zeros((20, 20), np.uint8), method=method)
            self.assertFalse(skeleton.any())


if __name__ == "__main__":
    unittest.main()
