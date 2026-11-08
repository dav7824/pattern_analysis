import unittest

import cv2
import numpy as np
from scipy import ndimage as ndi

from skeletonize_patterns import extract_skeleton, prune_spurs
from generate_examples import make_images


class SkeletonTests(unittest.TestCase):
    def test_attached_speck_does_not_create_branch(self):
        image = dict(make_images())["irregular_edges_and_specks"]
        # Match the JPEG input used by the example runner.
        success, encoded = cv2.imencode(".jpeg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        self.assertTrue(success)
        image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
        _, before, _ = extract_skeleton(image, opening_radius=0)
        mask, after, _ = extract_skeleton(image, opening_radius=2)

        def endpoints(skeleton):
            degree = ndi.convolve(skeleton.astype(int), np.ones((3, 3), int),
                                  mode="constant") - skeleton
            return np.argwhere(skeleton & (degree == 1))

        self.assertEqual(len(endpoints(before)), 7)
        ends = endpoints(after)
        self.assertEqual(len(ends), 6)
        self.assertTrue(np.all((ends[:, 1] == 0) | (ends[:, 1] == image.shape[1] - 1)))
        self.assertEqual(ndi.label(after, np.ones((3, 3)))[1], 3)
        self.assertTrue(np.all(mask[after]))

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
