"""2-D detector regression checks: python -m unittest tests.test_detection."""
import unittest

import cv2
import numpy as np

from segment_stamps import detect_stamps_2d, estimate_orientation


class DetectionTests(unittest.TestCase):
    def test_shifted_stamps_touching_pair_and_ruler(self):
        image = np.full((1000, 1600, 3), 25, np.uint8)
        # These objects deliberately do not form clean, non-overlapping rows.
        for x0, y0, x1, y1 in (
            (100, 80, 240, 270),
            (320, 170, 460, 360),
            (600, 100, 740, 290),
            (741, 100, 881, 290),  # connected pair: must be split in 2-D
            (1050, 270, 1190, 460),
        ):
            cv2.rectangle(image, (x0, y0), (x1, y1), (210, 210, 210), -1)
        # A bright elongated object must not become a sequence of "stamps".
        cv2.rectangle(image, (120, 700), (1480, 765), (225, 225, 225), -1)

        boxes, mask = detect_stamps_2d(
            image, max_processing_width=1600, dpi=100,
        )

        self.assertEqual(len(boxes), 5)
        self.assertTrue(all(estimate_orientation(mask, box) for box in boxes))

    def test_physical_limit_replaces_relative_size_limits(self):
        image = np.full((900, 1500, 3), 25, np.uint8)
        # Both are below 7 cm at 100 dpi.  Their very different sizes must not
        # cause the smaller stamp to be rejected by a relative lower bound.
        cv2.rectangle(image, (80, 100), (119, 149), (210, 210, 210), -1)
        cv2.rectangle(image, (300, 100), (499, 349), (210, 210, 210), -1)
        # This square exceeds 7 cm on both sides and must not be a proposal.
        cv2.rectangle(image, (700, 100), (1399, 799), (210, 210, 210), -1)

        boxes, _ = detect_stamps_2d(
            image, max_processing_width=1500, dpi=100,
        )

        self.assertEqual(boxes, [(80, 100, 119, 149), (300, 100, 499, 349)])


if __name__ == '__main__':
    unittest.main()
