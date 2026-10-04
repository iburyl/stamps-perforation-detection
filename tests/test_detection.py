"""2-D detector regression checks: python build/test_detection.py."""
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

        boxes, mask = detect_stamps_2d(image, max_processing_width=1600)

        self.assertEqual(len(boxes), 5)
        self.assertTrue(all(estimate_orientation(mask, box) for box in boxes))


if __name__ == '__main__':
    unittest.main()
