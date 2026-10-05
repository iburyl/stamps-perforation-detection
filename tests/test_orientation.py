"""Deterministic geometry checks: python -m unittest tests.test_orientation."""
import unittest

import cv2
import numpy as np

from segment_stamps import estimate_orientation


class OrientationTests(unittest.TestCase):
    def test_rotated_perforated_paper_with_cancellation(self):
        errors = []
        for perforated in (False, True):
            for cancelled in (False, True):
                for angle in (-44, -20, -6, -2, 0, 2, 6, 20, 44):
                    with self.subTest(perforated=perforated, cancelled=cancelled, angle=angle):
                        mask = np.zeros((1000, 1000), np.uint8)
                        cv2.rectangle(mask, (300, 220), (700, 780), 255, -1)
                        if perforated:
                            for x in range(310, 700, 20):
                                for y in (220, 780):
                                    cv2.circle(mask, (x, y), 6, 0, -1)
                            for y in range(230, 780, 20):
                                for x in (300, 700):
                                    cv2.circle(mask, (x, y), 6, 0, -1)
                        if cancelled:
                            cv2.line(mask, (270, 330), (730, 600), 0, 9)
                            cv2.circle(mask, (490, 440), 160, 0, 7)
                        transform = cv2.getRotationMatrix2D((500, 500), -angle, 1)
                        mask = cv2.warpAffine(mask, transform, (1000, 1000), flags=cv2.INTER_NEAREST)
                        x, y, w, h = cv2.boundingRect(mask)
                        result = estimate_orientation(mask, (x, y, x+w-1, y+h-1))
                        self.assertIsNotNone(result)
                        error = abs((result['angle_deg'] - angle + 45) % 90 - 45)
                        errors.append(error)
                        self.assertLess(error, 0.5)
                        self.assertAlmostEqual(result['center_x'], 500, delta=3)
                        self.assertAlmostEqual(result['center_y'], 500, delta=3)
                        self.assertAlmostEqual(result['width_px'], 400, delta=12)
                        self.assertAlmostEqual(result['height_px'], 560, delta=12)
        print(f'72 synthetic cases: max angle error {max(errors):.3f} deg')

    def test_empty_mask(self):
        self.assertIsNone(estimate_orientation(np.zeros((100, 100), np.uint8), (0, 0, 99, 99)))


if __name__ == '__main__':
    unittest.main()
