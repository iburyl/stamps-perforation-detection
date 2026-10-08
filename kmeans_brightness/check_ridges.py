"""Confirm the variants are drop-in substitutes before measuring anything."""
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import experiment as E  # noqa: E402
import ridges as R  # noqa: E402

image = cv2.imread(E.SCAN)
orientations, threshold = E.front_end(image)
P = E.patch_context(image, orientations[0], threshold)

for side in E.SIDES:
    V = E.side_view(P, side)
    band = V['work'][V['n0']:V['n1'] + 3, V['start']:V['end']]
    reference = E.boundary_profile(V['work'], V['n0'], V['n1'], V['start'],
                                   V['end'])
    mine = R.depth_of(band, V['n0'], 'raw')
    same = np.array_equal(np.nan_to_num(reference, nan=-1),
                          np.nan_to_num(mine, nan=-1))
    print(f'{side:<7} raw matches boundary_profile: {same}')
    for variant in R.VARIANTS:
        depth = R.depth_of(band, V['n0'], variant)
        finite = np.isfinite(depth)
        shift = (np.nanmedian(depth[finite & np.isfinite(reference)]
                              - reference[finite & np.isfinite(reference)])
                 if finite.any() else float('nan'))
        print(f'   {variant:<6} coverage {finite.mean():>5.0%}  '
              f'median shift vs raw {shift:>+6.1f}px  '
              f'range {np.nanmin(depth):.0f}..{np.nanmax(depth):.0f}')
