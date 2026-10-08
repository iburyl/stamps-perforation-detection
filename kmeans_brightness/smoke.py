"""Single-stamp check of the four methods and all six splits."""
import pathlib
import sys
import time

import cv2

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import experiment as E  # noqa: E402

STAMP = int(sys.argv[1]) if len(sys.argv) > 1 else 1

image = cv2.imread(E.SCAN)
orientations, threshold = E.front_end(image)
print(f'{len(orientations)} stamps, threshold {threshold:.1f}')

started = time.perf_counter()
result = E.measure_stamp_methods(image, orientations[STAMP - 1], threshold)
print(f'stamp {STAMP} measured in {time.perf_counter() - started:.1f}s'
      f'  (all 7 candidates per side)')
print()

for method in E.METHODS:
    cells = []
    for side in E.SIDES:
        v = result[method][side]
        shown = '--' if v['gauge'] is None else format(v['gauge'], '.3f')
        cells.append(f"{side[0]}:{str(v['pick'] or '-'):<6}{shown:>7} "
                     f"a{v['arcs']:<2} {str(v['status'])[:6]:<6}")
    print(f'{method:<10} ' + ' | '.join(cells))

print()
for side in E.SIDES:
    print(side)
    for split in result['splits'][side]:
        shown = '--' if split['gauge'] is None else format(split['gauge'], '.3f')
        print(f"   {split['name']:<6} score {split['score']:>8.2f} "
              f"gauge {shown:>7} arcs {split['arcs']:>2} "
              f"status {str(split['status'])[:11]:<11} L={split['centers_L']}")
