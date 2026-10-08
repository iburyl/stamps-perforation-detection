"""What the ridge variants cost, against one arc fit on the same side."""
import pathlib, sys, time
import cv2, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import experiment as E
import ridges as R

image = cv2.imread(E.SCAN)
orientations, threshold = E.front_end(image)
times = {v: [] for v in R.VARIANTS}
times['arc fit'] = []
for orientation in orientations[:6]:
    P = E.patch_context(image, orientation, threshold)
    for side in E.SIDES:
        V = E.side_view(P, side)
        band = V['work'][V['n0']:V['n1'] + 3, V['start']:V['end']]
        for variant in R.VARIANTS:
            t = time.perf_counter()
            depth = R.depth_of(band, V['n0'], variant)
            times[variant].append(time.perf_counter() - t)
        edge = E.measure_profile(V['positions'], depth, P['short'])
        if 'valleys' in edge:
            t = time.perf_counter()
            E.refine_edge(edge, V['gray'], V['positions'], depth)
            times['arc fit'].append(time.perf_counter() - t)

print(f'{"":<10}{"ms/side":>9}{"vs raw":>9}')
base = np.median(times['raw'])
for name, values in times.items():
    median = np.median(values)
    print(f'{name:<10}{median * 1000:>9.1f}{median / base:>8.1f}x')
