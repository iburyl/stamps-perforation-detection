"""Render --stamp style sheets for every side the k-means methods degrade.

One directory per degraded side, holding one panel per candidate boundary:
brightness plus all six centroid splits. Each panel carries the three stages
the selection rule has to get right, drawn on the band it was measured from:

    intermediate ridge   the boundary profile, yellow
    intermediate spots   the periodicity valleys before any arc fitting, blue
    final arc spots      the valleys the greyscale arc pass kept, magenta

A split that never reached arc fitting therefore shows blue with no magenta,
and a split whose ridge is nonsense shows a ragged yellow line and nothing
else, which is what rejecting it is supposed to look like.
"""
import copy
import pathlib
import shutil
import sys

import cv2
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import experiment as E  # noqa: E402
import ridges as R  # noqa: E402
from common import METHODS, degradations, load, show  # noqa: E402

sys.path.insert(0, E.REPO)
from segment_stamps import _display_gray  # noqa: E402

SHEETS = HERE / 'degradation_sheets'
RIDGE = (0, 220, 255)
INTERMEDIATE = (255, 170, 0)
ARC = (255, 0, 255)
PANEL_WIDTH = 1700


def banner(image, lines, color):
    scale, thickness, pad = 0.62, 2, 10
    height = int(28 * len(lines)) + 2 * pad
    bar = np.full((height, image.shape[1], 3), 245, np.uint8)
    for row, text in enumerate(lines):
        cv2.putText(bar, text, (12, pad + 21 + row * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, scale,
                    color if row == 0 else (40, 40, 40), thickness, cv2.LINE_AA)
    return np.vstack([bar, image])


def band_panel(V, background, candidate):
    """The edge band with ridge, intermediate spots and final arc spots."""
    n0, n1, start = V['n0'], V['n1'], V['start']
    image = cv2.cvtColor(_display_gray(background), cv2.COLOR_GRAY2BGR)

    depth = candidate['depth']
    for i in range(len(depth) - 1):
        if np.isfinite(depth[i]) and np.isfinite(depth[i + 1]):
            cv2.line(image, (i, int(round(depth[i] - n0))),
                     (i + 1, int(round(depth[i + 1] - n0))), RIDGE, 1,
                     cv2.LINE_AA)

    scale = PANEL_WIDTH / max(1, image.shape[1])
    shown = cv2.resize(image, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_CUBIC)

    def place(point):
        return (int(round((point[0] - start) * scale)),
                int(round((point[1] - n0) * scale)))

    profile = candidate['profile']
    for number, point in enumerate(profile.get('valleys', [])[:], 1):
        x, y = place(point)
        cv2.circle(shown, (x, y), 6, INTERMEDIATE, 2, cv2.LINE_AA)
        cv2.putText(shown, str(number), (x + 7, max(16, y - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, INTERMEDIATE, 1,
                    cv2.LINE_AA)

    edge = candidate['edge']
    points = edge.get('valleys', np.empty((0, 2)))
    fits = edge.get('refinement_fits', [None] * len(points))
    for index, point in enumerate(points):
        fit = fits[index] if index < len(fits) else None
        if fit is None or fit.get('model') != 'circle':
            continue
        x, y = place(point)
        cv2.circle(shown, (x, y), 8, ARC, -1, cv2.LINE_AA)
    return shown


def describe(candidate, side_note):
    edge, profile = candidate['edge'], candidate['profile']
    arcs = E.circular_arc_count(edge)
    centers = candidate.get('centers_L')
    return [
        f"{candidate['name']}   {side_note}",
        f"ridge -> intermediate: {profile.get('count', 0)} spots, "
        f"pitch {show(profile.get('pitch_px'), '.1f')}px, "
        f"gauge {show(E.gauge(profile.get('pitch_px')))}, "
        f"coverage {profile.get('coverage', 0.0):.2f}, "
        f"autocorr {profile.get('autocorrelation', 0.0):.2f}, "
        f"score {candidate['score']:.2f}",
        f"arc pass: {arcs} circles, gauge {show(E.gauge(edge.get('pitch_px')))}, "
        f"status {edge.get('status')}, source {edge.get('geometry_source')}"
        + (f'   centroids L={centers}' if centers else ''),
    ]


def ridge_candidate(V, short, source, variant):
    """The same binarization measured through a different ridge extractor."""
    depth = R.depth_of(source['band'], V['n0'], variant)
    edge = E.measure_profile(V['positions'], depth, short)
    edge['method'] = source['name']
    profile = copy.deepcopy(edge)
    if 'valleys' in edge:
        E.refine_edge(edge, V['gray'], V['positions'], depth)
    return {
        'name': f"{source['name']}/{variant}", 'k': source['k'],
        'centers_L': source.get('centers_L'), 'band': source['band'],
        'mask': source['mask'], 'depth': depth, 'profile': profile,
        'edge': edge,
        'score': E.profile_score(profile) if 'valleys' in profile else -1.0,
    }


def render_side(image, orientation, threshold, stamp, side, headline, methods):
    directory = SHEETS / f'stamp_{stamp:02}_{side}'
    directory.mkdir(parents=True, exist_ok=True)
    P = E.patch_context(image, orientation, threshold)
    V = E.side_view(P, side)
    brightness, splits = E.side_candidates(V, P['short'])
    for candidate in [brightness] + splits:
        candidate['band'] = (
            V['work'][V['n0']:V['n1'] + 3, V['start']:V['end']]
            if candidate['name'] == 'brightness' else candidate['mask'])

    grey_band = V['gray'][V['n0']:V['n1'] + 3, V['start']:V['end']]
    (directory / 'why.txt').write_text(
        f'stamp {stamp} {side}\n' + '\n'.join(f'  {m}: {t}'
                                              for m, t in headline.items()))
    cv2.imwrite(str(directory / '0_input_gray.png'), _display_gray(grey_band))

    def paint(candidate, note, order, prefix):
        background = (grey_band if candidate['name'].startswith('brightness')
                      else candidate['mask'].astype(float) * 255)
        chosen = note.startswith('PICKED')
        panel = banner(band_panel(V, background, candidate),
                       describe(candidate, note),
                       (0, 140, 0) if chosen else (90, 90, 90))
        cv2.imwrite(str(directory / f'{prefix}{order}_'
                        f'{candidate["name"].replace("/", "_")}.png'), panel)
        return panel

    # The seven binarizations on the ridge the pipeline uses today, which is
    # what decides the selection.
    rows = []
    for order, candidate in enumerate([brightness] + splits, 1):
        picked = [m for m in methods if methods[m] == candidate['name']]
        note = ('PICKED BY ' + ', '.join(picked)) if picked else 'not picked'
        rows.append(paint(candidate, f'{note}   [ridge raw]', order, ''))
    cv2.imwrite(str(directory / 'sheet.png'), stack_rows(rows))

    # The same side measured through every ridge extractor, for brightness and
    # for whichever splits the methods actually chose: this is where a
    # degradation is either repaired or shown to be out of the ridge's reach.
    interesting = [brightness] + [s for s in splits
                                  if s['name'] in set(methods.values())]
    ridge_rows = []
    for order, source in enumerate(interesting):
        base = source['name']
        for step, variant in enumerate(R.VARIANTS):
            candidate = (dict(source) if variant == 'raw'
                         else ridge_candidate(V, P['short'], source, variant))
            candidate['name'] = f'{base}/{variant}'
            ridge_rows.append(paint(candidate, f'ridge {variant}',
                                    f'{order}{step}', 'r'))
    cv2.imwrite(str(directory / 'sheet_ridges.png'), stack_rows(ridge_rows))
    return directory


def stack_rows(rows):
    return np.vstack([np.vstack([row, np.full((6, row.shape[1], 3), 160,
                                              np.uint8)]) for row in rows])


def main():
    results = load()
    wanted = {}
    for method in METHODS[1:]:
        hurt, _ = degradations(results, method)
        for key, reasons in hurt.items():
            wanted.setdefault(key, {})[method] = '; '.join(reasons)
    print(f'{len(wanted)} degraded sides across '
          f'{len({s for s, _ in wanted})} stamps')

    if SHEETS.exists():
        shutil.rmtree(SHEETS)
    image = cv2.imread(E.SCAN)
    orientations, threshold = E.front_end(image)
    by_stamp = {r['stamp']: r for r in results}

    for stamp, side in sorted(wanted):
        picks = {m: by_stamp[stamp][m][side]['pick'] or 'brightness'
                 for m in METHODS}
        directory = render_side(image, orientations[stamp - 1], threshold,
                                stamp, side, wanted[(stamp, side)], picks)
        print(f'  stamp {stamp:>2} {side:<7} -> {directory.name}')
    print(f'-> {SHEETS}')


if __name__ == '__main__':
    main()
