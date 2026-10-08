"""Compare pre-arc selectors; collection ruler deltas are legacy diagnostics."""
import json
import math
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from collection_report import manual_rows, summarize as manual_summary  # noqa: E402
from common import METHODS, SIDES, degradations  # noqa: E402
from hybrid_report import summarize as worst_summary  # noqa: E402

BASE_MAPS = {'brightness', 'k30', 'k31', 'k400', 'k401', 'k410', 'k411'}


def load(name):
    return json.loads((HERE / name).read_text())


def usable(candidate):
    return candidate['profile']['score'] > 0


def result(candidate):
    value = dict(candidate['result'])
    value['reliable'] = bool(
        value['geometry_source'] == 'circle_arcs'
        and (value['arcs'] or 0) >= 5
    )
    value['source'] = f"{candidate['map']}+{candidate['ridge']}"
    return value


def fixed(candidates, map_name, ridge):
    return next(candidate for candidate in candidates
                if candidate['map'] == map_name
                and candidate['ridge'] == ridge)


def max_score(candidates, ridges=('stack',), base_only=True,
              include_brightness=True):
    pool = [candidate for candidate in candidates
            if candidate['ridge'] in ridges and usable(candidate)
            and (not base_only or candidate['map'] in BASE_MAPS)]
    if not include_brightness:
        pool = [candidate for candidate in pool
                if candidate['map'] != 'brightness']
    return max(pool, key=lambda candidate: (
        candidate['profile']['score'],
        candidate['profile']['status'] == 'ok',
        candidate['ridge'] == 'stack',
    )) if pool else None


def consensus(candidates, ridges=('stack',), pitch_tol=.04,
              line_fraction=.15):
    pool = [candidate for candidate in candidates
            if candidate['map'] in BASE_MAPS
            and candidate['ridge'] in ridges and usable(candidate)]
    if not pool:
        return None

    def key(candidate):
        profile = candidate['profile']
        pitch = profile['pitch_px']
        line = profile['line_center']
        supporting_maps = set()
        for other in pool:
            op = other['profile']['pitch_px']
            ol = other['profile']['line_center']
            if (pitch is not None and op is not None and line is not None
                    and ol is not None
                    and abs(math.log(op / pitch)) <= pitch_tol
                    and abs(ol - line) <= line_fraction * pitch):
                supporting_maps.add(other['map'])
        return (
            len(supporting_maps),
            profile['status'] == 'ok',
            profile['score'],
            candidate['ridge'] == 'stack',
        )

    return max(pool, key=key)


def choose(candidates, selector):
    kind = selector[0]
    if kind == 'fixed':
        return fixed(candidates, selector[1], selector[2])
    if kind == 'score':
        return max_score(candidates, selector[1])
    if kind == 'score_kmeans':
        return max_score(candidates, selector[1], include_brightness=False)
    if kind == 'consensus':
        return consensus(candidates, *selector[1:])
    raise ValueError(selector)


def selected_sides(item, selector):
    sides = {}
    for side in SIDES:
        candidate = choose(item['candidates'][side], selector)
        sides[side] = (result(candidate) if candidate else {
            'status': 'unavailable', 'reliable': False, 'arcs': 0,
            'gauge': None, 'source': 'none',
        })
    return sides


def worst_rows(data, reference, selector):
    rows = []
    for item, base in zip(data, reference):
        row = {method: base[method]['raw'] for method in METHODS}
        row['stamp'] = item['stamp']
        row['hybrid'] = selected_sides(item, selector)
        rows.append(row)
    return rows


def collection_stats(data, selector, exact=False):
    manual = manual_rows()
    samples = []
    for item in data:
        truth = manual[item['source']][item['stamp'] - 1]
        for side, value in selected_sides(item, selector).items():
            if exact and side not in ('top', 'left'):
                continue
            samples.append({
                'target': truth['horizontal' if side in ('top', 'bottom')
                                else 'vertical'],
                'gauge': value['gauge'], 'reliable': value['reliable'],
            })
    return manual_summary(samples)


def selectors():
    out = {
        'brightness stack': ('fixed', 'brightness', 'stack'),
        'brightness close': ('fixed', 'brightness', 'close'),
        'score stack': ('score', ('stack',)),
        'score close': ('score', ('close',)),
        'score both': ('score', ('stack', 'close')),
        'score both kmeans': ('score_kmeans', ('stack', 'close')),
        'median7 stack': ('fixed', 'median7', 'stack'),
        'median7 close': ('fixed', 'median7', 'close'),
        'vote4/7 stack': ('fixed', 'vote4of7', 'stack'),
        'vote4/7 close': ('fixed', 'vote4of7', 'close'),
    }
    for ridges, suffix in ((('stack',), 'S'),
                            (('stack', 'close'), 'SC')):
        for pitch in (.02, .04, .08):
            for line in (.08, .15, .30):
                out[f'cons {suffix} p{pitch:.2f} l{line:.2f}'] = (
                    'consensus', ridges, pitch, line,
                )
    return out


def main():
    worst = load('one_pass_results.json')
    reference = load('ridge_results.json')
    collection_path = HERE / 'one_pass_collection_results.json'
    collection = (json.loads(collection_path.read_text())
                  if collection_path.exists() else None)
    print('WARNING: collection ruler measurements are not ground truth or tuning data')
    print('exactly one arc pass; all selection uses profile features only')
    print(f'{"selector":<24}{"Wpub":>6}{"Wp90":>8}{"gross":>7}'
          f'{"degr":>6}{"broke":>7}'
          f'{"Cpub":>9}{"Cp90":>8}{">.5":>6}'
          f'{"direct":>9}{"Dp90":>8}{"Dmax":>8}')
    ranked = []
    for label, selector in selectors().items():
        rows = worst_rows(worst, reference, selector)
        ws = worst_summary(rows)
        hurt, changes = degradations(rows, 'hybrid')
        if collection:
            cs = collection_stats(collection, selector)
            es = collection_stats(collection, selector, exact=True)
            rank = (cs['over_05'], -es['published'], es['p90'],
                    -ws['published'], ws['gross_axes'])
        else:
            cs = es = None
            rank = (ws['gross_axes'], -ws['published'], ws['p90'])
        ranked.append((rank, label, selector, ws, len(hurt), changes, cs, es))

    for _, label, _, ws, hurt, changes, cs, es in sorted(ranked):
        print(f'{label:<24}{ws["published"]:>6}{ws["p90"]:>8.3f}'
              f'{ws["gross_axes"]:>7}{hurt:>6}{changes["broke"]:>7}'
              + (f'{cs["published"]:>4}/{cs["total"]:<4}'
                 f'{cs["p90"]:>8.3f}{cs["over_05"]:>6}'
                 f'{es["published"]:>4}/{es["total"]:<4}'
                 f'{es["p90"]:>8.3f}{es["max"]:>8.3f}'
                 if collection else ''))


if __name__ == '__main__':
    main()
