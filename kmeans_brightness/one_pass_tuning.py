"""Legacy weight sweep against rough ruler data; do not use for validation."""
import itertools
import json
import math
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import one_pass_report as R  # noqa: E402
from collection_report import manual_rows, summarize as manual_summary  # noqa: E402
from hybrid_report import summarize as worst_summary  # noqa: E402


def selected(candidates, close_weight, brightness_weight, ok_weight):
    pool = [candidate for candidate in candidates
            if candidate['map'] in R.BASE_MAPS and R.usable(candidate)]
    if not pool:
        return None

    def key(candidate):
        profile = candidate['profile']
        return (profile['score']
                * (close_weight if candidate['ridge'] == 'close' else 1.)
                * (brightness_weight
                   if candidate['map'] == 'brightness' else 1.)
                * (ok_weight if profile['status'] == 'ok' else 1.))

    return max(pool, key=key)


def selected_with_cross_support(candidates, bonus, pitch_tol,
                                line_fraction):
    pool = [candidate for candidate in candidates
            if candidate['map'] in R.BASE_MAPS and R.usable(candidate)]
    if not pool:
        return None

    def cross_supported(candidate):
        profile = candidate['profile']
        candidate_is_brightness = candidate['map'] == 'brightness'
        for other in pool:
            if (other['map'] == 'brightness') == candidate_is_brightness:
                continue
            op = other['profile']
            if (abs(math.log(op['pitch_px'] / profile['pitch_px']))
                    <= pitch_tol
                    and abs(op['line_center'] - profile['line_center'])
                    <= line_fraction * profile['pitch_px']):
                return True
        return False

    return max(pool, key=lambda candidate: (
        candidate['profile']['score']
        * (bonus if cross_supported(candidate) else 1.)
    ))


def selected_with_ridge_stability(candidates, bonus, pitch_tol,
                                  line_fraction):
    pool = [candidate for candidate in candidates
            if candidate['map'] in R.BASE_MAPS and R.usable(candidate)]
    if not pool:
        return None

    def stable(candidate):
        profile = candidate['profile']
        for other in pool:
            if (other['map'] != candidate['map']
                    or other['ridge'] == candidate['ridge']):
                continue
            op = other['profile']
            if (abs(math.log(op['pitch_px'] / profile['pitch_px']))
                    <= pitch_tol
                    and abs(op['line_center'] - profile['line_center'])
                    <= line_fraction * profile['pitch_px']):
                return True
        return False

    return max(pool, key=lambda candidate: (
        candidate['profile']['score']
        * (bonus if stable(candidate) else 1.)
    ))


def sides(item, weights, supported=False):
    out = {}
    for side in R.SIDES:
        candidate = (selected_with_cross_support(
            item['candidates'][side], *weights,
        ) if supported else selected(item['candidates'][side], *weights))
        out[side] = R.result(candidate) if candidate else {
            'status': 'unavailable', 'reliable': False, 'arcs': 0,
            'gauge': None, 'source': 'none',
        }
    return out


def stable_sides(item, weights):
    out = {}
    for side in R.SIDES:
        candidate = selected_with_ridge_stability(
            item['candidates'][side], *weights,
        )
        out[side] = R.result(candidate) if candidate else {
            'status': 'unavailable', 'reliable': False, 'arcs': 0,
            'gauge': None, 'source': 'none',
        }
    return out


def collection_stats(data, weights, exact=False, supported=False):
    manual = manual_rows()
    samples = []
    for item in data:
        truth = manual[item['source']][item['stamp'] - 1]
        for side, value in sides(item, weights, supported).items():
            if exact and side not in ('top', 'left'):
                continue
            samples.append({
                'target': truth['horizontal' if side in ('top', 'bottom')
                                else 'vertical'],
                'gauge': value['gauge'], 'reliable': value['reliable'],
            })
    return manual_summary(samples)


def worst_stats(data, reference, weights, supported=False):
    rows = []
    for item, base in zip(data, reference):
        row = {method: base[method]['raw'] for method in R.METHODS}
        row['stamp'] = item['stamp']
        row['hybrid'] = sides(item, weights, supported)
        rows.append(row)
    return worst_summary(rows)


def stability_stats(collection, worst, reference, weights, exact=False):
    manual = manual_rows()
    samples = []
    for item in collection:
        truth = manual[item['source']][item['stamp'] - 1]
        for side, value in stable_sides(item, weights).items():
            if exact and side not in ('top', 'left'):
                continue
            samples.append({
                'target': truth['horizontal' if side in ('top', 'bottom')
                                else 'vertical'],
                'gauge': value['gauge'], 'reliable': value['reliable'],
            })
    rows = []
    for item, base in zip(worst, reference):
        row = {method: base[method]['raw'] for method in R.METHODS}
        row['stamp'] = item['stamp']
        row['hybrid'] = stable_sides(item, weights)
        rows.append(row)
    return manual_summary(samples), worst_summary(rows)


def main():
    print('WARNING: rough ruler measurements are not ground truth or tuning data')
    collection = R.load('one_pass_collection_results.json')
    worst = R.load('one_pass_results.json')
    reference = R.load('ridge_results.json')
    rows = []
    for weights in itertools.product(
            (0.6, 0.8, 1.0, 1.2, 1.4),
            (0.6, 0.8, 1.0, 1.2, 1.4),
            (0.9, 1.0, 1.1)):
        cs = collection_stats(collection, weights)
        es = collection_stats(collection, weights, exact=True)
        ws = worst_stats(worst, reference, weights)
        rank = (cs['over_05'], ws['gross_axes'],
                -(cs['published'] + es['published'] + ws['published']),
                es['p90'], cs['p90'])
        rows.append((rank, weights, cs, es, ws))
    print('close brightness ok multipliers; top robust one-pass selectors')
    print(f'{"weights":<18}{"Wpub":>6}{"gross":>7}'
          f'{"Cpub":>9}{"Cp90":>8}{">.5":>6}'
          f'{"direct":>9}{"Dp90":>8}{"Dmax":>8}')
    for _, weights, cs, es, ws in sorted(rows)[:20]:
        label = '/'.join(f'{value:.1f}' for value in weights)
        print(f'{label:<18}{ws["published"]:>6}{ws["gross_axes"]:>7}'
              f'{cs["published"]:>4}/{cs["total"]:<4}'
              f'{cs["p90"]:>8.3f}{cs["over_05"]:>6}'
              f'{es["published"]:>4}/{es["total"]:<4}'
              f'{es["p90"]:>8.3f}{es["max"]:>8.3f}')

    rows = []
    for weights in itertools.product(
            (1.05, 1.10, 1.20, 1.30, 1.40, 1.60, 2.00),
            (.02, .04, .08),
            (.08, .15, .30)):
        cs, ws = stability_stats(collection, worst, reference, weights)
        es, _ = stability_stats(
            collection, worst, reference, weights, exact=True,
        )
        rank = (cs['over_05'], ws['gross_axes'],
                -(cs['published'] + es['published'] + ws['published']),
                es['p90'], cs['p90'])
        rows.append((rank, weights, cs, es, ws))
    print('\nsame-map stack/close stability; bonus/pitch/line')
    print(f'{"weights":<18}{"Wpub":>6}{"gross":>7}'
          f'{"Cpub":>9}{"Cp90":>8}{">.5":>6}'
          f'{"direct":>9}{"Dp90":>8}{"Dmax":>8}')
    for _, weights, cs, es, ws in sorted(rows)[:20]:
        label = '/'.join(f'{value:.2f}' for value in weights)
        print(f'{label:<18}{ws["published"]:>6}{ws["gross_axes"]:>7}'
              f'{cs["published"]:>4}/{cs["total"]:<4}'
              f'{cs["p90"]:>8.3f}{cs["over_05"]:>6}'
              f'{es["published"]:>4}/{es["total"]:<4}'
              f'{es["p90"]:>8.3f}{es["max"]:>8.3f}')

    rows = []
    for weights in itertools.product(
            (1.05, 1.10, 1.20, 1.30, 1.40),
            (.02, .04, .08),
            (.08, .15, .30)):
        cs = collection_stats(collection, weights, supported=True)
        es = collection_stats(collection, weights, exact=True, supported=True)
        ws = worst_stats(worst, reference, weights, supported=True)
        rank = (cs['over_05'], ws['gross_axes'],
                -(cs['published'] + es['published'] + ws['published']),
                es['p90'], cs['p90'])
        rows.append((rank, weights, cs, es, ws))
    print('\ncross brightness/kmeans support; bonus/pitch/line')
    print(f'{"weights":<18}{"Wpub":>6}{"gross":>7}'
          f'{"Cpub":>9}{"Cp90":>8}{">.5":>6}'
          f'{"direct":>9}{"Dp90":>8}{"Dmax":>8}')
    for _, weights, cs, es, ws in sorted(rows)[:20]:
        label = '/'.join(f'{value:.2f}' for value in weights)
        print(f'{label:<18}{ws["published"]:>6}{ws["gross_axes"]:>7}'
              f'{cs["published"]:>4}/{cs["total"]:<4}'
              f'{cs["p90"]:>8.3f}{cs["over_05"]:>6}'
              f'{es["published"]:>4}/{es["total"]:<4}'
              f'{es["p90"]:>8.3f}{es["max"]:>8.3f}')


if __name__ == '__main__':
    main()
