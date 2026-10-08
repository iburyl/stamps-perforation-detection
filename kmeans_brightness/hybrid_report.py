"""Evaluate a conservative primary-plus-fallback production method.

The primary result is kept whenever it has the five-circle evidence required
for publication.  A k-means result is considered only when the primary result
is not publishable; this makes gains cheap and prevents a fallback from
replacing an already established side.
"""
import json
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import (AXES, METHODS, SIDES, degradations,  # noqa: E402
                    reference_gauge, wrong)


def load():
    return json.loads((HERE / 'ridge_results.json').read_text())


def entry(result, method, ridge, side):
    return result[method][ridge][side]


def choose(primary, fallbacks):
    """Keep established primary evidence; otherwise use the first rescue."""
    if primary['reliable']:
        return dict(primary, source='brightness+close')
    for name, candidate in fallbacks:
        if candidate['reliable']:
            return dict(candidate, source=name)
    pool = [('brightness+close', primary)] + fallbacks
    name, candidate = max(pool, key=lambda item: (
        item[1]['arcs'] or 0,
        item[1]['gauge'] is not None,
        item[1]['status'] == 'ok',
    ))
    return dict(candidate, source=name)


def choose_by_evidence(primary, fallbacks, strong_arcs):
    """Keep a strong primary; otherwise prefer the result with most circles."""
    if primary['reliable'] and (primary['arcs'] or 0) >= strong_arcs:
        return dict(primary, source='brightness+close')
    pool = [('brightness+close', primary)] + fallbacks
    name, candidate = max(pool, key=lambda item: (
        item[1]['reliable'],
        item[1]['arcs'] or 0,
        item[1]['status'] == 'ok',
        item[1]['gauge'] is not None,
        item[0] == 'brightness+close',
    ))
    return dict(candidate, source=name)


def choose_sequential(primary, fallbacks, strong_arcs):
    """Try fallbacks in order and stop on the first strong circle result."""
    if primary['reliable'] and (primary['arcs'] or 0) >= strong_arcs:
        return dict(primary, source='brightness+close')
    tried = [('brightness+close', primary)]
    for name, candidate in fallbacks:
        tried.append((name, candidate))
        if candidate['reliable'] and (candidate['arcs'] or 0) >= strong_arcs:
            return dict(candidate, source=name)
    name, candidate = max(tried, key=lambda item: (
        item[1]['reliable'], item[1]['arcs'] or 0,
        item[1]['status'] == 'ok', item[1]['gauge'] is not None,
        item[0] == 'brightness+close',
    ))
    return dict(candidate, source=name)


def build(results, fallbacks, strong_arcs=None):
    combined = []
    for result in results:
        item = {
            method: result[method]['raw'] for method in METHODS
        }
        item['stamp'] = result['stamp']
        item['hybrid'] = {}
        for side in SIDES:
            primary = entry(result, 'brightness', 'close', side)
            candidates = [
                (f'{method}+{ridge}', entry(result, method, ridge, side))
                for method, ridge in fallbacks
            ]
            item['hybrid'][side] = (
                choose(primary, candidates) if strong_arcs is None
                else choose_by_evidence(primary, candidates, strong_arcs)
            )
        combined.append(item)
    return combined


def quantile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    return values[round((len(values) - 1) * fraction)]


def summarize(results):
    sides = [result['hybrid'][side] for result in results for side in SIDES]
    axes = []
    for result in results:
        for first, second in AXES:
            a, b = result['hybrid'][first], result['hybrid'][second]
            if a['reliable'] and b['reliable']:
                axes.append(abs(a['gauge'] - b['gauge']))
    bad = []
    for result in results:
        for side in SIDES:
            value = result['hybrid'][side]
            reference = reference_gauge(result, side)
            if value['reliable'] and wrong(value['gauge'], reference):
                bad.append((result['stamp'], side, value['gauge'], reference,
                            value['source']))
    sources = {}
    for value in sides:
        sources[value['source']] = sources.get(value['source'], 0) + 1
    return {
        'published': sum(value['reliable'] for value in sides),
        'ok': sum(value['status'] == 'ok' for value in sides),
        'arcs': sum(value['arcs'] or 0 for value in sides),
        'axes': len(axes),
        'median': statistics.median(axes) if axes else None,
        'p90': quantile(axes, .9),
        'within': (sum(value <= .25 for value in axes) / len(axes)
                   if axes else None),
        'gross_axes': sum(value > 1 for value in axes),
        'bad': bad,
        'sources': sources,
    }


def show(value, spec='.3f'):
    return '--' if value is None else format(value, spec)


def main():
    original = load()
    plans = {
        'raw': [('kmeans34', 'raw')],
        'area': [('kmeans34', 'area')],
        'close': [('kmeans34', 'close')],
        'majority': [('kmeans34', 'majority')],
        'gate': [('kmeans34', 'gate')],
        'stack': [('kmeans34', 'stack')],
        'area->raw': [('kmeans34', 'area'), ('kmeans34', 'raw')],
        'area->stack': [('kmeans34', 'area'), ('kmeans34', 'stack')],
    }
    print('brightness+close, then kmeans34 only when primary is unreliable')
    print()
    print(f'{"fallback":<13}{"pub":>5}{"ok":>5}{"arcs":>7}{"axes":>6}'
          f'{"median":>9}{"p90":>9}{"<=.25":>8}{"gross":>7}'
          f'{"degraded":>10}{"gained":>8}{"lost":>6}{"broke":>7}')
    for name, fallbacks in plans.items():
        results = build(original, fallbacks)
        summary = summarize(results)
        _, changes = degradations(results, 'hybrid')
        print(f'{name:<13}{summary["published"]:>5}{summary["ok"]:>5}'
              f'{summary["arcs"]:>7}{summary["axes"]:>6}'
              f'{show(summary["median"]):>9}{show(summary["p90"]):>9}'
              f'{summary["within"]:>8.0%}{summary["gross_axes"]:>7}'
              f'{len(degradations(results, "hybrid")[0]):>10}'
              f'{changes["gained"]:>8}{changes["lost"]:>6}'
              f'{changes["broke"]:>7}')
        if summary['bad']:
            print(' ' * 13 + 'bad: ' + ', '.join(
                f'{stamp}/{side} {gauge:.3f} vs {reference:.3f} ({source})'
                for stamp, side, gauge, reference, source in summary['bad']
            ))
        print(' ' * 13 + 'sources: ' + ', '.join(
            f'{source}={count}' for source, count in summary['sources'].items()
        ))

    print('\nweak primary: compare circle evidence with kmeans34\n')
    print(f'{"fallback":<19}{"strong":>7}{"pub":>5}{"ok":>5}{"arcs":>7}'
          f'{"axes":>6}{"median":>9}{"p90":>9}{"<=.25":>8}'
          f'{"gross":>7}{"degraded":>10}{"broke":>7}')
    evidence_plans = {
        'close': [('kmeans34', 'close')],
        'stack': [('kmeans34', 'stack')],
        'area+stack': [('kmeans34', 'area'), ('kmeans34', 'stack')],
        'close+stack': [('kmeans34', 'close'), ('kmeans34', 'stack')],
    }
    for name, fallbacks in evidence_plans.items():
        for strong_arcs in (7, 8, 9, 10, 11):
            results = build(original, fallbacks, strong_arcs)
            summary = summarize(results)
            hurt, changes = degradations(results, 'hybrid')
            print(f'{name:<19}{strong_arcs:>7}{summary["published"]:>5}'
                  f'{summary["ok"]:>5}{summary["arcs"]:>7}'
                  f'{summary["axes"]:>6}{show(summary["median"]):>9}'
                  f'{show(summary["p90"]):>9}{summary["within"]:>8.0%}'
                  f'{summary["gross_axes"]:>7}{len(hurt):>10}'
                  f'{changes["broke"]:>7}')
            if summary['bad']:
                print(' ' * 19 + 'bad: ' + ', '.join(
                    f'{stamp}/{side} {gauge:.3f} ({source})'
                    for stamp, side, gauge, _, source in summary['bad']
                ))


if __name__ == '__main__':
    main()
