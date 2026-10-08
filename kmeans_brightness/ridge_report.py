"""Compare the six ridge extractors across the four methods.

Everything is judged against the one configuration in production today,
brightness measured from the raw first-run ridge.
"""
import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ridges as R  # noqa: E402
from common import (AXES, GROSS, METHODS, SIDES, OUT,  # noqa: E402
                    degradations, published, reference_gauge, show, wrong)

RESULTS = json.loads((OUT / 'ridge_results.json').read_text())
BASE = ('brightness', 'raw')


def flatten(variant):
    """The grid for one ridge, in the per-method shape common.py expects.

    ``base`` carries the configuration in production today, so a combination
    can be compared against it while the stamp's reference gauge is still
    drawn from the four methods of the variant under test.
    """
    return [dict({m: r[m][variant] for m in METHODS}, stamp=r['stamp'],
                 base=r['brightness']['raw'])
            for r in RESULTS]


def axis_deltas(flat, method):
    out = []
    for result in flat:
        for first, second in AXES:
            a, b = result[method][first], result[method][second]
            if published(a) and published(b):
                out.append(abs(a['gauge'] - b['gauge']))
    return out


def quantile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


def tally(flat, method):
    sides = [result[method][s] for result in flat for s in SIDES]
    deltas = axis_deltas(flat, method)
    gross_sides = 0
    for result in flat:
        for s in SIDES:
            entry = result[method][s]
            if published(entry) and wrong(entry['gauge'],
                                          reference_gauge(result, s)):
                gross_sides += 1
    return {
        'published': sum(e['reliable'] for e in sides),
        'ok': sum(e['status'] == 'ok' for e in sides),
        'arcs': sum(e['arcs'] or 0 for e in sides),
        'gross_sides': gross_sides,
        'axes': len(deltas),
        'median': statistics.median(deltas) if deltas else float('nan'),
        'p90': quantile(deltas, 0.9) if deltas else float('nan'),
        'within': (sum(d <= 0.25 for d in deltas) / len(deltas)
                   if deltas else float('nan')),
        'gross_axes': sum(d > GROSS for d in deltas),
    }


FLAT = {v: flatten(v) for v in R.VARIANTS}

print('=' * 92)
print(f'{len(RESULTS)} stamps, {len(RESULTS) * 4} sides, worst.jpg, '
      f'no fallbacks, {len(METHODS)} methods x {len(R.VARIANTS)} ridges')
print('=' * 92)

print('\n1  EVERY COMBINATION\n')
print(f'{"":<12}{"ridge":<8}{"published":>10}{"ok":>5}{"arcs":>7}'
      f'{"bad sides":>11}{"axes":>6}{"median":>8}{"p90":>8}{"<=0.25":>8}'
      f'{"gross":>7}')
best = {}
for method in METHODS:
    for variant in R.VARIANTS:
        t = tally(FLAT[variant], method)
        best[(method, variant)] = t
        mark = '  <- today' if (method, variant) == BASE else ''
        print(f'{method:<12}{variant:<8}{t["published"]:>10}{t["ok"]:>5}'
              f'{t["arcs"]:>7}{t["gross_sides"]:>11}{t["axes"]:>6}'
              f'{t["median"]:>8.3f}{t["p90"]:>8.3f}{t["within"]:>7.0%}'
              f'{t["gross_axes"]:>7}{mark}')
    print()

print('2  RIDGE EFFECT ON THE METHOD IN PRODUCTION TODAY (brightness)\n')
base = best[BASE]
print(f'{"ridge":<8}{"published":>10}{"arcs":>8}{"bad sides":>11}'
      f'{"p90 axis":>10}   change vs raw')
for variant in R.VARIANTS:
    t = best[('brightness', variant)]
    print(f'{variant:<8}{t["published"]:>10}{t["arcs"]:>8}'
          f'{t["gross_sides"]:>11}{t["p90"]:>10.3f}'
          f'   {t["published"] - base["published"]:+d} published, '
          f'{t["arcs"] - base["arcs"]:+d} arcs, '
          f'{t["gross_sides"] - base["gross_sides"]:+d} bad sides')

print('\n3  EVERY COMBINATION vs BRIGHTNESS+RAW, THE PIPELINE AS IT STANDS\n')
print(f'{"":<12}{"ridge":<8}{"degraded":>9}{"gained":>8}{"lost":>6}'
      f'{"fixed":>7}{"broke":>7}')
hurt_by = {}
for method in METHODS:
    for variant in R.VARIANTS:
        hurt, counts = degradations(FLAT[variant], method, base='base')
        hurt_by[(method, variant)] = hurt
        print(f'{method:<12}{variant:<8}{len(hurt):>9}{counts["gained"]:>8}'
              f'{counts["lost"]:>6}{counts["fixed"]:>7}{counts["broke"]:>7}')
    print()

print('3b  THE REMAINING DEGRADED SIDES OF THE LEADING COMBINATIONS\n')
for method, variant in (('brightness', 'close'), ('brightness', 'gate'),
                        ('kmeans3', 'gate'), ('kmeans34', 'gate')):
    hurt = hurt_by[(method, variant)]
    print(f'  {method} + {variant}  ({len(hurt)} sides)')
    for (stamp, side), reasons in sorted(hurt.items()):
        print(f'    stamp {stamp:>2} {side:<7} {"; ".join(reasons)}')
    if not hurt:
        print('    none')
    print()

print('4  THE SIDES BRIGHTNESS+RAW GETS GROSSLY WRONG TODAY\n')
print(f'{"st":>3} {"side":<7}' + ''.join(f'{v:>9}' for v in R.VARIANTS)
      + '   (brightness only)')
for result in RESULTS:
    for side in SIDES:
        entry = result['brightness']['raw'][side]
        flat = next(f for f in FLAT['raw'] if f['stamp'] == result['stamp'])
        if not (published(entry) and wrong(entry['gauge'],
                                           reference_gauge(flat, side))):
            continue
        row = f'{result["stamp"]:>3} {side:<7}'
        for variant in R.VARIANTS:
            value = result['brightness'][variant][side]
            mark = ' ' if value['reliable'] else '*'
            row += f'{show(value["gauge"]):>8}{mark}'
        print(row)

print('\n5  COST  (measured by gate_cost.py, median over 24 sides)\n')
print('   raw 0.8 ms/side, close 1.2, conn 3.0, count 3.3, area 6.8,')
print('   gate 6.7, stack 9.9, against 1037 ms for the arc fit on the same')
print('   side. The most expensive ridge is under 1% of one arc fit, so')
print('   every combination above costs what it costs today.')
