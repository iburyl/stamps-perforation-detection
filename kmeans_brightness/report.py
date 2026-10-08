"""Read results.json and compare the four methods against brightness."""
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import (AXES, GROSS, METHODS, SIDES, degradations,  # noqa: E402
                    load, published, reference_gauge, show, wrong)

RESULTS = load()
DPI = 1200.0


def gauge_of(pitch_px):
    return 20.0 * DPI / (25.4 * pitch_px) if pitch_px else None


def axis_pairs(method):
    """(stamp, axis, a, b, |difference|) for axes both of whose sides publish."""
    for result in RESULTS:
        for first, second in AXES:
            a, b = result[method][first], result[method][second]
            if published(a) and published(b):
                yield (result['stamp'], first[0] + second[0],
                       a['gauge'], b['gauge'], abs(a['gauge'] - b['gauge']))


def quantile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


print('=' * 78)
print(f'{len(RESULTS)} stamps, {len(RESULTS) * 4} sides, worst.jpg, no fallbacks')
print('=' * 78)

print('\n1  YIELD\n')
print(f'{"":<12}' + ''.join(f'{m:>12}' for m in METHODS))
rows = {
    'published': lambda s: s['reliable'],
    'status ok': lambda s: s['status'] == 'ok',
    'status review': lambda s: s['status'] == 'review',
    'unavailable': lambda s: s['status'] == 'unavailable',
    'arcs >= 9': lambda s: (s['arcs'] or 0) >= 9,
}
for label, test in rows.items():
    counts = [sum(test(r[m][s]) for r in RESULTS for s in SIDES) for m in METHODS]
    print(f'{label:<12}' + ''.join(f'{c:>12}' for c in counts))

print('\n2  OPPOSITE-SIDE AGREEMENT  (both sides published)\n')
print(f'{"":<12}{"axes":>8}{"median":>9}{"p90":>9}{"<=0.25":>9}{"gross":>8}')
for method in METHODS:
    deltas = [d for *_, d in axis_pairs(method)]
    gross = sum(d > GROSS for d in deltas)
    within = sum(d <= 0.25 for d in deltas)
    print(f'{method:<12}{len(deltas):>8}{statistics.median(deltas):>9.3f}'
          f'{quantile(deltas, 0.9):>9.3f}'
          f'{within / len(deltas):>8.0%}{gross:>8}')

print('\n3  AGREEMENT WITH BRIGHTNESS  (both published, both sane)\n')
print(f'{"":<12}{"sides":>8}{"median":>9}{"p90":>9}{"differ>1":>10}')
for method in METHODS[1:]:
    diffs, wild = [], 0
    for result in RESULTS:
        for side in SIDES:
            a, b = result['brightness'][side], result[method][side]
            if published(a) and published(b):
                delta = abs(a['gauge'] - b['gauge'])
                wild += delta > GROSS
                if delta <= GROSS:
                    diffs.append(delta)
    print(f'{method:<12}{len(diffs) + wild:>8}{statistics.median(diffs):>9.3f}'
          f'{quantile(diffs, 0.9):>9.3f}{wild:>10}')

print('\n4  CHANGES vs BRIGHTNESS, PER SIDE\n')
print(f'{"":<12}{"gained":>9}{"lost":>7}{"upgraded":>10}{"downgraded":>12}'
      f'{"gross fixed":>13}{"gross added":>13}')
hurt_by_method = {}
for method in METHODS[1:]:
    hurt, counts = degradations(RESULTS, method)
    hurt_by_method[method] = hurt
    print(f'{method:<12}{counts["gained"]:>9}{counts["lost"]:>7}'
          f'{counts["up"]:>10}{counts["down"]:>12}'
          f'{counts["fixed"]:>13}{counts["broke"]:>13}')

print('\n5  DEGRADATIONS vs BRIGHTNESS\n')
for method in METHODS[1:]:
    hurt = hurt_by_method[method]
    print(f'  {method}  ({len(hurt)} sides)')
    for (stamp, side), reasons in sorted(hurt.items()):
        print(f'    stamp {stamp:>2} {side:<7} {"; ".join(reasons)}')
    if not hurt:
        print('    none')
    print()

print('6  SELECTION: WHICH SPLIT WINS\n')
tally = {}
for result in RESULTS:
    for method in ('kmeans3', 'kmeans4', 'kmeans34'):
        for side in SIDES:
            pick = result[method][side]['pick']
            tally.setdefault(method, {}).setdefault(pick, 0)
            tally[method][pick] += 1
for method, counts in tally.items():
    ordered = sorted(counts.items(), key=lambda kv: -kv[1])
    print(f'  {method:<10} ' + '  '.join(f'{k or "none"}:{v}'
                                         for k, v in ordered))

print('\n7  SELECTION RULE vs ORACLE  (best split by arc count)\n')
print(f'{"":<12}{"rule hits":>11}{"rule arcs":>11}{"oracle arcs":>13}'
      f'{"rule >=5":>10}{"oracle >=5":>12}')


def arc_ready(split):
    """Would publish if nothing downstream removed it."""
    return (split['arcs'] or 0) >= 5 and split['geometry_source'] == 'circle_arcs'


for method in ('kmeans3', 'kmeans4', 'kmeans34'):
    hits = total = rule_arcs = oracle_arcs = rule_ok = oracle_ok = 0
    for result in RESULTS:
        for side in SIDES:
            pool = [s for s in result['splits'][side]
                    if method == 'kmeans34' or s['k'] == int(method[-1])]
            chosen_name = result[method][side]['pick']
            chosen = next((s for s in pool if s['name'] == chosen_name), None)
            if chosen is None:
                continue
            best = max(pool, key=lambda s: (s['arcs'] or 0))
            total += 1
            hits += chosen_name == best['name']
            rule_arcs += chosen['arcs'] or 0
            oracle_arcs += best['arcs'] or 0
            rule_ok += arc_ready(chosen)
            oracle_ok += arc_ready(best)
    print(f'{method:<12}{hits / total:>10.0%}{rule_arcs:>11}{oracle_arcs:>13}'
          f'{rule_ok:>10}{oracle_ok:>12}')

print('\n8  WHERE THE DEGRADATION HAPPENS  (steps 1-5 vs step 6)\n')
print('The chosen split on each degraded side: what its boundary profile said')
print('before arc fitting, and what survived the greyscale arc pass.\n')
print(f'{"st":>3} {"side":<7}{"pick":>7}{"intermediate":>14}{"spots":>7}'
      f'{"arc gauge":>11}{"arcs":>6}{"brightness":>12}{"arcs":>6}   verdict')
by_stamp = {r['stamp']: r for r in RESULTS}
blamed = {'clustering': 0, 'arc handoff': 0}
for method in METHODS[1:]:
    hurt = hurt_by_method[method]
    print(f'\n  {method}')
    for stamp, side in sorted(hurt):
        result = by_stamp[stamp]
        entry, base = result[method][side], result['brightness'][side]
        split = next((s for s in result['splits'][side]
                      if s['name'] == entry['pick']), None)
        if split is None:
            continue
        intermediate = gauge_of(split['profile_pitch'])
        reference = reference_gauge(result, side)
        if intermediate is not None and not wrong(intermediate, reference):
            verdict = 'steps 1-5 fine, lost in the arc pass'
            blamed['arc handoff'] += 1
        else:
            verdict = 'the chosen binarization was already wrong'
            blamed['clustering'] += 1
        print(f'{stamp:>3} {side:<7}{entry["pick"]:>7}'
              f'{show(intermediate):>14}{split["profile_count"]:>7}'
              f'{show(entry["gauge"]):>11}{entry["arcs"]:>6}'
              f'{show(base["gauge"]):>12}{base["arcs"]:>6}   {verdict}')
print(f'\n  totals over all three methods: '
      f'{blamed["arc handoff"]} lost in the arc pass, '
      f'{blamed["clustering"]} bad binarization')

print('\n9  PER STAMP  (gauge per side; * = not published)\n')
header = f'{"st":>3} {"side":<7}'
for method in METHODS:
    header += f'{method:>12}'
print(header + '   picks')
for result in RESULTS:
    for side in SIDES:
        line = f'{result["stamp"]:>3} {side:<7}'
        for method in METHODS:
            entry = result[method][side]
            mark = ' ' if entry['reliable'] else '*'
            line += f'{show(entry["gauge"]):>11}{mark}'
        picks = '/'.join(str(result[m][side]['pick'] or '-')
                         for m in METHODS[1:])
        print(line + f'   {picks}')
    print()
