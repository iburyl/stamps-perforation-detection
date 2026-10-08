"""Summarize fallback_results.json without external measurements."""
import json
import math
import pathlib
import statistics

HERE = pathlib.Path(__file__).resolve().parent
METHODS = (
    'current_final', 'second_cross', 'next_kmeans', 'opposite_k34',
    'sequential_k34', 'cross_k5', 'best_k5', 'opposite_k5',
    'sequential_k5', 'sequential_k345', 'extended_cross_345',
)


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * q
    low, high = math.floor(index), math.ceil(index)
    if low == high:
        return values[low]
    return values[low] * (high - index) + values[high] * (index - low)


def target_residual(side, gauge):
    targets = (14.25, 14.50) if side in ('top', 'bottom') else (14.75, 15.00)
    return min(abs(gauge - target) for target in targets)


def enriched(row, method):
    result = row['methods'][method]
    value = dict(result)
    gauge = value.get('gauge')
    opposite = row['opposite_gauge']
    value['delta'] = (abs(gauge - opposite)
                      if gauge is not None and opposite is not None else None)
    value['theory_residual'] = (target_residual(row['side'], gauge)
                                if gauge is not None else None)
    value['safe'] = bool(
        value.get('reliable') and value['delta'] is not None
        and value['delta'] <= .25 and value['theory_residual'] <= .50
    )
    return value


def summary(rows, method):
    values = [enriched(row, method) for row in rows]
    published = [value for value in values if value.get('reliable')]
    deltas = [value['delta'] for value in published
              if value['delta'] is not None]
    fits = [value.get('arc_fits') for value in values
            if value.get('arc_fits') is not None]
    return {
        'published': len(published),
        'safe': sum(value['safe'] for value in values),
        'over_025': sum(delta > .25 for delta in deltas),
        'over_05': sum(delta > .50 for delta in deltas),
        'theory_outlier': sum(value['theory_residual'] > .50
                              for value in published),
        'median': statistics.median(deltas) if deltas else None,
        'p90': percentile(deltas, .90),
        'max': max(deltas, default=None),
        'mean_fits': statistics.fmean(fits) if fits else None,
    }


def fmt(value):
    return f'{value:.3f}' if value is not None else '-'


def main():
    rows = json.loads((HERE / 'fallback_results.json').read_text())
    print(f'{len(rows)} sides rejected or replaced after primary cross-support')
    print('No manual measurements are used; safe means opposite delta <= .25 '
          'and axis-theory residual <= .50.\n')
    print(f'{"method":<20}{"pub":>5}{"safe":>6}{">.25":>6}{">.5":>5}'
          f'{"T>.5":>6}{"median":>9}{"p90":>8}{"max":>8}{"fits":>7}')
    for method in METHODS:
        stats = summary(rows, method)
        print(f'{method:<20}{stats["published"]:>5}{stats["safe"]:>6}'
              f'{stats["over_025"]:>6}{stats["over_05"]:>5}'
              f'{stats["theory_outlier"]:>6}'
              f'{fmt(stats["median"]):>9}{fmt(stats["p90"]):>8}'
              f'{fmt(stats["max"]):>8}{fmt(stats["mean_fits"]):>7}')

    print('\ncase detail')
    for row in rows:
        current = enriched(row, 'current_final')
        print(f'\n{row["scan"]}/{row["stamp"]} {row["side"]} '
              f'opposite={row["opposite_gauge"]:.3f} '
              f'current={current.get("method")} '
              f'{fmt(current.get("gauge"))} d={fmt(current["delta"])}')
        for method in METHODS[1:]:
            value = enriched(row, method)
            marker = 'OK' if value['safe'] else '--'
            print(f'  {method:<18} {marker} g={fmt(value.get("gauge"))} '
                  f'd={fmt(value["delta"])} arcs={value.get("arcs", 0):>2} '
                  f'fits={value.get("arc_fits")} '
                  f'{value.get("pick")}+{value.get("ridge")}')

if __name__ == '__main__':
    main()
