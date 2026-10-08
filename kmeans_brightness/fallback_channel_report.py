"""Report fallback_channel_results.json without manual measurements."""
import json
import math
import pathlib
import statistics

import fallback_channel_experiment as X

HERE = pathlib.Path(__file__).resolve().parent


def target_residual(side, gauge):
    targets = (14.25, 14.50) if side in ('top', 'bottom') else (14.75, 15.00)
    return min(abs(gauge - target) for target in targets)


def enriched(row, method):
    value = dict(row['methods'][method])
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


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * q
    low, high = math.floor(index), math.ceil(index)
    if low == high:
        return values[low]
    return values[low] * (high-index) + values[high] * (index-low)


def summary(rows, method):
    values = [enriched(row, method) for row in rows]
    published = [value for value in values if value.get('reliable')]
    deltas = [value['delta'] for value in published
              if value['delta'] is not None]
    fits = [value.get('arc_fits', 0) for value in values]
    return {
        'published': len(published),
        'safe': sum(value['safe'] for value in values),
        'bad': sum(delta > .25 for delta in deltas),
        'median': statistics.median(deltas) if deltas else None,
        'p90': percentile(deltas, .9),
        'max': max(deltas, default=None),
        'fits': sum(fits),
        'mean_fits': statistics.fmean(fits) if fits else 0,
    }


def cascade(rows, methods):
    remaining = set(range(len(rows)))
    selected = {}
    fits = 0
    for method in methods:
        for index in list(remaining):
            value = enriched(rows[index], method)
            fits += value.get('arc_fits', 0)
            if value['safe']:
                selected[index] = method
                remaining.remove(index)
    return len(selected), fits, selected


def fmt(value):
    return '-' if value is None else f'{value:.3f}'


def main():
    rows = json.loads((HERE / 'fallback_channel_results.json').read_text())
    print(f'{len(rows)} sides rejected or replaced by primary cross-support')
    print('Reliable requires >=5 post-corner circle arcs. Safe additionally '
          'requires opposite delta <= .25 and theory residual <= .50.\n')
    print(f'{"method":<24}{"pub":>5}{"safe":>6}{">.25":>6}'
          f'{"median":>9}{"p90":>8}{"max":>8}{"fits":>7}{"mean":>8}')
    for method in X.METHODS:
        stats = summary(rows, method)
        print(f'{method:<24}{stats["published"]:>5}{stats["safe"]:>6}'
              f'{stats["bad"]:>6}{fmt(stats["median"]):>9}'
              f'{fmt(stats["p90"]):>8}{fmt(stats["max"]):>8}'
              f'{stats["fits"]:>7}{stats["mean_fits"]:>8.2f}')

    print('\ncase detail')
    for row in rows:
        print(f'\n{row["scan"]}/{row["stamp"]} {row["side"]} '
              f'current={row["current_method"]} '
              f'opposite={fmt(row["opposite_gauge"])}')
        for method in X.METHODS:
            value = enriched(row, method)
            marker = 'OK' if value['safe'] else '--'
            print(f'  {method:<22} {marker} g={fmt(value.get("gauge"))} '
                  f'd={fmt(value["delta"])} arcs={value.get("arcs", 0):>2} '
                  f'fits={value.get("arc_fits", 0):>2} '
                  f'{value.get("pick")}+{value.get("ridge")} '
                  f'ch={value.get("channel")} pol={value.get("polarity")}')

    print('\nchannel wins')
    for method in X.METHODS:
        wins = {}
        for row in rows:
            value = enriched(row, method)
            if value['safe']:
                channel = value.get('channel')
                wins[channel] = wins.get(channel, 0) + 1
        print(f'  {method:<24} {wins}')

    print('\ncascades (later methods run only on unresolved sides)')
    for methods in (
        ('adaptive_lab', 'sequential_k34_lab'),
        ('adaptive_lab', 'cross_lab', 'sequential_k34'),
        ('adaptive_lab', 'sequential_k34'),
        ('cross_lab', 'sequential_k34_lab', 'adaptive_lab'),
    ):
        count, fits, _ = cascade(rows, methods)
        print(f'  {" -> ".join(methods):<65} safe={count:>2} fits={fits:>3}')


if __name__ == '__main__':
    main()
