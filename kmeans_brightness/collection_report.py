"""Legacy diagnostic against rough ruler measurements; not validation."""
import csv
import json
import os
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import SIDES  # noqa: E402
from hybrid_report import choose_by_evidence  # noqa: E402

MANUAL = pathlib.Path(os.environ.get(
    'STAMPS_MANUAL_CSV',
    r'D:\workspace\IX vs X analysis\IX vs X - Main-Stripped.csv',
))


def manual_rows():
    grouped = {}
    with MANUAL.open(newline='', encoding='utf-8-sig') as file:
        for row in csv.DictReader(file):
            grouped.setdefault(row['V'], []).append({
                'id': row['Id'],
                'horizontal': 220. / float(row['Prf.U']),
                'vertical': 320. / float(row['Prf.L']),
            })
    return grouped


def hybrid(item, primary_strategy, fallback_strategy, strong_arcs=8):
    out = {}
    for side in SIDES:
        primary = item[primary_strategy]['brightness_close'][side]
        fallbacks = [
            ('kmeans34+close',
             item[fallback_strategy]['kmeans34_close'][side]),
            ('kmeans34+stack',
             item[fallback_strategy]['kmeans34_stack'][side]),
        ]
        out[side] = choose_by_evidence(primary, fallbacks, strong_arcs)
    return out


def methods(item):
    return {
        'brightness raw': item['current']['brightness_raw'],
        'brightness close': item['current']['brightness_close'],
        'hybrid current': hybrid(item, 'current', 'current'),
        'hybrid kmeans-lock': hybrid(item, 'current', 'pitch_lock'),
        'hybrid full-lock': hybrid(item, 'pitch_lock', 'pitch_lock'),
    }


def collect(results, manual, primary_strategy, fallback_strategy,
            strong_arcs):
    samples = []
    for item in results:
        truth = manual[item['source']][item['stamp'] - 1]
        sides = hybrid(item, primary_strategy, fallback_strategy, strong_arcs)
        for side in SIDES:
            value = sides[side]
            samples.append({
                'target': truth['horizontal' if side in ('top', 'bottom')
                                else 'vertical'],
                'gauge': value['gauge'], 'reliable': value['reliable'],
                'side': side,
            })
    return samples


def quantile(values, fraction):
    values = sorted(values)
    return values[round((len(values) - 1) * fraction)] if values else None


def summarize(samples):
    errors = [abs(sample['gauge'] - sample['target'])
              for sample in samples if sample['reliable']]
    return {
        'published': sum(sample['reliable'] for sample in samples),
        'total': len(samples),
        'median': statistics.median(errors),
        'p90': quantile(errors, .9),
        'within_025': sum(error <= .25 for error in errors) / len(errors),
        'over_05': sum(error > .5 for error in errors),
        'over_1': sum(error > 1 for error in errors),
        'max': max(errors),
    }


def main():
    results = json.loads((HERE / 'collection_results.json').read_text())
    manual = manual_rows()
    samples = {}
    worst = {}
    for item in results:
        truth = manual[item['source']][item['stamp'] - 1]
        for name, sides in methods(item).items():
            for side in SIDES:
                value = sides[side]
                target = truth['horizontal' if side in ('top', 'bottom')
                               else 'vertical']
                sample = {
                    'source': item['source'], 'stamp': item['stamp'],
                    'id': truth['id'], 'side': side, 'target': target,
                    'gauge': value['gauge'], 'reliable': value['reliable'],
                    'arcs': value['arcs'],
                }
                samples.setdefault(name, []).append(sample)
                if value['reliable']:
                    error = abs(value['gauge'] - target)
                    if name not in worst or error > worst[name][0]:
                        worst[name] = (error, sample)
    print('WARNING: rough ruler measurements are not ground truth or tuning data')
    print('80 stamps, 320 sides, legacy ruler-delta diagnostic')
    print()
    print(f'{"method":<21}{"pub":>9}{"median":>9}{"p90":>9}'
          f'{"<=.25":>8}{">.5":>6}{">1":>5}{"max":>9}')
    for name, rows in samples.items():
        result = summarize(rows)
        print(f'{name:<21}{result["published"]:>4}/{result["total"]:<4}'
              f'{result["median"]:>9.3f}{result["p90"]:>9.3f}'
              f'{result["within_025"]:>8.0%}{result["over_05"]:>6}'
              f'{result["over_1"]:>5}{result["max"]:>9.3f}')
        error, sample = worst[name]
        print(' ' * 21 + f'worst {sample["source"]}/{sample["stamp"]}'
              f' ({sample["id"]}) {sample["side"]}: '
              f'{sample["gauge"]:.3f} vs {sample["target"]:.3f}, '
              f'{sample["arcs"]} arcs')

    print('\ndirectly ruler-measured sides only (top and left; still approximate)\n')
    print(f'{"method":<21}{"pub":>9}{"median":>9}{"p90":>9}'
          f'{"<=.25":>8}{">.5":>6}{">1":>5}{"max":>9}')
    for name, rows in samples.items():
        result = summarize([
            row for row in rows if row['side'] in ('top', 'left')
        ])
        print(f'{name:<21}{result["published"]:>4}/{result["total"]:<4}'
              f'{result["median"]:>9.3f}{result["p90"]:>9.3f}'
              f'{result["within_025"]:>8.0%}{result["over_05"]:>6}'
              f'{result["over_1"]:>5}{result["max"]:>9.3f}')

    print('\nstrong-primary threshold sensitivity\n')
    print(f'{"strategies":<23}{"arcs":>6}{"pub":>9}{"median":>9}'
          f'{"p90":>9}{"<=.25":>8}{">.5":>6}{">1":>5}{"max":>9}')
    for label, primary, fallback in (
        ('current/current', 'current', 'current'),
        ('current/kmeans-lock', 'current', 'pitch_lock'),
        ('full-lock', 'pitch_lock', 'pitch_lock'),
    ):
        for strong_arcs in range(5, 13):
            result = summarize(collect(
                results, manual, primary, fallback, strong_arcs,
            ))
            print(f'{label:<23}{strong_arcs:>6}'
                  f'{result["published"]:>4}/{result["total"]:<4}'
                  f'{result["median"]:>9.3f}{result["p90"]:>9.3f}'
                  f'{result["within_025"]:>8.0%}{result["over_05"]:>6}'
                  f'{result["over_1"]:>5}{result["max"]:>9.3f}')


if __name__ == '__main__':
    main()
