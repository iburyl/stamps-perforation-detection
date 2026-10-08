"""Report ranked all-partition k-means search on worst.jpg."""
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import METHODS, SIDES, degradations  # noqa: E402
from hybrid_report import summarize  # noqa: E402


def load(name):
    return json.loads((HERE / name).read_text())


def combine(search, ridge, key):
    out = []
    for trial, base in zip(search, ridge):
        out.append({
            'stamp': trial['stamp'],
            **{method: base[method]['raw'] for method in METHODS},
            'hybrid': trial[key],
        })
    return out


def row(label, results, attempts=None):
    summary = summarize(results)
    hurt, changes = degradations(results, 'hybrid')
    mean_attempts = (sum(attempts) / len(attempts) if attempts else None)
    print(f'{label:<15}{summary["published"]:>5}{summary["ok"]:>5}'
          f'{summary["p90"]:>9.3f}{summary["gross_axes"]:>7}'
          f'{len(hurt):>9}{changes["broke"]:>7}'
          f'{mean_attempts if mean_attempts is not None else 0:>10.2f}')
    if summary['bad']:
        print(' ' * 15 + 'bad: ' + ', '.join(
            f'{stamp}/{side}={gauge:.3f} ({source})'
            for stamp, side, gauge, _, source in summary['bad']
        ))
    return hurt


def actual_arc_fits(item, key, side):
    visited = item[f'{key}_attempts'][side]
    ordered = (item['candidates'][side]['stack']
               + item['candidates'][side]['close'])
    return sum(candidate['profile_score'] > 0
               for candidate in ordered[:visited])


def main():
    search = load('candidate_search_results.json')
    ridge = load('ridge_results.json')
    print('kmeans34 all partitions, pitch locked, stack then close')
    print(f'{"method":<15}{"pub":>5}{"ok":>5}{"p90":>9}{"gross":>7}'
          f'{"degraded":>9}{"broke":>7}{"mean fits":>10}')
    for threshold in range(5, 13):
        key = f'sequential_{threshold}'
        results = combine(search, ridge, key)
        attempts = [actual_arc_fits(item, key, side)
                    for item in search for side in SIDES]
        row(f'sequential {threshold}', results, attempts)
    oracle = combine(search, ridge, 'arc_oracle')
    row('arc oracle', oracle)

    results = combine(search, ridge, 'sequential_7')
    hurt = row('detail seq 7', results)
    print('\nsequential/7 degradations')
    for key, reasons in hurt.items():
        print(f'  {key[0]:>2}/{key[1]:<6} ' + '; '.join(reasons))
    print('\nunpublished sequential/7')
    for item in results:
        for side in SIDES:
            value = item['hybrid'][side]
            if not value['reliable']:
                print(f'  {item["stamp"]:>2}/{side:<6} '
                      f'{value["pick"]}+{value["ridge"]:<5} '
                      f'g={value["gauge"]!s:<18} arcs={value["arcs"]}')

    stamp = next(item for item in search if item['stamp'] == 27)
    print('\nstamp 27/left ranked candidates')
    for ridge_name in ('stack', 'close'):
        for value in stamp['candidates']['left'][ridge_name]:
            print(f'  {ridge_name:<5} #{value["rank"]} {value["pick"]:<4} '
                  f'score={value["profile_score"]:>6.2f} '
                  f'g={value["gauge"]!s:<18} arcs={value["arcs"]} '
                  f'reliable={value["reliable"]}')


if __name__ == '__main__':
    main()
