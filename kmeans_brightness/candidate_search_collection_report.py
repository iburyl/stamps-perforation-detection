"""Legacy ruler-delta diagnostic for ranked k-means search; not validation."""
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from collection_report import manual_rows, summarize  # noqa: E402
from common import SIDES  # noqa: E402


def samples(results, threshold, exact=False):
    manual = manual_rows()
    rows = []
    for item in results:
        truth = manual[item['source']][item['stamp'] - 1]
        for side in SIDES:
            if exact and side not in ('top', 'left'):
                continue
            value = item[f'sequential_{threshold}'][side]
            rows.append({
                'source': item['source'], 'stamp': item['stamp'],
                'side': side,
                'target': truth['horizontal' if side in ('top', 'bottom')
                                else 'vertical'],
                'gauge': value['gauge'],
                'reliable': value['reliable'],
                'arcs': value['arcs'], 'pick': value['pick'],
                'ridge': value['ridge'],
            })
    return rows


def actual_arc_fits(item, threshold, side):
    """Profiles with valleys that reach the expensive greyscale arc pass."""
    visited = item[f'sequential_{threshold}_attempts'][side]
    ordered = (item['candidates'][side]['stack']
               + item['candidates'][side]['close'])
    return sum(candidate['profile_score'] > 0
               for candidate in ordered[:visited])


def main():
    results = json.loads(
        (HERE / 'candidate_search_collection_results.json').read_text()
    )
    print('WARNING: rough ruler measurements are not ground truth or tuning data')
    print('ranked kmeans34 only; 80-stamp legacy ruler-delta diagnostic')
    print(f'{"arcs":>6}{"pub":>9}{"median":>9}{"p90":>9}'
          f'{"<=.25":>8}{">.5":>6}{">1":>5}{"max":>9}'
          f'{"direct":>9}{"D p90":>9}{"D max":>9}{"fits":>8}')
    for threshold in range(5, 13):
        all_stats = summarize(samples(results, threshold))
        exact_stats = summarize(samples(results, threshold, exact=True))
        fits = [actual_arc_fits(item, threshold, side)
                for item in results for side in SIDES]
        mean_fits = sum(fits) / len(fits)
        print(f'{threshold:>6}'
              f'{all_stats["published"]:>4}/{all_stats["total"]:<4}'
              f'{all_stats["median"]:>9.3f}{all_stats["p90"]:>9.3f}'
              f'{all_stats["within_025"]:>8.0%}{all_stats["over_05"]:>6}'
              f'{all_stats["over_1"]:>5}{all_stats["max"]:>9.3f}'
              f'{exact_stats["published"]:>4}/{exact_stats["total"]:<4}'
              f'{exact_stats["p90"]:>9.3f}{exact_stats["max"]:>9.3f}'
              f'{mean_fits:>8.2f}')

    print('\nthreshold 7 ruler deltas over 0.25')
    for row in samples(results, 7):
        if (row['reliable']
                and abs(row['gauge'] - row['target']) > .25):
            exact = ('direct-ruler' if row['side'] in ('top', 'left')
                     else 'axis-proxy')
            print(f'  {row["source"]}/{row["stamp"]} {row["side"]:<6} '
                  f'{row["gauge"]:.3f} vs {row["target"]:.3f} '
                  f'delta={abs(row["gauge"] - row["target"]):.3f} '
                  f'{row["pick"]}+{row["ridge"]}, {row["arcs"]} arcs, '
                  f'{exact}')


if __name__ == '__main__':
    main()
