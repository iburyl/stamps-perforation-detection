"""Legacy ruler-delta report for selected scans; not validation."""
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import one_pass_report as R  # noqa: E402
import one_pass_tuning as T  # noqa: E402
from collection_report import manual_rows, summarize  # noqa: E402

SOURCES = ('1K', '2K', '3K')
CROSS_WEIGHTS = (1.10, .02, .08)


def load(name):
    return json.loads((HERE / name).read_text())


def unavailable():
    return {
        'status': 'unavailable', 'reliable': False, 'arcs': 0,
        'gauge': None, 'source': 'none',
    }


def one_pass_sides(item, method):
    out = {}
    for side in R.SIDES:
        candidates = item['candidates'][side]
        if method == 'cross support':
            candidate = T.selected_with_cross_support(
                candidates, *CROSS_WEIGHTS,
            )
        elif method == 'plain score':
            candidate = R.max_score(candidates, ('stack', 'close'))
        elif method == 'kmeans only':
            candidate = R.max_score(
                candidates, ('stack', 'close'),
                include_brightness=False,
            )
        else:
            raise ValueError(method)
        out[side] = R.result(candidate) if candidate else unavailable()
    return out


def samples(items, side_getter, exact=False):
    manual = manual_rows()
    rows = []
    for item in items:
        truth = manual[item['source']][item['stamp'] - 1]
        for side, value in side_getter(item).items():
            if exact and side not in ('top', 'left'):
                continue
            rows.append({
                'source': item['source'], 'stamp': item['stamp'],
                'side': side, 'target': truth[
                    'horizontal' if side in ('top', 'bottom') else 'vertical'
                ],
                'gauge': value['gauge'], 'reliable': value['reliable'],
                'arcs': value.get('arcs'), 'method': value.get('source'),
            })
    return rows


def row(source, label, rows, exact_rows):
    result = summarize(rows)
    exact = summarize(exact_rows)
    print(f'{source:<5}{label:<17}'
          f'{result["published"]:>3}/{result["total"]:<3}'
          f'{result["p90"]:>8.3f}{result["max"]:>8.3f}'
          f'{result["over_05"]:>6}'
          f'{exact["published"]:>3}/{exact["total"]:<3}'
          f'{exact["p90"]:>8.3f}{exact["max"]:>8.3f}')


def main():
    one_pass = load('one_pass_collection_results.json')
    multipass = load('candidate_search_collection_results.json')
    multi_by_key = {
        (item['source'], item['stamp']): item for item in multipass
    }
    print('WARNING: rough ruler measurements are not ground truth or tuning data')
    print('strict one-arc selector; legacy ruler-delta diagnostic')
    print(f'{"scan":<5}{"method":<17}{"pub":>7}{"p90":>8}{"max":>8}'
          f'{">.5":>6}{"direct":>7}{"D p90":>8}{"D max":>8}')
    for source in SOURCES:
        items = [item for item in one_pass if item['source'] == source]
        for label in ('cross support', 'plain score', 'kmeans only'):
            rows = samples(items, lambda item, name=label:
                           one_pass_sides(item, name))
            exact = [sample for sample in rows
                     if sample['side'] in ('top', 'left')]
            row(source, label, rows, exact)
        multi_items = [multi_by_key[(item['source'], item['stamp'])]
                       for item in items]
        rows = samples(
            multi_items, lambda item: item['sequential_7'],
        )
        exact = [sample for sample in rows
                 if sample['side'] in ('top', 'left')]
        row(source, 'multipass', rows, exact)

    print('\ncross-support details')
    for source in SOURCES:
        items = [item for item in one_pass if item['source'] == source]
        rows = samples(items, lambda item: one_pass_sides(
            item, 'cross support',
        ))
        print(f'\n{source}')
        for sample in rows:
            if not sample['reliable']:
                print(f'  unpublished {sample["stamp"]}/{sample["side"]:<6} '
                      f'{sample["method"]}, arcs={sample["arcs"]}')
                continue
            error = abs(sample['gauge'] - sample['target'])
            if error > .25:
                kind = ('direct-ruler' if sample['side'] in ('top', 'left')
                        else 'axis-proxy')
                print(f'  ruler delta {sample["stamp"]}/{sample["side"]:<6} '
                      f'{sample["gauge"]:.3f} vs {sample["target"]:.3f}, '
                      f'{error:.3f}, {sample["method"]}, {kind}')


if __name__ == '__main__':
    main()
