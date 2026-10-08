"""Legacy ruler-delta exploration of a k-means-only primary method."""
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from arc_strategy_report import with_reference  # noqa: E402
from collection_report import manual_rows, summarize as manual_summary  # noqa: E402
from common import METHODS, SIDES, degradations  # noqa: E402
from hybrid_report import (choose_by_evidence, choose_sequential,  # noqa: E402
                           summarize as worst_summary)

EMPTY = {
    'status': 'unavailable', 'reliable': False, 'arcs': 0,
    'gauge': None, 'geometry_source': None,
}


def load(name):
    return json.loads((HERE / name).read_text())


def select(item, policy, strong_arcs):
    candidates = [
        ('kmeans34+stack', item['pitch_lock']['kmeans34_stack']),
        ('kmeans34+close', item['pitch_lock']['kmeans34_close']),
    ]
    sides = {}
    close_calls = 0
    for side in SIDES:
        per_side = [(name, values[side]) for name, values in candidates]
        if policy == 'max':
            sides[side] = choose_by_evidence(EMPTY, per_side, strong_arcs)
            close_calls += 1
        else:
            sides[side] = choose_sequential(EMPTY, per_side, strong_arcs)
            stack = per_side[0][1]
            close_calls += int(
                not stack['reliable'] or (stack['arcs'] or 0) < strong_arcs
            )
    return sides, close_calls


def collection_stats(results, policy, strong_arcs, exact=False):
    manual = manual_rows()
    samples, close_calls = [], 0
    for item in results:
        sides, calls = select(item, policy, strong_arcs)
        close_calls += calls
        truth = manual[item['source']][item['stamp'] - 1]
        for side, value in sides.items():
            if exact and side not in ('top', 'left'):
                continue
            samples.append({
                'target': truth['horizontal' if side in ('top', 'bottom')
                                else 'vertical'],
                'gauge': value['gauge'], 'reliable': value['reliable'],
            })
    return manual_summary(samples), close_calls


def worst_stats(arc, ridge, policy, strong_arcs):
    reference = with_reference(
        arc, ridge, 'pitch_lock', 'kmeans34_stack', hybrid=False,
    )
    close_calls = 0
    for output, item in zip(reference, arc):
        output['hybrid'], calls = select(item, policy, strong_arcs)
        close_calls += calls
    summary = worst_summary(reference)
    hurt, changes = degradations(reference, 'hybrid')
    return summary, hurt, changes, close_calls


def main():
    collection = load('collection_results.json')
    arc = load('arc_strategy_results.json')
    ridge = load('ridge_results.json')
    print('WARNING: rough ruler measurements are not ground truth or tuning data')
    print('kmeans34 only, pitch locked, stack then close')
    print()
    print(f'{"policy":<11}{"arcs":>6}{"close C/W":>11}'
          f'{"C pub":>7}{"C p90":>8}{"C >.5":>7}{"C >1":>6}'
          f'{"direct":>8}{"D p90":>8}{"D max":>8}'
          f'{"W pub":>7}{"W p90":>8}{"W gross":>9}'
          f'{"W degr":>8}{"broke":>7}')
    for policy in ('max', 'sequential'):
        for strong_arcs in range(5, 13):
            manual, c_calls = collection_stats(
                collection, policy, strong_arcs,
            )
            exact, _ = collection_stats(
                collection, policy, strong_arcs, exact=True,
            )
            worst, hurt, changes, w_calls = worst_stats(
                arc, ridge, policy, strong_arcs,
            )
            print(f'{policy:<11}{strong_arcs:>6}{c_calls:>5}/{w_calls:<5}'
                  f'{manual["published"]:>7}{manual["p90"]:>8.3f}'
                  f'{manual["over_05"]:>7}{manual["over_1"]:>6}'
                  f'{exact["published"]:>4}/{exact["total"]:<3}'
                  f'{exact["p90"]:>8.3f}{exact["max"]:>8.3f}'
                  f'{worst["published"]:>7}{worst["p90"]:>8.3f}'
                  f'{worst["gross_axes"]:>9}{len(hurt):>8}'
                  f'{changes["broke"]:>7}')

    reference = with_reference(
        arc, ridge, 'pitch_lock', 'kmeans34_stack', hybrid=False,
    )
    for output, item in zip(reference, arc):
        output['hybrid'], _ = select(item, 'sequential', 7)
    hurt, _ = degradations(reference, 'hybrid')
    print('\nsequential/7 degradations against published brightness baseline')
    for (stamp, side), reasons in hurt.items():
        value = reference[stamp - 1]['hybrid'][side]
        print(f'  {stamp:>2}/{side:<6} {value["source"]:<16} '
              f'g={value["gauge"]!s:<18} arcs={value["arcs"]}: '
              + '; '.join(reasons))
    print('\nunpublished worst sides under sequential/7')
    for result in reference:
        for side in SIDES:
            value = result['hybrid'][side]
            if not value['reliable']:
                print(f'  {result["stamp"]:>2}/{side:<6} '
                      f'{value["source"]:<16} g={value["gauge"]!s:<18} '
                      f'arcs={value["arcs"]} status={value["status"]}')


if __name__ == '__main__':
    main()
