"""Legacy ruler-delta comparison of fallback selection policies."""
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from arc_strategy_report import with_reference  # noqa: E402
from collection_report import manual_rows, summarize as manual_summary  # noqa: E402
from common import SIDES, degradations  # noqa: E402
from hybrid_report import (choose_by_evidence, choose_sequential,  # noqa: E402
                           summarize as worst_summary)


def load(name):
    return json.loads((HERE / name).read_text())


def choose(policy, primary, candidates):
    if policy == 'max':
        return choose_by_evidence(primary, candidates, 8)
    return choose_sequential(primary, candidates, 8)


def selected(item, policy, order):
    sides = {}
    second_calls = 0
    for side in SIDES:
        primary = item['pitch_lock']['brightness_close'][side]
        candidates = [
            (f'kmeans34+{name}', item['pitch_lock'][f'kmeans34_{name}'][side])
            for name in order
        ]
        primary_weak = (not primary['reliable']
                        or (primary['arcs'] or 0) < 8)
        if policy == 'max':
            second_calls += int(primary_weak)
        elif primary_weak:
            first = candidates[0][1]
            second_calls += int(
                not first['reliable'] or (first['arcs'] or 0) < 8
            )
        sides[side] = choose(policy, primary, candidates)
    return sides, second_calls


def collection_samples(results, policy, order):
    manual = manual_rows()
    samples, second_calls = [], 0
    for item in results:
        sides, extra = selected(item, policy, order)
        second_calls += extra
        truth = manual[item['source']][item['stamp'] - 1]
        for side, value in sides.items():
            target = truth['horizontal' if side in ('top', 'bottom')
                           else 'vertical']
            samples.append({
                'target': target, 'gauge': value['gauge'],
                'reliable': value['reliable'],
            })
    return manual_summary(samples), second_calls


def worst_results(arc, ridge, policy, order):
    reference = with_reference(
        arc, ridge, 'pitch_lock', hybrid=True, strong_arcs=8,
    )
    second_calls = 0
    for output, item in zip(reference, arc):
        output['hybrid'], extra = selected(item, policy, order)
        second_calls += extra
    return reference, second_calls


def main():
    collection = load('collection_results.json')
    arc = load('arc_strategy_results.json')
    ridge = load('ridge_results.json')
    print('WARNING: rough ruler measurements are not ground truth or tuning data')
    print(f'{"policy":<18}{"2nd C/W":>10}{"C pub":>7}{"C p90":>8}'
          f'{"C >.5":>7}{"C max":>8}{"W pub":>7}{"W p90":>8}'
          f'{"W gross":>9}{"W degr":>8}')
    for policy, order in (
        ('max', ('close', 'stack')),
        ('sequential', ('close', 'stack')),
        ('sequential', ('stack', 'close')),
    ):
        label = f'{policy} {order[0]}->{order[1]}'
        manual, c_second = collection_samples(collection, policy, order)
        worst, w_second = worst_results(arc, ridge, policy, order)
        ws = worst_summary(worst)
        hurt, _ = degradations(worst, 'hybrid')
        print(f'{label:<18}{c_second:>4}/{w_second:<5}'
              f'{manual["published"]:>7}{manual["p90"]:>8.3f}'
              f'{manual["over_05"]:>7}{manual["max"]:>8.3f}'
              f'{ws["published"]:>7}{ws["p90"]:>8.3f}'
              f'{ws["gross_axes"]:>9}{len(hurt):>8}')


if __name__ == '__main__':
    main()
