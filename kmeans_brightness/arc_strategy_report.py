"""Report the profile-pitch constraint experiment."""
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import METHODS, SIDES, degradations  # noqa: E402
from hybrid_report import choose_by_evidence, summarize  # noqa: E402


def load(name):
    return json.loads((HERE / name).read_text())


def with_reference(arc_results, ridge_results, strategy, candidate=None,
                   hybrid=False, strong_arcs=8,
                   fallback_strategy=None):
    out = []
    for arc, ridge in zip(arc_results, ridge_results):
        item = {method: ridge[method]['raw'] for method in METHODS}
        item['stamp'] = arc['stamp']
        if hybrid:
            item['hybrid'] = {}
            for side in SIDES:
                primary = arc[strategy]['brightness_close'][side]
                fallback = fallback_strategy or strategy
                fallbacks = [
                    ('kmeans34+close',
                     arc[fallback]['kmeans34_close'][side]),
                    ('kmeans34+stack',
                     arc[fallback]['kmeans34_stack'][side]),
                ]
                item['hybrid'][side] = choose_by_evidence(
                    primary, fallbacks, strong_arcs,
                )
        else:
            item['hybrid'] = {
                side: dict(arc[strategy][candidate][side], source=candidate)
                for side in SIDES
            }
        out.append(item)
    return out


def row(label, results):
    summary = summarize(results)
    hurt, changes = degradations(results, 'hybrid')
    print(f'{label:<28}{summary["published"]:>5}{summary["ok"]:>5}'
          f'{summary["arcs"]:>7}{summary["axes"]:>6}'
          f'{summary["median"]:>9.3f}{summary["p90"]:>9.3f}'
          f'{summary["within"]:>8.0%}{summary["gross_axes"]:>7}'
          f'{len(hurt):>10}{changes["broke"]:>7}')
    if summary['bad']:
        print(' ' * 28 + 'bad: ' + ', '.join(
            f'{stamp}/{side} {gauge:.3f} ({source})'
            for stamp, side, gauge, _, source in summary['bad']
        ))


def main():
    arc = load('arc_strategy_results.json')
    ridge = load('ridge_results.json')
    print(f'{"method":<28}{"pub":>5}{"ok":>5}{"arcs":>7}{"axes":>6}'
          f'{"median":>9}{"p90":>9}{"<=.25":>8}{"gross":>7}'
          f'{"degraded":>10}{"broke":>7}')
    for candidate in ('brightness_close', 'kmeans34_close',
                      'kmeans34_stack'):
        for strategy in ('current', 'pitch_lock', 'brightness_prior',
                         'prior_lock'):
            row(f'{candidate} {strategy}', with_reference(
                arc, ridge, strategy, candidate,
            ))
    for strategy in ('current', 'pitch_lock', 'brightness_prior',
                     'prior_lock'):
        row(f'hybrid8 {strategy}', with_reference(
            arc, ridge, strategy, hybrid=True,
        ))

    print('\nstrong-primary threshold sensitivity on worst.jpg\n')
    for label, primary, fallback in (
        ('current/current', 'current', 'current'),
        ('current/kmeans-lock', 'current', 'pitch_lock'),
        ('full-lock', 'pitch_lock', 'pitch_lock'),
    ):
        for strong_arcs in range(5, 13):
            row(f'{label} {strong_arcs}', with_reference(
                arc, ridge, primary, hybrid=True,
                strong_arcs=strong_arcs, fallback_strategy=fallback,
            ))


if __name__ == '__main__':
    main()
