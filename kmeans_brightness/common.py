"""Shared reading of results.json and the definition of a degradation."""
import json
import pathlib
import statistics

OUT = pathlib.Path(__file__).resolve().parent
SIDES = ('top', 'bottom', 'left', 'right')
METHODS = ('brightness', 'kmeans3', 'kmeans4', 'kmeans34')
AXES = (('top', 'bottom'), ('left', 'right'))
GROSS = 1.0
RANK = {'unavailable': 0, 'rejected': 0, None: 0, 'review': 1, 'ok': 2}


def load():
    return json.loads((OUT / 'results.json').read_text())


def published(side):
    return side['reliable']


def show(value, spec='.3f'):
    return '--' if value is None else format(value, spec)


def reference_gauge(result, side):
    """What this side most likely measures, judged by the rest of the stamp.

    The three other sides of the stamp, pooled over all four methods. A
    harmonic is rare enough per stamp that the median of around twenty values
    is not fooled by one, which makes this a usable arbiter when two methods
    differ by a factor of two and neither can be assumed right.
    """
    others = [result[m][s]['gauge'] for m in METHODS for s in SIDES
              if s != side and published(result[m][s])]
    return statistics.median(others) if len(others) >= 4 else None


def wrong(value, reference):
    return reference is not None and abs(value - reference) > GROSS


def rescued(result, side, method, base='brightness'):
    """True when the method replaced a wrong published baseline value.

    Such a side can still look worse on paper, because a correct measurement
    carrying fewer arcs is demoted to review while a confident harmonic was
    marked ok. It is not a degradation and is kept out of the list.
    """
    a, b = result[base][side], result[method][side]
    if a['gauge'] is None or b['gauge'] is None or not published(a):
        return False
    reference = reference_gauge(result, side)
    return wrong(a['gauge'], reference) and not wrong(b['gauge'], reference)


def degradations(results, method, base='brightness'):
    """Sides where the method is worse than the baseline, with the reasons.

    Returns {(stamp, side): [reason, ...]} plus the counts that summarise the
    whole comparison, since the two are derived from one pass.
    """
    hurt, counts = {}, dict(gained=0, lost=0, up=0, down=0, fixed=0, broke=0)
    for result in results:
        for side in SIDES:
            a, b = result[base][side], result[method][side]
            key = (result['stamp'], side)
            counts['gained'] += published(b) and not published(a)
            if rescued(result, side, method, base):
                counts['fixed'] += 1
                continue
            if published(a) and not published(b):
                counts['lost'] += 1
                # Losing publication is not the same as losing the answer: the
                # side usually still measures the right pitch and merely falls
                # under the five-arc evidence bar. Say which happened.
                reference = reference_gauge(result, side)
                if b['gauge'] is None:
                    verdict = 'no pitch at all'
                elif wrong(a['gauge'], reference):
                    verdict = (f"but the baseline was wrong anyway "
                               f"(stamp reads {show(reference)})")
                elif not wrong(b['gauge'], reference):
                    verdict = f"value still right at {show(b['gauge'])}"
                else:
                    verdict = f"and the value went wrong, {show(b['gauge'])}"
                hurt.setdefault(key, []).append(
                    f"lost publication: baseline {show(a['gauge'])} with "
                    f"{a['arcs']} arcs -> {b['arcs']} arcs, {verdict}")
            counts['up'] += RANK[b['status']] > RANK[a['status']]
            if RANK[b['status']] < RANK[a['status']]:
                counts['down'] += 1
                if published(a) and published(b):
                    reference = reference_gauge(result, side)
                    if wrong(a['gauge'], reference) and wrong(b['gauge'],
                                                              reference):
                        aside = (f"both wrong, stamp reads {show(reference)}, "
                                 f"so less confidence is warranted")
                    else:
                        aside = (f"{show(a['gauge'])} -> {show(b['gauge'])}, "
                                 f"{a['arcs']} -> {b['arcs']} arcs")
                    hurt.setdefault(key, []).append(
                        f"status {a['status']} -> {b['status']} ({aside})")
            if published(a) and published(b) \
                    and abs(a['gauge'] - b['gauge']) > GROSS:
                reference = reference_gauge(result, side)
                if wrong(b['gauge'], reference):
                    counts['broke'] += 1
                    hurt.setdefault(key, []).append(
                        f"gross: {show(a['gauge'])} -> {show(b['gauge'])} "
                        f"(stamp reads {show(reference)})")
    return hurt, counts
