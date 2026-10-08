"""Validate experimental selectors without treating hand measurements as truth.

The only external hypothesis used here is that the horizontal/vertical gauge pair is
either 14.50 x 15.00 or 14.25 x 14.75.  Because the class of each stamp is not known,
distance to the nearest pair is only a plausibility check, not an accuracy estimate.
The stronger checks are opposite-side agreement, horizontal/vertical class coherence,
coverage, and agreement between independent selection policies.
"""
import json
import math
import pathlib
import random
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import one_pass_report as one  # noqa: E402

SIDES = ('top', 'bottom', 'left', 'right')
HORIZONTAL = ('top', 'bottom')
VERTICAL = ('left', 'right')
THEORY = {
    '14.5x15': (14.50, 15.00),
    '14.25x14.75': (14.25, 14.75),
}


def load(name):
    return json.loads((HERE / name).read_text())


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * q
    lo = math.floor(index)
    hi = math.ceil(index)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - index) + values[hi] * (index - lo)


def fmt(value, width=7, decimals=3):
    return (f'{value:>{width}.{decimals}f}' if value is not None
            else f'{"-":>{width}}')


def cross_support_sides(item, bonus=1.1, pitch_tol=.02,
                        line_fraction=.08):
    # This policy is deliberately profile-only.  The arcs are read only after the
    # candidate is selected, so there is still exactly one expensive pass.
    out = {}
    for side in SIDES:
        pool = [candidate for candidate in item['candidates'][side]
                if candidate['map'] in one.BASE_MAPS and one.usable(candidate)]

        def cross_supported(candidate):
            profile = candidate['profile']
            candidate_is_brightness = candidate['map'] == 'brightness'
            for other in pool:
                if (other['map'] == 'brightness') == candidate_is_brightness:
                    continue
                other_profile = other['profile']
                if (abs(math.log(other_profile['pitch_px']
                                 / profile['pitch_px'])) <= pitch_tol
                        and abs(other_profile['line_center']
                                - profile['line_center'])
                        <= line_fraction * profile['pitch_px']):
                    return True
            return False

        candidate = max(
            pool,
            key=lambda value: value['profile']['score']
            * (bonus if cross_supported(value) else 1.),
        ) if pool else None
        out[side] = one.result(candidate) if candidate else {
            'status': 'unavailable', 'reliable': False, 'arcs': 0,
            'gauge': None, 'source': 'none',
        }
    return out


def one_pass_sides(item):
    return cross_support_sides(item)


def multipass_sides(item):
    return item['sequential_7']


def score_both_sides(item):
    return one.selected_sides(item, ('score', ('stack', 'close')))


def score_kmeans_sides(item):
    return one.selected_sides(item, ('score_kmeans', ('stack', 'close')))


def brightness_stack_sides(item):
    return one.selected_sides(item, ('fixed', 'brightness', 'stack'))


def published(value):
    return bool(value.get('reliable') and value.get('gauge') is not None)


def axis_value(sides, names, require_both=False):
    values = [sides[name]['gauge'] for name in names
              if published(sides[name])]
    if require_both and len(values) != 2:
        return None
    return statistics.fmean(values) if values else None


def nearest_axis_class(value, axis):
    position = 0 if axis == 'horizontal' else 1
    return min(THEORY, key=lambda name: abs(value - THEORY[name][position]))


def summarize(items, getter):
    side_values = []
    opposite = []
    axis_records = []
    complete = 0
    theory_side_residuals = []
    implausible = []

    for item in items:
        sides = getter(item)
        for side in SIDES:
            value = sides[side]
            if not published(value):
                continue
            gauge = value['gauge']
            side_values.append(gauge)
            targets = ([pair[0] for pair in THEORY.values()]
                       if side in HORIZONTAL
                       else [pair[1] for pair in THEORY.values()])
            residual = min(abs(gauge - target) for target in targets)
            theory_side_residuals.append(residual)
            if residual > .50:
                implausible.append((item.get('source', 'worst'), item['stamp'],
                                    side, gauge, residual,
                                    value.get('source') or value.get('pick')))

        for names in (HORIZONTAL, VERTICAL):
            if all(published(sides[name]) for name in names):
                opposite.append(abs(sides[names[0]]['gauge']
                                    - sides[names[1]]['gauge']))

        if all(published(sides[name]) for name in SIDES):
            complete += 1
            h = axis_value(sides, HORIZONTAL, require_both=True)
            v = axis_value(sides, VERTICAL, require_both=True)
            distances = {
                name: math.hypot(h - target_h, v - target_v)
                for name, (target_h, target_v) in THEORY.items()
            }
            nearest = min(distances, key=distances.get)
            h_class = nearest_axis_class(h, 'horizontal')
            v_class = nearest_axis_class(v, 'vertical')
            axis_records.append({
                'source': item.get('source', 'worst'),
                'stamp': item['stamp'], 'horizontal': h, 'vertical': v,
                'nearest': nearest, 'distance': distances[nearest],
                'margin': abs(distances['14.5x15']
                              - distances['14.25x14.75']),
                'coherent': h_class == v_class,
                'horizontal_class': h_class, 'vertical_class': v_class,
            })

    return {
        'published': len(side_values), 'total': len(items) * 4,
        'complete': complete,
        'opposite_pairs': len(opposite),
        'opposite_median': statistics.median(opposite) if opposite else None,
        'opposite_p90': percentile(opposite, .90),
        'opposite_max': max(opposite) if opposite else None,
        'opposite_over_025': sum(value > .25 for value in opposite),
        'side_theory_p90': percentile(theory_side_residuals, .90),
        'side_theory_max': max(theory_side_residuals)
                           if theory_side_residuals else None,
        'implausible': implausible,
        'axis_records': axis_records,
        'coherent': sum(record['coherent'] for record in axis_records),
        'axis_theory_p90': percentile(
            [record['distance'] for record in axis_records], .90),
        'axis_theory_max': max(
            (record['distance'] for record in axis_records), default=None),
        'class_counts': {
            name: sum(record['nearest'] == name for record in axis_records)
            for name in THEORY
        },
    }


def compare(items, left_getter, right_getter):
    differences = []
    left_only = []
    right_only = []
    for item in items:
        left = left_getter(item)
        right = right_getter(item)
        for side in SIDES:
            lp = published(left[side])
            rp = published(right[side])
            key = (item.get('source', 'worst'), item['stamp'], side)
            if lp and rp:
                differences.append((abs(left[side]['gauge']
                                        - right[side]['gauge']), key,
                                    left[side]['gauge'], right[side]['gauge']))
            elif lp:
                left_only.append(key)
            elif rp:
                right_only.append(key)
    values = [row[0] for row in differences]
    return {
        'both': len(values),
        'median': statistics.median(values) if values else None,
        'p90': percentile(values, .90),
        'max': max(values) if values else None,
        'over_025': sum(value > .25 for value in values),
        'left_only': left_only, 'right_only': right_only,
        'largest': sorted(differences, reverse=True)[:12],
    }


def blind_two_clusters(records):
    """Fit two clusters without using either theoretical pair."""
    points = [(record['horizontal'], record['vertical'])
              for record in records]
    if len(points) < 2:
        return None
    centers = [min(points, key=lambda point: point[0] + point[1]),
               max(points, key=lambda point: point[0] + point[1])]
    assignments = None
    for _ in range(100):
        new_assignments = [
            min(range(2), key=lambda index: math.hypot(
                point[0] - centers[index][0],
                point[1] - centers[index][1],
            ))
            for point in points
        ]
        groups = [[point for point, assignment in zip(points, new_assignments)
                   if assignment == index] for index in range(2)]
        if not all(groups):
            return None
        new_centers = [
            (statistics.fmean(point[0] for point in group),
             statistics.fmean(point[1] for point in group))
            for group in groups
        ]
        if assignments == new_assignments:
            centers = new_centers
            break
        assignments = new_assignments
        centers = new_centers
    order = sorted(range(2), key=lambda index: sum(centers[index]))
    centers = [centers[index] for index in order]
    groups = [[points[position] for position, assignment
               in enumerate(assignments) if assignment == index]
              for index in order]
    squared = [
        (point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2
        for center, group in zip(centers, groups) for point in group
    ]
    return {
        'centers': centers,
        'sizes': [len(group) for group in groups],
        'rms': math.sqrt(statistics.fmean(squared)),
        'separation': math.dist(*centers),
    }


def bootstrap_clusters(records, repetitions=1000):
    rng = random.Random(20261007)
    fits = []
    for _ in range(repetitions):
        sample = [records[rng.randrange(len(records))]
                  for _ in range(len(records))]
        fit = blind_two_clusters(sample)
        if fit:
            fits.append(fit['centers'])
    intervals = []
    for cluster in range(2):
        intervals.append(tuple(
            (percentile([fit[cluster][axis] for fit in fits], .05),
             percentile([fit[cluster][axis] for fit in fits], .95))
            for axis in range(2)
        ))
    return intervals


def print_blind_clusters(label, items, getter):
    stats = summarize(items, getter)
    fit = blind_two_clusters(stats['axis_records'])
    if not fit:
        print(f'{label}: insufficient complete stamps')
        return
    intervals = bootstrap_clusters(stats['axis_records'])
    print(f'{label}: n={sum(fit["sizes"])} sizes={fit["sizes"]} '
          f'within-rms={fit["rms"]:.3f} separation={fit["separation"]:.3f}')
    expected = (THEORY['14.25x14.75'], THEORY['14.5x15'])
    for index, (center, bounds, target) in enumerate(zip(
            fit['centers'], intervals, expected), 1):
        print(f'  cluster {index}: H={center[0]:.3f} '
              f'[p05 {bounds[0][0]:.3f}, p95 {bounds[0][1]:.3f}], '
              f'V={center[1]:.3f} '
              f'[p05 {bounds[1][0]:.3f}, p95 {bounds[1][1]:.3f}]; '
              f'post-hoc delta from theory=({center[0] - target[0]:+.3f}, '
              f'{center[1] - target[1]:+.3f})')


def print_summary(label, stats):
    print(f'{label:<18}'
          f'{stats["published"]:>4}/{stats["total"]:<4}'
          f'{stats["complete"]:>7}'
          f'{fmt(stats["opposite_p90"], 8)}'
          f'{fmt(stats["opposite_max"], 8)}'
          f'{stats["opposite_over_025"]:>7}'
          f'{stats["coherent"]:>5}/{len(stats["axis_records"]):<4}'
          f'{fmt(stats["axis_theory_p90"], 8)}'
          f'{len(stats["implausible"]):>8}')


def source_rows(collection, source):
    return [item for item in collection if item['source'].upper() == source]


def main():
    one_collection = load('one_pass_collection_results.json')
    multi_collection = load('candidate_search_collection_results.json')
    one_worst = load('one_pass_results.json')
    multi_worst = load('candidate_search_results.json')

    multi_by_key = {
        (item.get('source'), item['stamp']): item for item in multi_collection
    }
    paired_collection = []
    for item in one_collection:
        key = (item.get('source'), item['stamp'])
        paired_collection.append((item, multi_by_key[key]))

    print('No manual measurements are used.')
    print('Theory is a hypothesis: HxV = 14.50x15.00 or 14.25x14.75.')
    print('Nearest-theory distance is plausibility only because stamp classes are unknown.\n')
    print(f'{"dataset/method":<18}{"published":>9}{"4-side":>7}'
          f'{"opp p90":>8}{"opp max":>8}{">.25":>7}'
          f'{"coherent":>9}{"pair p90":>8}{"outlier":>8}')

    for source in ('1K', '2K', '3K'):
        one_items = source_rows(one_collection, source)
        multi_items = source_rows(multi_collection, source)
        print_summary(f'{source} one-pass', summarize(
            one_items, one_pass_sides))
        print_summary(f'{source} multipass', summarize(
            multi_items, multipass_sides))
    print_summary('all one-pass', summarize(one_collection, one_pass_sides))
    print_summary('all multipass', summarize(multi_collection,
                                              multipass_sides))
    print_summary('worst one-pass', summarize(one_worst, one_pass_sides))
    print_summary('worst multipass', summarize(multi_worst,
                                                multipass_sides))

    print('\nblind two-cluster test (theory values are not used for fitting)')
    print_blind_clusters('collection one-pass', one_collection,
                         one_pass_sides)
    print_blind_clusters('collection multipass', multi_collection,
                         multipass_sides)
    for source in ('1K', '2K', '3K'):
        print_blind_clusters(f'{source} one-pass',
                             source_rows(one_collection, source),
                             one_pass_sides)

    print('\npre-arc selectors, judged without manual labels')
    print(f'{"dataset/method":<24}{"published":>9}{"4-side":>7}'
          f'{"opp p90":>8}{"opp max":>8}{">.25":>7}'
          f'{"coherent":>9}{"pair p90":>8}{"outlier":>8}')
    for name, getter in (
            ('brightness stack', brightness_stack_sides),
            ('max profile score', score_both_sides),
            ('max score kmeans', score_kmeans_sides),
            ('cross-support 1.1', one_pass_sides)):
        print_summary(f'C {name}', summarize(one_collection, getter))
        print_summary(f'W {name}', summarize(one_worst, getter))

    sensitivity = []
    for bonus in (1.05, 1.10, 1.20):
        for pitch_tol in (.02, .04, .08):
            for line_fraction in (.08, .15):
                getter = lambda item, b=bonus, p=pitch_tol, line=line_fraction: (
                    cross_support_sides(item, b, p, line)
                )
                sensitivity.append((
                    bonus, pitch_tol, line_fraction,
                    summarize(one_collection, getter),
                    summarize(one_worst, getter),
                ))
    print('\ncross-support sensitivity (18 profile-only settings)')
    print('  collection published range: '
          f'{min(row[3]["published"] for row in sensitivity)}..'
          f'{max(row[3]["published"] for row in sensitivity)} / 320')
    print('  worst published range: '
          f'{min(row[4]["published"] for row in sensitivity)}..'
          f'{max(row[4]["published"] for row in sensitivity)} / 108')
    print('  settings with no side residual > .50: '
          f'{sum(not row[3]["implausible"] and not row[4]["implausible"] for row in sensitivity)}'
          f'/{len(sensitivity)}')

    print('\nmultipass stopping threshold on the six-scan collection')
    print(f'{"arcs":>4}{"published":>11}{"opp p90":>9}{"opp max":>9}'
          f'{">.25":>7}{"coherent":>11}{"pair p90":>10}{"outlier":>9}')
    for threshold in range(5, 13):
        stats = summarize(
            multi_collection,
            lambda item, threshold=threshold: item[f'sequential_{threshold}'],
        )
        print(f'{threshold:>4}{stats["published"]:>6}/{stats["total"]:<4}'
              f'{fmt(stats["opposite_p90"], 9)}'
              f'{fmt(stats["opposite_max"], 9)}'
              f'{stats["opposite_over_025"]:>7}'
              f'{stats["coherent"]:>5}/{stats["complete"]:<5}'
              f'{fmt(stats["axis_theory_p90"], 10)}'
              f'{len(stats["implausible"]):>9}')

    # Compare the same stamps without assuming either method is correct.
    joined = []
    for one_item, multi_item in paired_collection:
        joined.append({
            'source': one_item['source'], 'stamp': one_item['stamp'],
            'one': one_item, 'multi': multi_item,
        })
    agreement = compare(joined,
                        lambda item: one_pass_sides(item['one']),
                        lambda item: multipass_sides(item['multi']))
    print('\none-pass versus multipass on the six-scan collection')
    print(f'  both published: {agreement["both"]}; '
          f'median delta {agreement["median"]:.3f}; '
          f'p90 {agreement["p90"]:.3f}; max {agreement["max"]:.3f}; '
          f'delta > .25: {agreement["over_025"]}')
    print(f'  one-pass only: {len(agreement["left_only"])}; '
          f'multipass only: {len(agreement["right_only"])}')
    if agreement['over_025']:
        print('  largest disagreements (not labelled as errors):')
        for delta, key, one_gauge, multi_gauge in agreement['largest']:
            if delta <= .25:
                break
            print(f'    {key[0]}/{key[1]} {key[2]:<6} '
                  f'one={one_gauge:.3f} multi={multi_gauge:.3f} '
                  f'delta={delta:.3f}')

    for label, items, getter in (
            ('collection one-pass', one_collection, one_pass_sides),
            ('collection multipass', multi_collection, multipass_sides),
            ('worst one-pass', one_worst, one_pass_sides),
            ('worst multipass', multi_worst, multipass_sides)):
        stats = summarize(items, getter)
        mixed = [record for record in stats['axis_records']
                 if not record['coherent']]
        print(f'\n{label}: nearest-pair counts '
              f'{stats["class_counts"]}; mixed-axis={len(mixed)}')
        for record in sorted(mixed, key=lambda row: row['distance'],
                             reverse=True)[:10]:
            print(f'  {record["source"]}/{record["stamp"]} '
                  f'H={record["horizontal"]:.3f} '
                  f'V={record["vertical"]:.3f} '
                  f'H->{record["horizontal_class"]} '
                  f'V->{record["vertical_class"]} '
                  f'nearest-distance={record["distance"]:.3f}')
        if stats['implausible']:
            print('  side residual > .50 from both axis-specific theory values:')
            for source, stamp, side, gauge, residual, method in stats['implausible']:
                print(f'    {source}/{stamp} {side:<6} g={gauge:.3f} '
                      f'residual={residual:.3f} ({method})')


if __name__ == '__main__':
    main()
