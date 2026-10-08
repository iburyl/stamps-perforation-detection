"""Summarize Adaptive Lab profile variants without manual measurements."""
import json
import pathlib

import adaptive_profile_experiment as A
import fallback_channel_report as R

HERE = pathlib.Path(__file__).resolve().parent


def main():
    rows = json.loads((HERE / 'adaptive_profile_results.json').read_text())
    print(f'{len(rows)} sides rejected or replaced by primary cross-support')
    print('Reliable requires >=5 post-corner circle arcs. Safe additionally '
          'requires opposite delta <= .25 and theory residual <= .50.\n')
    print(f'{"method":<20}{"pub":>5}{"safe":>6}{">.25":>6}'
          f'{"median":>9}{"p90":>8}{"max":>8}{"fits":>7}')
    for method in A.METHODS:
        stats = R.summary(rows, method)
        print(f'{method:<20}{stats["published"]:>5}{stats["safe"]:>6}'
              f'{stats["bad"]:>6}{R.fmt(stats["median"]):>9}'
              f'{R.fmt(stats["p90"]):>8}{R.fmt(stats["max"]):>8}'
              f'{stats["fits"]:>7}')

    print('\ncase detail')
    for row in rows:
        print(f'\n{row["scan"]}/{row["stamp"]} {row["side"]} '
              f'current={row["current_method"]} '
              f'opposite={R.fmt(row["opposite_gauge"])} '
              f'stable={row["stable_candidates"]}')
        for method in A.METHODS:
            value = R.enriched(row, method)
            marker = 'OK' if value['safe'] else '--'
            print(f'  {method:<18} {marker} g={R.fmt(value.get("gauge"))} '
                  f'd={R.fmt(value["delta"])} arcs={value.get("arcs", 0):>2} '
                  f'{value.get("pick")}+{value.get("ridge")} '
                  f'ch={value.get("channel")} pol={value.get("polarity")} '
                  f't={value.get("color_threshold")}')

    print('\nnew-profile winner distribution')
    for method in ('adaptive_close', 'adaptive_stack', 'adaptive_best'):
        variants = {}
        channels = {}
        for row in rows:
            value = R.enriched(row, method)
            if not value['safe']:
                continue
            variant = value.get('ridge')
            channel = value.get('channel')
            variants[variant] = variants.get(variant, 0) + 1
            channels[channel] = channels.get(channel, 0) + 1
        print(f'  {method:<20} variants={variants} channels={channels}')


if __name__ == '__main__':
    main()
