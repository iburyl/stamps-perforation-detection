"""Run the one-arc selector dataset over the six-scan collection."""
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import collection_experiment as C  # noqa: E402
import one_pass_experiment as O  # noqa: E402

OUT = HERE / 'one_pass_collection_results.json'


def main():
    results = []
    for source in C.SOURCES:
        path = C.ROOT / source
        measured = O.run_scan(path, workers=4)
        for item in measured:
            item['source'] = path.stem
        results.extend(measured)
    OUT.write_text(json.dumps(results, indent=1))
    print(f'-> {OUT}')


if __name__ == '__main__':
    main()
