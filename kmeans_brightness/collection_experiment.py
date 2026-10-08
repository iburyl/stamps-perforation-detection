"""Run the leading method on six complete scans; ruler gauges are not truth."""
import json
import os
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import arc_strategy_experiment as A  # noqa: E402
import experiment as E  # noqa: E402

ROOT = pathlib.Path(os.environ.get(
    'STAMPS_COLLECTION_DIR', r'D:\workspace\IX vs X\IX-X',
))
SOURCES = ('1K.png', '2K.png', '3K.png', '5K.png', '7K.png', '14K.png')
OUT = HERE / 'collection_results.json'
_IMAGE = None


def _init(path):
    global _IMAGE
    _IMAGE = cv2.imread(path)


def _run(task):
    index, orientation, threshold = task
    result = A.measure(
        _IMAGE, orientation, threshold,
        strategies=('current', 'pitch_lock'),
    )
    result['stamp'] = index + 1
    return result


def measure_source(path):
    image = cv2.imread(str(path))
    orientations, threshold = E.front_end(image)
    del image
    tasks = [(index, orientation, threshold)
             for index, orientation in enumerate(orientations)
             if orientation is not None]
    results = []
    with ProcessPoolExecutor(max_workers=4, initializer=_init,
                             initargs=(str(path),)) as pool:
        for done, result in enumerate(pool.map(_run, tasks), 1):
            result['source'] = path.stem
            results.append(result)
            print(f'{path.stem} stamp {result["stamp"]:>2} '
                  f'({done}/{len(tasks)})', flush=True)
    return results


def main():
    results = []
    for source in SOURCES:
        results.extend(measure_source(ROOT / source))
    OUT.write_text(json.dumps(results, indent=1))
    print(f'-> {OUT}')


if __name__ == '__main__':
    main()
