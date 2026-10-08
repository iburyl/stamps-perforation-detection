"""Run ranked all-partition k-means search on the six-scan collection."""
import json
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import candidate_search_experiment as C  # noqa: E402
import collection_experiment as B  # noqa: E402
import experiment as E  # noqa: E402

OUT = HERE / 'candidate_search_collection_results.json'
_IMAGE = None


def _init(path):
    global _IMAGE
    _IMAGE = cv2.imread(path)


def _run(task):
    index, orientation, threshold = task
    result = C.measure(_IMAGE, orientation, threshold)
    result['stamp'] = index + 1
    return result


def measure_source(path):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
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
    for source in B.SOURCES:
        results.extend(measure_source(B.ROOT / source))
    OUT.write_text(json.dumps(results, indent=1))
    print(f'-> {OUT}')


if __name__ == '__main__':
    main()
