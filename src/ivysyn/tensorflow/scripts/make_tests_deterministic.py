
import os
from glob import glob

TF_BASE = "/home/ivyusr/ivysyn/src/frameworks/tensorflow-2.6-ivysyn/"
PYTHON_TEST_FOLDER = os.path.join(TF_BASE, "tensorflow/python/")
RNG_SEED = '42'


def make_seeds_fixed(test_filename):

    print(test_filename)

    with open(test_filename, "r") as f:
        lines = f.read().split('\n')

    has_numpy_as_np = any(['import numpy as np' in x for x in lines])

    last_future_import = next((i for i in range(
        len(lines)-1, -1, -1) if 'from __future__' in lines[i]), 0)

    lines.insert(
        last_future_import + 1, f"from tensorflow.random import set_seed\nset_seed({RNG_SEED})")

    if has_numpy_as_np:
        for i, line in enumerate(lines):
            if 'import numpy as np' in line:
                spaces = len(line) - len(line.lstrip(' '))
                fixed_seed = f"np.random.seed({RNG_SEED})"
                lines.insert(
                    i + 1, fixed_seed.rjust(len(fixed_seed) + spaces, ' '))
                break

    with open(test_filename, "w") as f:
        f.write('\n'.join(lines))


def main():

    py_tests = glob(PYTHON_TEST_FOLDER + "**/*_test*.py", recursive=True)
    py_tests.remove(PYTHON_TEST_FOLDER +
                    "distribute/multi_worker_test_base.py")
    for test in py_tests:
        make_seeds_fixed(test)


if __name__ == "__main__":
    main()
