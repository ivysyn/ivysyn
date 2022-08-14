import logging
import os
import subprocess
import time
from glob import glob
from multiprocessing import Manager, Pool

PYTORCH_PATH = "/home/ivyusr/ivysyn/src/frameworks/pytorch-1.11-ivysyn/"
RESULTS_PATH = "/mnt/pytorch-ivysyn/"
PYTHON_TEST_FOLDER = os.path.join(PYTORCH_PATH, "test/")
KERNELS_TO_TESTS_FILE = "/home/ivyusr/ivysyn/src/ivysyn/pytorch/kernels_to_tests.txt"

NUM_PARALLEL_PROCESSES = 1
MAX_TIMEOUT_NS = 14400 * 1e+9

EXCLUDE_TESTS = [
]

last_logged = []

tests_to_run = glob(PYTHON_TEST_FOLDER + "test_*.py", recursive=True)
print(f"Python tests: {len(tests_to_run) - len(EXCLUDE_TESTS)}")
for t in EXCLUDE_TESTS:
    tests_to_run.remove(t)

TOTAL_TESTS = len(tests_to_run)

done_tests = []
crashes_set = set()
active_tests = set()


def execute(test):

    args = ["python3", test]
    p = subprocess.Popen(
        args,
        # stdout=subprocess.DEVNULL,
        # stderr=subprocess.DEVNULL,
    )
    p.wait()

    return test


def get_logged():

    logged = glob(RESULTS_PATH + "*types")
    logged = [x.split('/')[-1] for x in logged]
    logged = [x[:x.index('.')] for x in logged]

    return logged


def proc_finished(testname):

    global last_logged

    logged = get_logged()
    new_logged = [x for x in logged if x not in last_logged]

    with open(KERNELS_TO_TESTS_FILE, "a+") as f:
        for kernel in new_logged:
            f.write(kernel + " " + testname + "\n")

    last_logged = get_logged()
    done_tests.append(testname)


if __name__ == "__main__":

    os.chdir(PYTORCH_PATH)
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.DEBUG)
    manager = Manager()
    pool = Pool(processes=NUM_PARALLEL_PROCESSES)
    last_finished = 0

    while len(done_tests) < TOTAL_TESTS:

        being_run = active_tests - set(done_tests)
        while len(being_run) < NUM_PARALLEL_PROCESSES and len(tests_to_run) > 0:

            test_to_run = tests_to_run.pop()
            print(f"Running {test_to_run}")
            p = pool.apply_async(execute, args=(
                test_to_run, ),
                callback=proc_finished, error_callback=proc_finished)
            active_tests.add(test_to_run)
            being_run = active_tests - set(done_tests)

        time.sleep(1)

        finished = len(done_tests)
        if finished != last_finished:
            logging.info(
                f"Finished tests so far: {finished}")
            last_finished = finished

    pool.close()
    pool.join()

    logging.info(
        f"Total tests run: {len(done_tests)}")
