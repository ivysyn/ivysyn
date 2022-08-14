import logging
import os
import random
import signal
import subprocess
import time
from glob import glob
from multiprocessing import Manager, Pool

RNG_SEED = 777

PYTORCH_PATH = "/home/ivyusr/ivysyn/src/frameworks/pytorch-1.11-ivysyn/"
RESULTS_PATH = "/mnt/pytorch-ivysyn/"
PYTHON_TEST_FOLDER = os.path.join(PYTORCH_PATH, "test/")
TEST_DURATION_FILE = os.path.join(RESULTS_PATH, "test_durations.txt")

NUM_PARALLEL_PROCESSES = 1
MAX_TIMEOUT_NS = 14400 * 1e+9

TEST_ARGS = [
    "--subprocess=false",
]

EXCLUDE_TESTS = [
]

tests_dur_file = open(TEST_DURATION_FILE, "w")

tests_to_run = glob(PYTHON_TEST_FOLDER + "test_*.py", recursive=True)
print(f"Python tests: {len(tests_to_run) - len(EXCLUDE_TESTS)}")
for t in EXCLUDE_TESTS:
    tests_to_run.remove(t)

random.seed(RNG_SEED)
random.shuffle(tests_to_run)
TOTAL_TESTS = len(tests_to_run)

done_tests = []
crashes_set = set()
active_tests = set()


def execute(test, run_start_time):

    args = ["python3", test]

    retcode = -1

    p = subprocess.Popen(
        args,
        # stdout=subprocess.DEVNULL,
        # stderr=subprocess.DEVNULL,
    )

    pid = p.pid

    fuzzed_kernel = None
    proc_kernel = None
    start_time = None

    test_start = time.clock_gettime(time.CLOCK_MONOTONIC)
    while True:

        kernel_mutfile = glob(RESULTS_PATH + f"*_mutations.log.{pid}")

        if len(kernel_mutfile) > 0:
            kernel_mutfile = kernel_mutfile[0]
        else:
            # No kernel being fuzzed right now
            if start_time is None:
                start_time = time.clock_gettime(time.CLOCK_MONOTONIC)

            cur_time = time.clock_gettime(time.CLOCK_MONOTONIC)

            elapsed = cur_time - start_time
            if elapsed > MAX_TIMEOUT_NS:
                print(
                    f"Process timed out (Run for {elapsed // 1e+9}s), killing")
                p.kill()
                p.wait()
                return (test, cur_time - test_start, -signal.SIGKILL, )

            retcode = p.poll()
            if retcode is not None:
                print(
                    f"Process {pid} ({test}) exited normally")
                p.wait()
                return (test, cur_time - test_start, p.poll(), )

            continue

        fuzzed_kernel_filename = kernel_mutfile[: kernel_mutfile.index(
            "_mutations")]
        fuzzed_kernel = fuzzed_kernel_filename.split('/')[-1]

        if proc_kernel is None:
            proc_kernel = fuzzed_kernel
            start_time = time.clock_gettime(time.CLOCK_MONOTONIC)
        elif proc_kernel != fuzzed_kernel:
            # We started fuzzing a new kernel
            proc_kernel = fuzzed_kernel
            # Save the starting time of the new kernel
            start_time = time.clock_gettime(time.CLOCK_MONOTONIC)
        else:
            retcode = p.poll()
            if retcode is not None:
                if retcode < 0:
                    print(
                        f"Process {pid} ({test.split('/')[-1]}:{fuzzed_kernel}) exited with code {retcode}")
                    if retcode == -signal.SIGKILL:
                        kill_time = int(
                            time.clock_gettime(time.CLOCK_MONOTONIC))
                        killed_filename = fuzzed_kernel_filename + ".killed"
                        with open(killed_filename, "w") as kf:
                            kf.write(str(kill_time))
                else:
                    # Finished without crashing
                    print(f"Process {pid} ({fuzzed_kernel}) exited normally")
                p.wait()
                return (test, cur_time - test_start, p.poll(), )

        time.sleep(1)


def log_test_duration(dur, test):
    secs = int(dur)
    tests_dur_file.write(test + " " + str(secs) + "\n")
    tests_dur_file.flush()


def proc_finished(results):

    test, running_time, exitcode = results

    log_test_duration(running_time, test)

    # If the test crashed, re-queue it so that it runs again
    if exitcode < 0 or exitcode > 128:

        logging.debug(
            f"Test {test} crashed with exit code {exitcode}, requeueing")
        tests_to_run.append(test)
        active_tests.remove(test)

    else:
        done_tests.append(test)


if __name__ == "__main__":

    os.chdir(PYTORCH_PATH)
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.DEBUG)
    manager = Manager()
    pool = Pool(processes=NUM_PARALLEL_PROCESSES)
    last_finished = 0

    run_start_time = time.clock_gettime(time.CLOCK_MONOTONIC)
    with open(RESULTS_PATH + "start_time.txt", "w") as f:
        f.write(str(run_start_time))

    while len(done_tests) < TOTAL_TESTS:

        being_run = active_tests - set(done_tests)
        while len(being_run) < NUM_PARALLEL_PROCESSES and len(tests_to_run) > 0:

            test_to_run = tests_to_run.pop()
            print(f"Running {test_to_run}")
            p = pool.apply_async(execute, args=(
                test_to_run, run_start_time,),
                callback=proc_finished, error_callback=proc_finished)
            active_tests.add(test_to_run)
            being_run = active_tests - set(done_tests)

        time.sleep(1)

        crashed_filenames = glob(RESULTS_PATH + "*_crashes.log")
        crashed_kernels = [x.split('/')[-1].replace('_crashes.log', '')
                           for x in crashed_filenames]
        new_crashes = set(crashed_kernels) - crashes_set
        cur_time = time.clock_gettime(time.CLOCK_MONOTONIC)
        elapsed = (cur_time - run_start_time) / 1e+9
        for i, crash in enumerate(new_crashes):
            crash_filename = crashed_filenames[i]
            if os.path.getsize(crash_filename) == 0:
                # os.remove(crash_filename)
                continue
            crashes_set.add(crash)
            print(f"New crash: {crash} ({elapsed}s into run)")

        finished = len(done_tests)
        if finished != last_finished:
            logging.info(
                f"Finished tests so far: {finished}")
            last_finished = finished

    pool.close()
    pool.join()

    run_end_time = time.clock_gettime(time.CLOCK_MONOTONIC)
    run_elapsed_time = (run_end_time - run_start_time)
    run_elased_time_mins = run_elapsed_time / 60
    with open(RESULTS_PATH + "end_time.txt", "w") as f:
        f.write(str(run_end_time))
    tests_dur_file.close()

    logging.info(
        f"Total tests run: {len(done_tests)} in {run_elased_time_mins} mins")
