import argparse
import logging
import os
import shutil
import signal
import subprocess
import time
from glob import glob

TENSORFLOW_PATH = "/home/ivyusr/ivysyn/src/frameworks/tensorflow-2.6-ivysyn/"
RESULTS_PATH = "/mnt/tensorflow-ivysyn/"
PYTHON_TEST_FOLDER = os.path.join(TENSORFLOW_PATH, "tensorflow/python/")
CC_TEST_FOLDER = os.path.join(
    TENSORFLOW_PATH, "bazel-out/k8-opt/bin/tensorflow/core/kernels/")
ABORT_CRASH_PATH_BASE = "/home/ivyusr/ivysyn/results/tensorflow/crash_types/"
KERNELS_TO_TESTS_FILE = "/home/ivyusr/ivysyn/src/ivysyn/tensorflow/kernels_to_tests.txt"

IVYSYN_PATH = "/home/ivyusr/ivysyn/"
CRASHES_DIR_BASE = os.path.join(IVYSYN_PATH, "results/tensorflow/crashes/")
VALIDATION_FILES_PATH_BASE = os.path.join(
    IVYSYN_PATH, "results/tensorflow/synthesized/")

ASAN_OUT_BASE = "/home/ivyusr/ivysyn/results/pytorch/crashes/"
asan_rt = "/usr/lib/llvm-11/lib/clang/11.0.1/lib/linux/libclang_rt.asan-x86_64.so"

NUM_PARALLEL_PROCESSES = 1
MAX_TIMEOUT_NS = 1500 * 1e+9

BAZEL_TEST_ARGS = [
    "--test_output=all",
    "--cache_test_results=no",
    "--runs_per_test=10", "--flaky_test_attempts=10",
    "--jobs=1"
]

EXCLUDE_TESTS = [
    # Opens connection, gets confused because of fuzzing
    os.path.join(PYTHON_TEST_FOLDER, "eager/remote_cluster_test.py"),
    os.path.join(PYTHON_TEST_FOLDER, "keras/saving/save_weights_test.py"),
    # os.path.join(PYTHON_TEST_FOLDER, "kernel_tests/while_v2_test.py"),
]

last_cur_positives = []


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()

    args_parser.add_argument(
        "--dir", dest="dir", required=True)

    args = args_parser.parse_args()
    asan_out = ASAN_OUT_BASE + args.dir + "/asan_output/"
    abort_crash_path = ABORT_CRASH_PATH_BASE + args.dir + "/abort/"
    crashes_dir = CRASHES_DIR_BASE + args.dir + "/"
    validation_dir = VALIDATION_FILES_PATH_BASE + args.dir + "/validate/"

    os.chdir(TENSORFLOW_PATH)
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.DEBUG)

    with open(KERNELS_TO_TESTS_FILE, "r") as f:
        lines = f.read().strip().split('\n')

    kernels_to_tests = {}
    for line in lines:
        kernel, test = line.split(' ')
        if kernel not in kernels_to_tests:
            kernels_to_tests[kernel] = test

    positives = glob(crashes_dir + "*positive")
    positives = [x.split('/')[-1] for x in positives]
    positives = [x[:x.index('.')] for x in positives]

    to_validate = glob(validation_dir + "*.validate")
    to_validate = [x.split('/')[-1].replace('.validate', '')
                   for x in to_validate]
    to_validate = [x for x in to_validate if x not in positives]

    print(f"Validating {len(to_validate)} kernels without drivers")
    # print(kernels_to_tests)

    for kernel in to_validate:
        test = kernels_to_tests[kernel]
        print(kernel, test)
        validation_file = validation_dir + kernel + ".validate"
        shutil.copy(validation_file, RESULTS_PATH)

        test_path = os.path.abspath(test)

        if PYTHON_TEST_FOLDER in test_path:
            args = ["python3", test]
        elif CC_TEST_FOLDER in test_path:
            continue

        print(f"Running {test} for kernel {kernel}")

        env = os.environ.copy()
        env['ASAN_OPTIONS'] = f"detect_leaks=0:symbolize=1:detect_odr_violation=0:allocator_may_return_null=1:log_path='{asan_out}{kernel}'"
        env['LD_PRELOAD'] = asan_rt
        p = subprocess.Popen(args, env=env,)
        p.wait()

        exitcode = p.poll()

        if exitcode == -signal.SIGABRT:
            with open(abort_crash_path + kernel, "w"):
                pass

        p = subprocess.Popen(args, env=env,)
        p.wait()

        positive_file = glob(RESULTS_PATH + "*positive")
        if len(positive_file) > 0:
            print(f"Result for {kernel}: {positive_file}")
            positive_file = positive_file[0]
            shutil.copy(positive_file, crashes_dir)

        for filename in glob(RESULTS_PATH + "*"):
            os.remove(filename)
