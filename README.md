
# General

### Package installs

    sudo apt install git python3-pip python3-dev llvm-dev clang-11 ripgrep fd-find python3-venv cmake clang-tools-11 libclang-11-dev openjdk-11-jdk java-common bc build-essential clang nasm

### Prepare IvySyn files

This script does the following:
- Mounts logging directories for the IvySyn fuzzing results
- Downloads PyTorch and TensorFlow and patches them to make them compatible to be run with IvySyn
- Creates Python virtual environments required by IvySyn and installs the required packages, including the two DL frameworks

        cd /home/ivyusr/ivysyn/src/ivysyn/scripts
        bash ./prep-ivysyn.sh

# PyTorch

## Running the fuzzer

Make sure you are in the correct virtual environment (`conda activate pytorch-1.11-ivysyn`).

    cd /home/ivyusr/ivysyn/src/ivysyn/pytorch/scripts
    python3 run_kernel_tests.py

Ivysyn will produce results under the temporary, tmpfs mounted directory `/mnt/pytorch-ivysyn`.

## Synthesizing and running PoVs

### Running the synthesizer

First, create a new directory under `/home/ivyusr/ivysyn/results/ivysyn/crashes/` and copy over the results produced by IvySyn, e.g.,

    cd /home/ivyusr/ivysyn/results/pytorch/crashes/
    mkdir testrun
    cp /mnt/pytorch-ivysyn/* testrun

Then, run the synthesizer, passing as the `--dir` argument the name of the created subdirectory, e.g.,

    cd /home/ivyusr/ivysyn/src/ivysyn/pytorch/scripts
    python3 synthesizer.py --dir testrun

The PoVs will be produced under `/home/ivyusr/ivysyn/results/pytorch/synthesized/<dirname>/all`.

## Installation steps for ASan version

First, restore the modified kernels by running

    cd /home/ivyusr/ivysyn/src/frameworks/pytorch-1.11-ivysyn/aten/src/ATen/native
    rm native_functions_no_dups.yaml
    git restore *

### Create a new conda virtual environment

    conda create --name pytorch-1.11-asan
    conda activate pytorch-1.11-asan

### Build with ASan configuration

If you have installed PyTorch before, make sure to clean the old installation first by running `python3 setup.py clean` in the PyTorch root directory.

    cd /home/ivyusr/ivysyn/src/frameworks/pytorch-1.11-ivysyn
    export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
    ASAN_OPTIONS=detect_leaks=0:symbolize=1:detect_odr_violation=0 CC="clang" CXX="clang++" LDSHARED="clang --shared" CFLAGS="-fsanitize=address -Wno-unused-command-line-argument -shared-libasan -pthread -Wl,-rpath=$(dirname $($CXX --print-file-name libclang_rt.asan-x86_64.so)) -Wno-unused-command-line-argument -fno-omit-frame-pointer" CXX_FLAGS="-fsanitize=address -shared-libasan -pthread -Wl,-rpath=$(dirname $($CXX --print-file-name libclang_rt.asan-x86_64.so)) -fno-omit-frame-pointer" USE_ASAN=1 DEBUG=0 python3 setup.py install

In case of a `permission denied` error, add an extra `python3 setup.py develop` commad as follows:

    cd /home/ivyusr/ivysyn/src/frameworks/pytorch-1.11-ivysyn
    export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
    ASAN_OPTIONS=detect_leaks=0:symbolize=1:detect_odr_violation=0 CC="clang" CXX="clang++" LDSHARED="clang --shared" CFLAGS="-fsanitize=address -Wno-unused-command-line-argument -shared-libasan -pthread -Wl,-rpath=$(dirname $($CXX --print-file-name libclang_rt.asan-x86_64.so)) -Wno-unused-command-line-argument -fno-omit-frame-pointer" CXX_FLAGS="-fsanitize=address -shared-libasan -pthread -Wl,-rpath=$(dirname $($CXX --print-file-name libclang_rt.asan-x86_64.so)) -fno-omit-frame-pointer" USE_ASAN=1 DEBUG=0 python3 setup.py develop
    ASAN_OPTIONS=detect_leaks=0:symbolize=1:detect_odr_violation=0 CC="clang" CXX="clang++" LDSHARED="clang --shared" CFLAGS="-fsanitize=address -Wno-unused-command-line-argument -shared-libasan -pthread -Wl,-rpath=$(dirname $($CXX --print-file-name libclang_rt.asan-x86_64.so)) -Wno-unused-command-line-argument -fno-omit-frame-pointer" CXX_FLAGS="-fsanitize=address -shared-libasan -pthread -Wl,-rpath=$(dirname $($CXX --print-file-name libclang_rt.asan-x86_64.so)) -fno-omit-frame-pointer" USE_ASAN=1 DEBUG=0 python3 setup.py install

### Running PoVs with ASan

To run a PoV with the ASan-compiled PyTorch installation, first activate the conda virtual environment containing the ASan installation and then run the PoV with some environment variables set as follows:

    conda activate pytorch-1.11-asan
    LD_PRELOAD=$(clang --print-file-name libclang_rt.asan-x86_64.so) ASAN_OPTIONS=detect_leaks=0:symbolize=1:detect_odr_violation=0 python3 pov.py

# TensorFlow

### Sanity checks

- Make sure TensorFlow is installed properly by running `python3 -c "import tensorflow as tf; print(tf.__version__)` **outside the `/home/ivyusr/ivysyn/src/frameworks/tensorflow-2.6-ivysyn` directory**. The output should be `2.6.0`.

## Preparing and running the fuzzer

### Setting the RNG seed

IvySyn uses a random seed in 3 different places in TensorFlow:
First, set a seed of your choosing as the `RNG_SEED` define in `/home/ivyusr/ivysyn/src/frameworks/tensorflow-2.6-ivysyn/tensorflow/core/framework/fuzzing.h`. This seeds controls the random generator associated with the fuzzer.

Second, change the `RNG_SEED` variable in `/home/ivyusr/ivysyn/src/ivysyn/tensorflow/scripts/run_kernel_tests.py`. This sets the seed associated with shuffling with the developer tests.

Finally change `RNG_SEED` in `/home/ivyusr/ivysyn/src/ivysyn/tensorflow/scripts/make_tests_deterministic.py` and run the script. This sets the random seed in all developer tests we will run.

### Running the fuzzer

Make sure you are in the correct virtual environment (`source /home/ivyusr/ivysyn/venv/tensorflow-2.6-ivysyn/bin/activate`).

    cd /home/ivyusr/ivysyn/src/ivysyn/tensorflow/scripts
    python3 run_kernel_tests.py

Ivysyn will produce results under the temporary, tmpfs mounted directory `/mnt/tensorflow-ivysyn`.


## Synthesizing and running PoVs

### Running the synthesizer

First, create a new directory under `/home/ivyusr/ivysyn/results/tensorflow/crashes/` and copy over the results produced by IvySyn, e.g.,

    cd /home/ivyusr/ivysyn/results/tensorflow/crashes/
    mkdir testrun
    cp /mnt/tensorflow-ivysyn/* testrun

Then, run the synthesizer, passing as the `--dir` argument the name of the created subdirectory, e.g.,

    cd /home/ivyusr/ivysyn/src/ivysyn/tensorflow/scripts
    python3 synthesizer.py --dir testrun

The PoVs will be produced under `/home/ivyusr/ivysyn/results/tensorflow/synthesized/<dirname>/all`.

### Running the PoVs
A script is provided which runs the synthesized PoVs and categorizes them based on the signal they exit with when they crash.
The script will also run the PoVs using the latest TensorFlow release.
The name of the directory created in the previous step needs to be provided as an argument to `--dir`.

    /home/ivyusr/ivysyn/src/ivysyn/tensorflow/scripts
    ./reproducer.sh


## Installation steps for ASan version

First, complete steps `Download bazel` and `Clone TensorFlow` (or, re-clone TensorFlow in a new directory and checkout to the correct version) from `Installation steps for IvySyn version`.

If you have run `prep-tensorflow-ivysyn.sh`, restore the modified kernels by running

    cd /home/ivyusr/ivysyn/src/frameworks/tensorflow-2.6-ivysyn/tensorflow/core/kernels
    git restore *

If you have run `make_tests_deterministic.py`, restore the modified scripts by running
    cd /home/ivyusr/ivysyn/src/frameworks/tensorflow-2.6-ivysyn/python
    git restore *

If you have previously installed TensorFlow from source, clean the build by running

    cd /home/ivyusr/ivysyn/src/frameworks/tensorflow-2.6-ivysyn
    bazel clean --expunge

### Create new virtual environment for ASan installation and install dependencies

    cd /home/ivyusr/ivysyn/venv
    python3 -m venv tensorflow-2.6-asan
    source /home/ivyusr/ivysyn/venv/tensorflow-2.6-asan/bin/activate
    pip3 install -U pip numpy wheel packaging
    pip3 install -U keras_preprocessing --no-deps
    pip3 install -r /home/ivyusr/ivysyn/tensorflow-requirements.txt

### Apply the following patches

    patch /home/ivyusr/ivysyn/src/frameworks/tensorflow-2.6-ivysyn/third_party/jpeg/BUILD.bazel /home/ivyusr/ivysyn/src/ivysyn/tensorflow/patches/jpeg_BUILD.bazel.patch
    patch /home/ivyusr/ivysyn/src/frameworks/tensorflow-2.6-ivysyn/.bazelrc /home/ivyusr/ivysyn/src/ivysyn/tensorflow/patches/bazelrc.patch

### Build and install TensorFlow with ASan

    cd /home/ivyusr/ivysyn/src/frameworks/tensorflow-2.6-ivysyn
    source /home/ivyusr/ivysyn/venv/tensorflow-2.6-asan/bin/activate
    pip3 install -r tensorflow-requirements.txt
    ./configure # Choose all default options
    pip3 install -r
    CC=$(which clang) bazel build --config asan --action_env=LD_PRELOAD=$(clang --print-file-name libclang_rt.asan-x86_64.so) --action_env=ASAN_OPTIONS=detect_leaks=0:detect_odr_violation=0 //tensorflow/tools/pip_package:build_pip_package
    ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp
    pip3 install --force-reinstall --no-deps /tmp/tensorflow-2.6.0-cp39-cp39-linux_x86_64.whl

If you wish to install in a separate directory, you can specify the installation path using the option `--output_user_root=/path/to/install` before the `build` keyword, as follows
    CC=$(which clang) bazel --output_user_root=/path/to/install build --config asan --action_env=LD_PRELOAD=$(clang --print-file-name libclang_rt.asan-x86_64.so) --action_env=ASAN_OPTIONS=detect_leaks=0:detect_odr_violation=0 //tensorflow/tools/pip_package:build_pip_package

If your system is limited on resources, you can append `-j N` at the end of the build command to reduce the number of parallel jobs while building (where `N` is the number of parallel jobs).

> Note: IvySyn runs some developer tests (when running `run_kernel_tests.py`) which are produced when the `bazel build //tensorflow/core/kernels:all --runs_per_test=10 --cache_test_results=no` command is run. If you have built TensorFlow with ASan, and wish to run the fuzzer, you will need to build TensorFlow again following the steps in `Install TensorFlow` in order to produce these tests (make sure you have installed the ASan version first by completing all of the steps above).

### Running PoVs with ASan

To run a PoV with the ASan-compiled TensorFlow installation, first activate the virtual environment containing the ASan installation and then run the PoV with some environment variables set as follows:

    source /home/ivyusr/ivysyn/venv/tensorflow-2.6-asan/bin/activate
    LD_PRELOAD=$(clang --print-file-name libclang_rt.asan-x86_64.so) ASAN_OPTIONS=detect_leaks=0:symbolize=1:detect_odr_violation=0 python3 pov.py

