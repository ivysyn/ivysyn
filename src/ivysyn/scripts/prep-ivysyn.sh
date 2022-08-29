#!/bin/bash

IVYSYN_BASE="/home/ivyusr/ivysyn/"
IVYSYN_SCRIPTS_BASE="${IVYSYN_BASE}src/ivysyn/scripts/"
FRAMEWORKS_BASE="${IVYSYN_BASE}src/frameworks/"
VENV_BASE="${IVYSYN_BASE}venv/"
PYTORCH_BASE="${FRAMEWORKS_BASE}pytorch-1.11-ivysyn/"
TENSORFLOW_BASE="${FRAMEWORKS_BASE}tensorflow-2.6-ivysyn/"
CONDA_ACTIVATE_PATH="${VENV_BASE}anaconda3/bin/activate"

mount_result_dirs()
{
    pushd "${IVYSYN_SCRIPTS_BASE}"
    bash mount-result-dirs.sh
    popd
}

install_anaconda()
{
    pushd "${VENV_BASE}"
    bash Anaconda3-2022.05-Linux-x86_64.sh -b -p "${VENV_BASE}anaconda3"
    popd
}

install_pytorch()
{
    # Clone PyTorch repo
    git clone --depth 1 -b v1.11.0 --recursive https://github.com/pytorch/pytorch.git ${PYTORCH_BASE}
    pushd ${PYTORCH_BASE}
    git submodule sync
    git submodule update --init --recursive
    popd

    source "${CONDA_ACTIVATE_PATH}"

    # Install pre-built PyTorch version 1.11 with anaconda
    conda create --yes --name pytorch-1.11-orig
    conda activate pytorch-1.11-orig
    conda install --yes pytorch=1.11 torchvision torchaudio cpuonly -c pytorch
    conda deactivate

    # Install pre-built latest PyTorch version with anaconda
    conda create --yes --name pytorch-1.12-orig
    conda activate pytorch-1.12-orig
    conda install --yes pytorch torchvision torchaudio cpuonly -c pytorch
    conda deactivate

    # Create IvySyn virtual environment for PyTorch and install dependencies
    conda create --name pytorch-1.11-ivysyn --yes --file /home/ivyusr/ivysyn/pytorch-conda-requirements.txt
    conda activate pytorch-1.11-ivysyn
    pip3 install -r ${IVYSYN_BASE}pytorch-pip-requirements.txt

    # Install PyTorch
    export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
    pushd ${PYTORCH_BASE}
    python3 setup.py install
    popd
    pushd ${IVYSYN_SCRIPTS_BASE}
    bash prep-pytorch-ivysyn.sh
    popd
    pushd ${PYTORCH_BASE}
    python3 setup.py clean
    python3 setup.py develop
    python3 setup.py install
    popd

    conda deactivate
# > Make sure it is properly installed (before deactivating): `python3 -c "import torch; print(torch.__version__)"` should print `1.11.0`.

}

install_tensorflow()
{
    # Download bazel
    wget https://github.com/bazelbuild/bazelisk/releases/download/v1.11.0/bazelisk-linux-amd64
    [[ -d ${HOME}/.local/bin ]] || mkdir -p ${HOME}/.local/bin
    mv bazelisk-linux-amd64 ${HOME}/.local/bin/bazel
    chmod +x ${HOME}/.local/bin/bazel
    export PATH=${PATH}:${HOME}/.local/bin

    # Clone TensorFlow
    git clone --depth 1 -b v2.6.0 --recursive https://github.com/tensorflow/tensorflow.git ${TENSORFLOW_BASE}

    # Create virtual environments and install pre-build TensorFlow
    pushd ${VENV_BASE}
    python3 -m venv tensorflow-2.6-ivysyn
    python3 -m venv tensorflow-2.6-orig
    python3 -m venv tensorflow-2.9-orig

    source "${VENV_BASE}tensorflow-2.6-orig/bin/activate"
    pip3 install tensorflow==2.6.0
    deactivate

    source "${VENV_BASE}tensorflow-2.9-orig/bin/activate"
    pip3 install tensorflow==2.9
    deactivate

    source "${VENV_BASE}tensorflow-2.9-orig/bin/activate"
    pip3 install -U pip numpy wheel packaging
    pip3 install -U keras_preprocessing --no-deps

    # Prepare TensorFlow files for installation
    pushd ${IVYSYN_SCRIPTS_BASE}
    bash prep-tensorflow-ivysyn.sh

    # Install TensorFlow
    pushd ${TENSORFLOW_BASE}
    source "${VENV_BASE}tensorflow-2.6-orig/bin/activate"
    pip3 install -r ${IVYSYN_BASE}tensorflow-requirements.txt
    PYTHON_BIN_PATH="${VENV_BASE}tensorflow-2.6-orig/bin/python3" USE_DEFAULT_PYTHON_LIB_PATH=1 \
        TF_NEED_CUDA=0 TF_NEED_ROCM=0 TF_DOWNLOAD_CLANG=0 CC_OPT_FLAGS=0 TF_SET_ANDROID_WORKSPACE=0 \
        ./configure
    bazel build //tensorflow/tools/pip_package:build_pip_package --runs_per_test=10 --cache_test_results=no
    bazel build //tensorflow/core/kernels:all --runs_per_test=10 --cache_test_results=no
    ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp
    pip3 install --force-reinstall /tmp/tensorflow-2.6.0-cp39-cp39-linux_x86_64.whl
    pip3 install -r ${IVYSYN_BASE}tensorflow-requirements.txt
    popd

}

main()
{
    # Mount fuzzing result directories
    mount_result_dirs

    # Install anaconda3 for PyTorch virtual environments
    install_anaconda

    # Download, patch and install the frameworks
    install_pytorch
    install_tensorflow
}

main
