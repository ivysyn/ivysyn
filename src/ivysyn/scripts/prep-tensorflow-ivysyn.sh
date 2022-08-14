#/bin/bash
set -x

IVYSYN_VENV="/home/ivyusr/ivysyn/venv/tensorflow-2.6-ivysyn/bin/activate"
TENSORFLOW_PATH="/home/ivyusr/ivysyn/src/frameworks/tensorflow-2.6-ivysyn/"
TF_FILES_PATH="/home/ivyusr/ivysyn/src/ivysyn/tensorflow/"
PATCHES_PATH="${TF_FILES_PATH}patches/"
SCRIPTS_PATH="${TF_FILES_PATH}scripts/"
PASS_PATH="${TF_FILES_PATH}inject-fuzzer/"
RESULTS_PATH="/home/ivyusr/ivysyn/results/tensorflow/"

apply_patches()
{
    echo "Patching TensorFlow files..."

    patch "${TENSORFLOW_PATH}tensorflow/c/exported_symbols.lds" "${PATCHES_PATH}c_exported_symbols.lds.patch"
    patch "${TENSORFLOW_PATH}tensorflow/BUILD" "${PATCHES_PATH}tensorflow_BUILD.patch"
    patch "${TENSORFLOW_PATH}tensorflow/core/framework/BUILD" "${PATCHES_PATH}tensorflow_core_framework_BUILD.patch"
    patch "${TENSORFLOW_PATH}tensorflow/python/BUILD" "${PATCHES_PATH}tensorflow_python_BUILD.patch"
    patch "${TENSORFLOW_PATH}tensorflow/tf_exported_symbols.lds" "${PATCHES_PATH}tf_exported_symbols.lds.patch"
    patch "${TENSORFLOW_PATH}tensorflow/tf_version_script.lds" "${PATCHES_PATH}tf_version_script.lds.patch"
    patch "${TENSORFLOW_PATH}tensorflow/core/framework/op_kernel.h" "${PATCHES_PATH}op_kernel.h.patch"
    # patch "${TENSORFLOW_PATH}tensorflow/core/framework/op_kernel.cc" "${PATCHES_PATH}op_kernel.cc.patch"

    echo "Patching done"
}

copy_files()
{
    echo "Copying ivysyn files..."
    cp ${TF_FILES_PATH}fuzzing* "${TENSORFLOW_PATH}tensorflow/core/framework"
    echo "Files copied"
}

run_pass()
{
    echo "Compiling and running code injecting pass..."

    pushd ${PASS_PATH}
    cmake .
    make
    popd

    pushd ${SCRIPTS_PATH}
    bash inject_fuzzing_code.sh > "${RESULTS_PATH}instrumentation/instrumentation_output.txt"
    popd

    echo "Pass run"
}

gen_helper_files()
{
    pushd ${SCRIPTS_PATH}
    python3 collect_registered_kernels.py
    popd
}

main()
{
    . ${IVYSYN_VENV}
    apply_patches
    copy_files
    run_pass
    gen_helper_files
    deactivate
}

main
