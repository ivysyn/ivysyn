#/bin/bash
set -x

CONDA_ACTIVATE_PATH="/home/ivyusr/ivysyn/venv/anaconda3/bin/activate"
PYTORCH_PATH="/home/ivyusr/ivysyn/src/frameworks/pytorch-1.11-ivysyn/"
PT_FILES_PATH="/home/ivyusr/ivysyn/src/ivysyn/pytorch/"
PATCHES_PATH="${PT_FILES_PATH}patches/"
SCRIPTS_PATH="${PT_FILES_PATH}scripts/"
RESULTS_PATH="/home/ivyusr/ivysyn/results/pytorch/"

apply_patches()
{
    echo "Patching PyTorch files..."

    patch "${PYTORCH_PATH}torch/include/ATen/ATen.h" "${PATCHES_PATH}ATen.h.patch"
    patch "${PYTORCH_PATH}aten/src/ATen/ATen.h" "${PATCHES_PATH}ATen.h.patch"
    # patch "${PYTORCH_PATH}torch/testing/_internal/common_utils.py" "${PATCHES_PATH}common_utils.py.patch"
    patch "${PYTORCH_PATH}tools/autograd/gen_python_functions.py" "${PATCHES_PATH}gen_python_functions.py.patch"

    echo "Patching done"
}

copy_files()
{
    echo "Copying IvySyn files..."

    cp ${PT_FILES_PATH}fuzzing* "${PYTORCH_PATH}aten/src/ATen/core"
    cp ${PT_FILES_PATH}native_functions_no_dups.yaml "${PYTORCH_PATH}aten/src/ATen/native"

    echo "Files copied"
}

run_pass()
{
    echo "Running code injecting pass..."

    source "${CONDA_ACTIVATE_PATH}"
    conda activate pytorch-1.11-ivysyn

    pushd ${SCRIPTS_PATH}
    python3 get_fnames.py
    python3 inject_fuzzing_code.py > ${RESULTS_PATH}instrumentation/instrumentation_output.txt
    popd

    conda deactivate
    echo "Pass run"
}

main()
{
    apply_patches
    copy_files
    run_pass
}

main
