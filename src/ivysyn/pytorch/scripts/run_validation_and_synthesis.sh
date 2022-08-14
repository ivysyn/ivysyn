#!/bin/bash

# set -x
# set -euo pipefail

sources="/home/ivyusr/ivysyn/results/pytorch/atheris_comp/one_to_one_pt"
IVYSYN_PATH="/home/ivyusr/ivysyn/"
SCRIPTS_PATH="${IVYSYN_PATH}/src/ivysyn/pytorch/scripts/"
IVYSYN_SCRIPTS_PATH="${IVYSYN_PATH}/src/ivysyn/scripts/"
VALIDATION_DRIVERS="${sources}"
RESULTS_DIR="/mnt/pytorch-ivysyn/"
IVYSYN_ENV="pytorch-1.11-ivysyn"
VALIDATION_ENV="pytorch-1.11-ivysyn-validate"
ORIG_ENV="pytorch-1.11-orig"
CONDA_ACTIVATE_PATH="${IVYSYN_PATH}venv/anaconda3/bin/activate"
subdirs=("segfault" "fpe" "abort" "other")

do_usage()
{
    echo "Usage: ./`basename $0` --dir <synth_folder> [--clean]"
}

do_clean()
{
    asan_out=$1
    crashes_dir=$2
    crash_type_dir=$3
    rm -v $RESULTS_DIR/*validate || true
    rm -v $RESULTS_DIR/*positive || true
    rm -v $RESULTS_DIR/*check || true
    rm -v $RESULTS_DIR/*killed || true
    rm -v $asan_out/* || true
    rm -v $crashes_dir/*positive || true
    rm -v $crashes_type_dir/segfault/* || true
    rm -v $crashes_type_dir/abort/* || true
    rm -v $crashes_type_dir/fpe/* || true
    rm -v $crashes_type_dir/other/* || true
}

do_validate()
{

    asan_out=$1
    crash_type_out_all=$2
    crash_type_out_run=$3
    validation_path=$4
    conda activate "${VALIDATION_ENV}"
    for line in $(cat validation_list.txt); do
        op=$(echo $line | cut -d ':' -f 1)
        kernel=$(echo $line | cut -d ':' -f 2)
        echo $op
        echo $kernel
        cp ${validation_files_path}/${kernel}.validate ${RESULTS_DIR}
        LD_PRELOAD=$(clang --print-file-name libclang_rt.asan-x86_64.so) ASAN_OPTIONS=detect_leaks=0:symbolize=1:detect_odr_violation=0:allocator_may_return_null=1:log_path="${asan_out}/${kernel}" timeout -s 9 60 python3 ${VALIDATION_DRIVERS}/${op}_op.py
        exit_code=$?
        sig=$(($exit_code - 128))
        [[ $sig -eq -9 || $sig -eq 9 ]] && touch "${RESULTS_DIR}/${kernel}.false_positive" && rm "${RESULTS_DIR}/${kernel}.check"
        [[ $sig -eq -6 || $sig -eq 6 ]] && touch "${crash_type_out_all}abort/${kernel}" && touch "${crash_type_out_run}abort/${kernel}"
        LD_PRELOAD=$(clang --print-file-name libclang_rt.asan-x86_64.so) ASAN_OPTIONS=detect_leaks=0:symbolize=1:detect_odr_violation=0:allocator_may_return_null=1 timeout -s 9 60 python3 ${VALIDATION_DRIVERS}/${op}_op.py 2>&1 2> /dev/null
        rm ${RESULTS_DIR}/${kernel}.validate
    done
    conda deactivate
}

main()
{

    clean=0

    while [[ $# -gt 0 ]]; do
        key="$1"
        case $key in
        --dir)
            shift
            dir_to_check=$1
            shift
            ;;
        --clean)
            clean=1
            shift
            ;;
        *)
            do_usage
            exit 1
            ;;
        esac
    done

    source "${CONDA_ACTIVATE_PATH}"

    validation_files_path="${IVYSYN_PATH}/results/pytorch/synthesized/${dir_to_check}/validate/"
    crashes_path="${IVYSYN_PATH}/results/pytorch/crashes/${dir_to_check}/"
    crash_type_path_all="${IVYSYN_PATH}/results/pytorch/crash_types/all/"
    crash_type_path_run="${IVYSYN_PATH}/results/pytorch/crash_types/${dir_to_check}/"
    asan_out_path="${crashes_path}/asan_output/"
    mkdir -p "${asan_out_path}"

    for subd in ${subdirs[@]}; do
        [[ -d "${crash_type_path_all}${subd}" ]] || mkdir -p "${crash_type_path_all}${subd}"
        [[ -d "${crash_type_path_run}${subd}" ]] || mkdir -p "${crash_type_path_run}${subd}"
    done

    [[ $clean -eq 1 ]] && do_clean "${asan_out_path}" "${crashes_path}" "${crash_type_path_run}"

    pushd "${SCRIPTS_PATH}"

    echo "Producing validation files..."
    conda activate "${IVYSYN_ENV}"
    python3 synthesizer.py --dir "${dir_to_check}" --validate

    echo "Generating validation list..."
    python3 prep_validation.py --dir "${dir_to_check}" > validation_list.txt
    # Deactivate orig env here
    conda deactivate

    echo "Validating crashes with drivers..."
    do_validate "${asan_out_path}" "${crash_type_path_all}" "${crash_type_path_run}" "${validation_files_path}"

    echo "Moving results..."
    mv ${RESULTS_DIR}/*positive ${crashes_path}
    rm -v $RESULTS_DIR/*validate || true

    echo "Validating crashes without drivers..."
    conda activate "${VALIDATION_ENV}"
    python3 run_kernel_tests_validate.py --dir "${dir_to_check}"
    conda deactivate

    echo "Synthesizing PoVs..."
    conda activate "${IVYSYN_ENV}"
    python3 synthesizer.py --dir "${dir_to_check}"
    conda deactivate

    echo "Checking PoV reproducibility..."
    "${SCRIPTS_PATH}/reproducer.sh" --dir "${dir_to_check}"

    popd

    pushd  "${IVYSYN_SCRIPTS_PATH}"
    echo "Parsing ASan output and categorizing crashes..."
    python3 parse_asan_output.py --dir "${dir_to_check}" --pytorch
    popd

}

main $@

