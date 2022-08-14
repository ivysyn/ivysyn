#!/bin/bash

# set -x
# set -euo pipefail

sources="/home/ivyusr/ivysyn/results/tensorflow/atheris_comp/one_to_one_tf"
IVYSYN_PATH="/home/ivyusr/ivysyn/"
SCRIPTS_PATH="${IVYSYN_PATH}/src/ivysyn/tensorflow/scripts/"
IVYSYN_SCRIPTS_PATH="${IVYSYN_PATH}/src/ivysyn/scripts/"
VALIDATION_DRIVERS="${IVYSYN_PATH}/src/ivysyn/tensorflow/validation_drivers/"
RESULTS_DIR="/mnt/tensorflow-ivysyn/"
VALIDATION_ENV="${IVYSYN_PATH}venv/tensorflow-2.6-ivysyn-validate/bin/activate"
ORIG_ENV="${IVYSYN_PATH}venv/tensorflow-2.6-orig/bin/activate"
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
    rm -v $crash_type_dir/segfault/* || true
    rm -v $crash_type_dir/abort/* || true
    rm -v $crash_type_dir/fpe/* || true
    rm -v $crash_type_dir/other/* || true
}

do_validate()
{

    asan_out=$1
    crash_type_out_all=$2
    crash_type_out_run=$3
    source "${VALIDATION_ENV}"
    for line in $(cat validation_list.txt); do
        op=$(echo $line | cut -d ':' -f 1)
        kernel=$(echo $line | cut -d ':' -f 2)
        echo $op
        echo $kernel
        cp ${validation_files_path}/${kernel}.validate ${RESULTS_DIR}
        TF_CPP_MIN_LOG_LEVEL=2 LD_PRELOAD=$(clang --print-file-name libclang_rt.asan-x86_64.so) ASAN_OPTIONS=detect_leaks=0:symbolize=1:detect_odr_violation=0:allocator_may_return_null=1:log_path="${asan_out}/${kernel}" timeout -s 9 60 python3 ${VALIDATION_DRIVERS}/$op.py
        exit_code=$?
        sig=$(($exit_code - 128))
        [[ $sig -eq -9 || $sig -eq 9 ]] && touch "${RESULTS_DIR}/${kernel}.false_positive" && rm "${RESULTS_DIR}/${kernel}.check"
        [[ $sig -eq -6 || $sig -eq 6 ]] && touch "${crash_type_out_all}abort/${kernel}" && touch "${crash_type_out_run}abort/${kernel}"
        TF_CPP_MIN_LOG_LEVEL=2 LD_PRELOAD=$(clang --print-file-name libclang_rt.asan-x86_64.so) ASAN_OPTIONS=detect_leaks=0:symbolize=1:detect_odr_violation=0:allocator_may_return_null=1 timeout -s 9 60 python3 ${VALIDATION_DRIVERS}/$op.py
        rm ${RESULTS_DIR}/${kernel}.validate
    done
    deactivate
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

    validation_files_path="${IVYSYN_PATH}/results/tensorflow/synthesized/${dir_to_check}/validate/"
    crashes_path="${IVYSYN_PATH}/results/tensorflow/crashes/${dir_to_check}/"
    crash_type_path_all="${IVYSYN_PATH}/results/tensorflow/crash_types/all/"
    crash_type_path_run="${IVYSYN_PATH}/results/tensorflow/crash_types/${dir_to_check}/"
    asan_out_path="${crashes_path}/asan_output/"
    mkdir -p "${asan_out_path}"

    for subd in ${subdirs[@]}; do
        [[ -d "${crash_type_path_all}${subd}" ]] || mkdir -p "${crash_type_path_all}${subd}"
        [[ -d "${crash_type_path_run}${subd}" ]] || mkdir -p "${crash_type_path_run}${subd}"
    done

    [[ $clean -eq 1 ]] && do_clean "${asan_out_path}" "${crashes_path}" "${crash_type_path_run}"

    pushd "${SCRIPTS_PATH}"

    echo "Producing validation files..."
    source "${ORIG_ENV}"
    python3 synthesizer.py --dir "${dir_to_check}" --validate

    echo "Generating validation list and patching attributes..."
    python3 prep_validation.py --dir "${dir_to_check}" > validation_list.txt
    # Deactivate orig env here
    deactivate

    echo "Validating crashes with drivers..."
    do_validate "${asan_out_path}" "${crash_type_path_all}" "${crash_type_path_run}"

    echo "Moving results..."
    mv ${RESULTS_DIR}/*positive ${crashes_path}
    rm -v $RESULTS_DIR/*validate || true

    echo "Validating crashes without drivers..."
    source "${VALIDATION_ENV}"
    python3 run_kernel_tests_validate.py --dir "${dir_to_check}"
    deactivate

    source "${ORIG_ENV}"
    echo "Synthesizing PoVs..."
    python3 synthesizer.py --dir "${dir_to_check}"

    echo "Checking PoV reproducibility..."
    "${SCRIPTS_PATH}/reproducer.sh" --dir "${dir_to_check}"

    popd

    pushd  "${IVYSYN_SCRIPTS_PATH}"
    echo "Parsing ASan output and categorizing crashes..."
    python3 parse_asan_output.py --dir "${dir_to_check}" --tensorflow
    popd

}

main $@

