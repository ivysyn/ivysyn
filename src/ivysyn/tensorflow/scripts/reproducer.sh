#!/bin/bash

# set -x

TIMEOUT_DUR=30
IVYSYN_PATH="/home/ivyusr/ivysyn/"
TF26_ENV="${IVYSYN_PATH}venv/tensorflow-2.6-orig/bin/activate"
TF29_ENV="${IVYSYN_PATH}venv/tensorflow-2.9-orig/bin/activate"
TF_ASAN_ENV="${IVYSYN_PATH}venv/tensorflow-2.6-asan/bin/activate"
SYNTHESIZED_PATH="${IVYSYN_PATH}results/tensorflow/synthesized/"
subdirs=("segfault" "fpe" "abort" "other")

do_usage()
{
    echo "Usage: ./`basename $0` --dir <synth_folder> [--clean]"
}

do_clean()
{
    synth_dir=$1
    for subd in ${subdirs[@]}; do
        rm -v $synth_dir/v2.6/reproducible/$subd/* || true
        rm -v $synth_dir/v2.9/reproducible/$subd/* || true
    done

    rm -v $synth_dir/v2.6/non-reproducible/* || true
    rm -v $synth_dir/v2.9/non-reproducible/* || true

    rm -v $synth_dir/reproducible/* || true
    rm -v $synth_dir/non-reproducible/* || true
    rm -v $synth_dir/asan_out/* || true
}

do_reproduce_atheris()
{
    export LD_PRELOAD=$(clang --print-file-name libclang_rt.asan-x86_64.so)

    synth_dir=$1

    FILES=$(ls -1 -d $PWD/$synth_dir/all/*.py)
    for f in $FILES; do
        echo $f
        op_name=$(echo $f | rev | cut -d '/' -f 1 | rev | cut -d '_' -f 1)
        # Kill after TIMEOUT_DUR seconds
        asan_out="${PWD}/${synth_dir}/asan_out/"
        mkdir -p "${asan_out}"
        output=$(ASAN_OPTIONS=detect_leaks=0:symbolize=1:detect_odr_violation=0 timeout $TIMEOUT_DUR python3 $f 2>&1)
        exit_code=$?
        fname=$(basename $f)
        # Categorize crashes based on the exit code
        if [[ ! $exit_code =~ ^(0|1)$ ]]; then
            out_path="$PWD/$synth_dir/reproducible/"
        elif [[ ! $output == *"AddressSanitizer"* ]]; then
            out_path="$PWD/$synth_dir/non-reproducible/"
        fi
        [[ -d $out_path ]] || mkdir -p $out_path
        output=${output//$'\n'/}
        cp $f $out_path
        # Append the output message
        sed -i "1i# $output\n" "$out_path/$fname"
    done
}

do_reproduce()
{
    synth_dir=$1
    version_dir=$2

    [[ -d "$synth_dir/$version_dir/reproducible" ]] || mkdir -p "$synth_dir/$version_dir/reproducible"
    [[ -d "$synth_dir/$version_dir/non-reproducible" ]] || mkdir -p "$synth_dir/$version_dir/non-reproducible"

    for subd in ${subdirs[@]}; do
        [[ -d "$synth_dir/$version_dir/reproducible/$subd" ]] || mkdir -p "$synth_dir/$version_dir/reproducible/$subd/"
    done

    FILES=$(ls -1 -d $PWD/$synth_dir/all/*.py)
    for f in $FILES; do
        echo $f
        kernel_name=$(head -1 $f | sed 's/# //')
        # Kill after TIMEOUT_DUR seconds
        output=$(timeout $TIMEOUT_DUR python3 $f 2>&1)
        exit_code=$?
        fname=$(basename $f)
        # Categorize crashes based on the exit code
        if [[ ! $exit_code =~ ^(0|1)$ ]]; then
            sig=$(($exit_code - 128))
            case $sig in
                '11')
                    repr_folder="segfault"
                    ;;
                '8')
                    repr_folder="fpe"
                    ;;
                '6')
                    repr_folder="abort"
                    ;;
                *)
                    repr_folder="other"
                    ;;
            esac
            output="Signal -$sig;$output"
            out_path="$PWD/$synth_dir/$version_dir/reproducible/$repr_folder/$kernel_name"
        else
            out_path="$PWD/$synth_dir/$version_dir/non-reproducible/$kernel_name"
        fi
        [[ -d $out_path ]] || mkdir -p $out_path
        output=${output//$'\n'/}
        cp $f $out_path
        # Append the output message
        sed -i "1i# $output\n" "$out_path/$fname"
    done
}

main()
{

    clean=0
    atheris=0

    while [[ $# -gt 0 ]]; do
        key="$1"
        case $key in
        --dir)
            shift
            synth_folder=$1
            shift
            ;;
        --clean)
            clean=1
            shift
            ;;
        --atheris)
            atheris=1
            shift
            ;;
        *)
            do_usage
            exit 1
            ;;
        esac
    done

    pushd ${SYNTHESIZED_PATH}

    [[ $clean -eq 1 ]] && do_clean $synth_folder

    if [[ $atheris -eq 0 ]]; then
        echo "Checking reproducibility for v2.6..."
        source "${TF26_ENV}"
        do_reproduce "${synth_folder}" "v2.6" $atheris
        deactivate

        echo "Checking reproducibility for v2.9..."
        source "${TF29_ENV}"
        do_reproduce "${synth_folder}" "v2.9" $atheris
        deactivate
    else
        echo "Checking reproducibility Atheris PoVs"
        source "${TF_ASAN_ENV}"
        do_reproduce_atheris "${synth_folder}"
        deactivate
    fi

    popd
}

main $@
