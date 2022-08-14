#!/bin/bash
# set -x

TIMEOUT_DUR=30
subdirs=("segfault" "fpe" "abort" "other")
IVYSYN_PATH="/home/ivyusr/ivysyn/"
CONDA_ACTIVATE_PATH="${IVYSYN_PATH}venv/anaconda3/bin/activate"
SYNTHESIZED_PATH="${IVYSYN_PATH}results/pytorch/synthesized/"

do_usage()
{
    echo "Usage: ./`basename $0` --dir <synth_folder> [--clean]"
}

do_clean()
{
    synth_dir=$1
    for subd in ${subdirs[@]}; do
        rm $synth_dir/v1.11/reproducible/$subd/* || true
        rm $synth_dir/v1.12/reproducible/$subd/* || true
    done

    rm $synth_dir/v1.11/non-reproducible/* || true
    rm $synth_dir/v1.12/non-reproducible/* || true
}

do_reproduce()
{
    synth_dir=$1
    version_dir=$2

    [[ -d "$synth_dir/$version_dir/reproducible" ]] || mkdir -p "$synth_dir/$version_dir/reproducible"
    [[ -d "$synth_dir/$version_dir/non-reproducible" ]] || mkdir -p "$synth_dir/$version_dir/non-reproducible"

    for subd in ${subdirs[@]}; do
        [[ -d "$synth_dir/reproducible/$subd" ]] || mkdir -p "$synth_dir/reproducible/$subd/"
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
        *)
            do_usage
            exit 1
            ;;
        esac
    done

    pushd ${SYNTHESIZED_PATH}

    [[ $clean -eq 1 ]] && do_clean $synth_folder

    source "${CONDA_ACTIVATE_PATH}"

    echo "Checking reproducibility for v1.11..."
    conda activate pytorch-1.11-orig
    do_reproduce $synth_folder "v1.11"
    conda deactivate

    echo "Checking reproducibility for v1.12..."
    conda activate pytorch-1.12-orig
    do_reproduce $synth_folder "v1.12"
    conda deactivate

    popd

}

main $@
