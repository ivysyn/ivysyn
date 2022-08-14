import argparse
import glob
import os

IVYSYN_PATH = "/home/ivyusr/ivysyn/"
PT_IVYSYN_PATH = os.path.join(IVYSYN_PATH, "src/ivysyn/pytorch/")
RESULT_DIR_BASE = os.path.join(IVYSYN_PATH, "results/pytorch/")
VALIDATION_DRIVERS = os.path.join(
    RESULT_DIR_BASE, "atheris_comp/one_to_one_pt/")
SYNTH_DIR_BASE = os.path.join(RESULT_DIR_BASE, "synthesized/")
CRASHES_DIR_BASE = os.path.join(RESULT_DIR_BASE, "crashes/")
ATHERIS_FUZZ_LIST = os.path.join(
    PT_IVYSYN_PATH, "scripts/atheris_comp_scripts/atheris_fuzz_list.txt")
OPS_TO_KERNELS_FILE = os.path.join(PT_IVYSYN_PATH + "ops_to_kernels.txt")


def gen_validation_list(validation_kernels, kernel_mappings):

    validation_list = {}
    for kernel_name in validation_kernels:
        if kernel_name not in kernel_mappings:
            continue
        op_names = kernel_mappings[kernel_name]
        for op_name in op_names:
            validation_list[op_name] = kernel_name

    return validation_list


def main():

    args_parser = argparse.ArgumentParser()

    args_parser.add_argument(
        "--dir", dest="dir", required=True)

    args = args_parser.parse_args()

    synthesized_dir = os.path.join(SYNTH_DIR_BASE, args.dir + "/validate/")
    crashes_dir = os.path.join(CRASHES_DIR_BASE, args.dir + "/")

    validation_kernels = glob.glob(synthesized_dir + "*.validate")
    validation_kernels = [x.split('/')[-1].replace('.validate', '')
                          for x in validation_kernels]

    # with open(ATHERIS_FUZZ_LIST, "r") as f:
    #     fuzzed_ops = f.read().strip().split('\n')

    fuzzed_ops = glob.glob(VALIDATION_DRIVERS + "*.py")
    fuzzed_ops = [x.split('/')[-1].replace('_op.py', '') for x in fuzzed_ops]

    ops_to_kernels = {}
    with open(OPS_TO_KERNELS_FILE, "r") as f:
        mappings = f.read().strip().split("\n")
        for mapping in mappings:
            op_name, kernels = mapping.split(' ')
            kernels = kernels.split(',')
            ops_to_kernels[op_name] = kernels

    kernel_mappings = {}
    for op in fuzzed_ops:
        kernels = ops_to_kernels[op]
        kernel = kernels[0]
        if kernel not in kernel_mappings:
            kernel_mappings[kernel] = []
        kernel_mappings[kernel].append(op)

    validation_list = gen_validation_list(validation_kernels, kernel_mappings)

    # This should run first for validation if it exists since it gets called
    # by other Ops
    print('\n'.join([f"{k}:{v}" for k, v in validation_list.items()]))


if __name__ == "__main__":
    main()
