import argparse
import glob
import os
import re

from synthesizer import get_function_param_names, get_tf_type, parse_attrs, parse_crash_argument

IVYSYN_PATH = "/home/ivyusr/ivysyn/"
TF_IVYSYN_PATH = os.path.join(IVYSYN_PATH, "src/ivysyn/tensorflow/")
VALIDATION_DRIVERS = os.path.join(TF_IVYSYN_PATH, "validation_drivers/")
RESULT_DIR_BASE = os.path.join(IVYSYN_PATH, "results/tensorflow/")
SYNTH_DIR_BASE = os.path.join(RESULT_DIR_BASE, "synthesized/")
CRASHES_DIR_BASE = os.path.join(RESULT_DIR_BASE, "crashes/")
ATHERIS_FUZZ_LIST = os.path.join(
    TF_IVYSYN_PATH, "scripts/atheris_comp_scripts/atheris_fuzz_list.txt")
OPS_TO_KERNELS_FILE = os.path.join(TF_IVYSYN_PATH + "ops_to_kernels.txt")


def gen_validation_list(validation_kernels, kernel_mappings):

    validation_list = {}
    for kernel_name in validation_kernels:
        if kernel_name.endswith("BaseOp") or kernel_name.endswith("OpBase"):
            b_kernel_name = kernel_name.replace("Base", "")
        else:
            b_kernel_name = kernel_name
        if b_kernel_name not in kernel_mappings:
            # print(f"No validation driver found for {b_kernel_name}")
            continue
        op_names = kernel_mappings[b_kernel_name]
        for op_name in op_names:
            validation_list[op_name] = kernel_name

    return validation_list


def patch_driver_attributes(validation_list, crashes_dir):

    for op, kernel in validation_list.items():

        crash_file = crashes_dir + kernel + "_crashes.log"
        driver = VALIDATION_DRIVERS + op + ".py"

        with open(crash_file, "r") as f:
            crash = f.read().rstrip()
            crashing_args = crash.split("\n")
            attrs = parse_attrs(crashing_args[0])

        with open(driver, "r") as f:
            driver_lines = f.read().strip().split("\n")

        # Find the lines containing attrs/args, match to attr names and replace
        for i, line in enumerate(driver_lines):
            arg = line.split(" = ")
            if not len(arg) >= 2:
                continue
            arg_name = arg[0]
            if arg_name in attrs:
                line = re.sub(fr'{arg_name} = .*',
                              fr"{arg_name} = {attrs[arg_name]}",
                              line)
                driver_lines = driver_lines[:i] + [line] + driver_lines[i + 1:]

        with open(driver, "w") as f:
            f.write("\n".join(driver_lines) + "\n")


def patch_driver_argtypes(validation_list, crashes_dir):

    for op, kernel in validation_list.items():

        crash_file = crashes_dir + kernel + "_crashes.log"
        driver = VALIDATION_DRIVERS + op + ".py"

        with open(crash_file, "r") as f:
            crash = f.read().rstrip()
            crashing_args = crash.split("\n")

        attrs = parse_attrs(crashing_args[0])
        crashing_args = crashing_args[1:]
        argnames = get_function_param_names(op)
        argnames = [x for x in argnames if x not in attrs]

        arg_types = []
        for arg in crashing_args:
            res = parse_crash_argument(arg)
            if res is None:
                continue
            arg_type = res[0]
            arg_types.append(arg_type)

        with open(driver, "r") as f:
            driver_lines = f.read().strip().split("\n")

        api_args = [x for x in driver_lines if any(
            [x.startswith(arg_name) for arg_name in argnames])]
        # print(op, kernel)
        # print(len(api_args))
        # print(len(arg_types))
        arg_idx = 0
        if len(api_args) != len(arg_types):
            continue
        for i, line in enumerate(driver_lines):
            arg = line.split(" = ")
            if not len(arg) >= 2:
                continue
            arg_name = arg[0]
            if arg_name in argnames and line.startswith(arg_name):
                arg_type = get_tf_type(arg_types[arg_idx])
                cur_arg_type = line[line.index("dtype=") + 9: -1]
                if arg_type == cur_arg_type:
                    continue
                if arg_type == "bool":
                    dummy_val = True
                elif arg_type == "string":
                    dummy_val = ""
                else:
                    dummy_val = "1"
                arg_idx += 1
                line = re.sub(fr'({arg_name} = .*dtype=)(.*)\)',
                              fr"\1tf.{arg_type})",
                              line)
                line = re.sub(fr'({arg_name} = .*\().*(,.*\))',
                              fr"\1 {dummy_val}\2",
                              line)
                driver_lines = driver_lines[:i] + [line] + driver_lines[i + 1:]

        with open(driver, "w") as f:
            f.write("\n".join(driver_lines) + "\n")


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
    fuzzed_ops = [x.split('/')[-1].replace('.py', '') for x in fuzzed_ops]

    ops_to_kernels = {}
    with open(OPS_TO_KERNELS_FILE, "r") as f:
        mappings = f.read().strip().split("\n")
        for mapping in mappings:
            op_name, kernels = mapping.split(' ')
            kernels = kernels.split(',')
            kernels = [x for x in kernels if x !=
                       "NoOp" and "GPU" not in x.upper()]
            ops_to_kernels[op_name] = kernels

    kernel_mappings = {}
    for op in fuzzed_ops:
        kernels = ops_to_kernels[op]
        kernel = kernels[0]
        if kernel not in kernel_mappings:
            kernel_mappings[kernel] = []
        kernel_mappings[kernel].append(op)

    validation_list = gen_validation_list(validation_kernels, kernel_mappings)

    patch_driver_attributes(validation_list, crashes_dir)
    # patch_driver_argtypes(validation_list, crashes_dir)

    # This should run first for validation if it exists since it gets called
    # by other Ops
    if 'Fill' in validation_list:
        del validation_list['Fill']
        print("Fill:FillOp")
    print('\n'.join([f"{k}:{v}" for k, v in validation_list.items()]))


if __name__ == "__main__":
    main()
