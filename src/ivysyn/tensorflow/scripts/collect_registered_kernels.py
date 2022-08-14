import glob
import os
import re

IVYSYN_PATH = "/home/ivyusr/ivysyn/"
TF_IVYSYN_PATH = os.path.join(IVYSYN_PATH, "src/ivysyn/tensorflow/")
RAW_REGS_FILE = os.path.join(TF_IVYSYN_PATH, "raw_regs.txt")
KERNELS_TO_OPS_MAPPINGS = os.path.join(TF_IVYSYN_PATH, "kernels_to_ops.txt")
OPS_TO_KERNELS_MAPPINGS = os.path.join(TF_IVYSYN_PATH, "ops_to_kernels.txt")
TF_SRC_PATH = os.path.join(
    IVYSYN_PATH, "src/frameworks/tensorflow-2.6-ivysyn/")
TF_KERNELS_PATH = TF_SRC_PATH + "tensorflow/core/kernels/"
DELIM = " ----- "


def get_kernel_regs_regex(kernel_regs):
    kernel_files = glob.glob(TF_KERNELS_PATH + "**/*.cc", recursive=True) + glob.glob(
        TF_KERNELS_PATH + "**/*.h", recursive=True
    )
    for filename in kernel_files:
        with open(filename, "r") as f:
            data = f.read().strip()
            registrations = re.findall(
                r'REGISTER_KERNEL_BUILDER\(.*?Name\("(.*?)"\).*?\),(.*?)\)', data, flags=re.DOTALL
            )
            for reg in registrations:
                parsed_reg = [re.sub("\<.*", "", x, flags=re.DOTALL)
                              for x in reg]
                parsed_reg = [re.sub("\s+", "", x) for x in parsed_reg]
                parsed_reg = [re.sub("\\\\", "", x) for x in parsed_reg]
                op_name = parsed_reg[0]
                kernel_name = parsed_reg[1]

                if kernel_name.endswith("Base"):
                    kernel_name = kernel_name.replace("Base", "")

                if kernel_name not in kernel_regs:
                    kernel_regs[kernel_name] = set()

                kernel_regs[kernel_name].add(op_name)

    return kernel_regs


def get_kernel_regs():

    kernel_regs = {}

    with open(RAW_REGS_FILE, "r") as f:
        raw_regs = f.read().strip().split('\n')

    for reg in raw_regs:
        kernel_name, op_name = reg.split(DELIM)
        kernel_name = kernel_name.split('<')[0]
        kernel_name = kernel_name.split(':')[-1]
        kernel_name = kernel_name.rstrip()
        if kernel_name.startswith('('):
            kernel_name = kernel_name[1:]

        if kernel_name not in kernel_regs:
            kernel_regs[kernel_name] = set()

        kernel_regs[kernel_name].add(op_name)

    kernel_regs = get_kernel_regs_regex(kernel_regs)
    return kernel_regs


def get_ops_to_kernels_mappings(kernel_regs):

    ops_to_kernels = {}

    for kernel, ops in kernel_regs.items():
        for op in ops:
            if op not in ops_to_kernels:
                ops_to_kernels[op] = set()
            ops_to_kernels[op].add(kernel)

    return ops_to_kernels


def main():

    kernel_regs = get_kernel_regs()

    with open(KERNELS_TO_OPS_MAPPINGS, "w") as f:
        for kernel, ops in kernel_regs.items():
            f.write(f"{kernel} {','.join(ops)}\n")

    ops_to_kernels = get_ops_to_kernels_mappings(kernel_regs)

    with open(OPS_TO_KERNELS_MAPPINGS, "w") as f:
        for op, kernels in ops_to_kernels.items():
            f.write(f"{op} {','.join(kernels)}\n")


if __name__ == "__main__":
    main()
