import glob
import os
import zipfile

IVYSYN_PATH = "/home/ivyusr/ivysyn/"
TF_SRC_PATH = os.path.join(
    IVYSYN_PATH, "src/frameworks/tensorflow-2.6-ivysyn/")
TF_IVYSYN_PATH = os.path.join(IVYSYN_PATH, "src/ivysyn/tensorflow/")
OPS_PATH = os.path.join(IVYSYN_PATH,
                        "results/tensorflow/atheris_comp/kernel_types/synthed_ops/")
KERNEL_REGS_FILE = os.path.join(TF_IVYSYN_PATH, "kernels_to_ops.txt")
ONE_TO_ONE_MAPPINGS_FILE = os.path.join(
    TF_IVYSYN_PATH, "one_to_one_mappings.txt")
ONE_TO_ONE_KERNELS_FILE = os.path.join(
    TF_IVYSYN_PATH, "one_to_one_kernels.txt")
ONE_TO_ONE_OPS_FILE = os.path.join(
    TF_IVYSYN_PATH, "one_to_one_ops.txt")
OUT_FILE = os.path.join(
    IVYSYN_PATH, "results/tensorflow/atheris_comp/one_to_one_tf.zip")
OUT_FILE_MORE_THAN_ONE = os.path.join(
    IVYSYN_PATH, "results/tensorflow/atheris_comp/more_than_one.zip")

EXCLUDE_KERNELS = ['BatchNormOp', 'BatchNormGradOp', 'CompositeTensorVariantFromComponents',
                   'RaggedCrossOp', 'SummaryImageOp', 'WhileOp']


def main():

    op_mappings = {}
    op_mappings_gpu = {}
    mapped_kernels = []
    mapped_kernels_gpu = []

    with open(KERNEL_REGS_FILE, "r") as f:
        kernels_to_ops = f.read().strip().split('\n')

    for reg in kernels_to_ops:
        kernel, ops = reg.split(' ')
        if kernel == 'NoOp':
            continue
        ops = ops.split(',')
        for op in ops:
            if op not in op_mappings:
                op_mappings[op] = []
                op_mappings_gpu[op] = []

            if 'Gpu' not in kernel and 'GPU' not in kernel:
                op_mappings[op].append(kernel)

            op_mappings_gpu[op].append(kernel)

    op_files = glob.glob(OPS_PATH + "*.py")
    op_names = [x.split('/')[-1].replace('.py', '') for x in op_files]

    print(f"Total ops: {len(op_names)}")
    print(f"Total kernels: {len(mapped_kernels)}")
    print(f"Total kernels (with Gpu): {len(mapped_kernels_gpu)}")

    synthed_ops = {k: v for k, v in op_mappings.items() if k in op_names}
    one_to_one = {k: v for k, v in synthed_ops.items() if len(v) == 1}
    more_than_one = {k: v for k, v in synthed_ops.items() if len(v) > 1}
    more_than_two = {k: v for k, v in synthed_ops.items() if len(v) > 2}

    print(f"Total ops with 1-1 mapping to kernels: {len(one_to_one)}")
    print(f"Ops with more than one mapping ({len(more_than_one)}):")
    print("\n".join(f"{k}: {v}" for k, v in more_than_one.items()))

    one_to_one = {k: v for k, v in one_to_one.items()
                  if not any([x in EXCLUDE_KERNELS for x in v])}
    one_to_one_filenames = [f"{x}.py" for x in one_to_one.keys()]
    with zipfile.ZipFile(OUT_FILE, mode="w") as zipfile_out:
        for filename in one_to_one_filenames:
            zipfile_out.write(os.path.join(
                OPS_PATH, filename), arcname=f"one_to_one_tf/{filename}")
        zipfile_out.close()

    more_than_one_filenames = [f"{x}.py" for x in more_than_one.keys()]
    with zipfile.ZipFile(OUT_FILE_MORE_THAN_ONE, mode="w") as zipfile_out:
        for filename in more_than_one_filenames:
            zipfile_out.write(os.path.join(
                OPS_PATH, filename), arcname=f"more_than_one_tf/{filename}")
        zipfile_out.close()

    with open(ONE_TO_ONE_MAPPINGS_FILE, "w") as f:
        for k, v in one_to_one.items():
            f.write(f"{v[0]} {k}\n")

    with open(ONE_TO_ONE_KERNELS_FILE, "w") as f:
        f.write('\n'.join(sorted([v[0] for v in one_to_one.values()])))
        f.write('\n')

    with open(ONE_TO_ONE_OPS_FILE, "w") as f:
        f.write('\n'.join(sorted([k for k in one_to_one.keys()])))
        f.write('\n')


if __name__ == "__main__":
    main()
