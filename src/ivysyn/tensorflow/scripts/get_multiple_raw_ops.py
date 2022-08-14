
import os

IVYSYN_PATH = "/home/ivyusr/ivysyn/"
TF_IVYSYN_PATH = os.path.join(IVYSYN_PATH, "src/ivysyn/tensorflow/")
KERNEL_REGS_FILE = os.path.join(TF_IVYSYN_PATH, "kernels_to_ops.txt")
INSTRUMENTED_FILE = os.path.join(
    IVYSYN_PATH, "results/tensorflow/instrumentation/cpu_instrumented.txt")


def get_multiple_raw_ops():

    multiple_raw_ops = {}

    with open(KERNEL_REGS_FILE, "r") as f:
        kernels_to_ops = f.read().strip().split('\n')

    for reg in kernels_to_ops:
        kernel, ops = reg.split(' ')
        ops = ops.split(',')

        if kernel == 'NoOp':
            continue

        if len(ops) > 2:
            multiple_raw_ops[kernel] = ops

    return multiple_raw_ops


def main():

    with open(INSTRUMENTED_FILE, "r") as f:
        instrumented = f.read().strip().split('\n')

    multiple_raw_ops = get_multiple_raw_ops()
    multiple_instrumented = [
        x for x in multiple_raw_ops.keys() if x in instrumented]

    print(
        f"Kernels with more than two raw ops ({len(multiple_raw_ops)}) (Instrumented: {len(multiple_instrumented)}):")
    # print("\n".join(f"{k}: {v}" for k, v in multiple_raw_ops.items()))

    more_than_3 = {k: v for k, v in multiple_raw_ops.items() if len(v) > 3}
    more_that_3_instrumented = [
        x for x in more_than_3.keys() if x in instrumented]

    print(
        f"Kernels with more than three raw ops ({len(more_than_3)}) :")
    print(f"Instrumented: ({len(more_that_3_instrumented)})")
    print("\n".join(f"{k}: {v}" for k, v in more_than_3.items()
                    if k in more_that_3_instrumented))


if __name__ == "__main__":
    main()
