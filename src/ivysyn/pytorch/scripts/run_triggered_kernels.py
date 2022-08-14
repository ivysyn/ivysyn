import os
import subprocess
from glob import glob

PYTORCH_PATH = "/home/ivyusr/ivysyn/src/frameworks/pytorch-1.11-ivysyn/"
RESULTS_PATH = "/mnt/pytorch-ivysyn/"
TRIGGERED_KERNEL_FILE = os.path.join(RESULTS_PATH, "triggered_kernels.txt")
asan_out = "/home/ivyusr/ivysyn/results/pytorch/crashes/full_run/asan_output/"
asan_rt = "/usr/lib/llvm-11/lib/clang/11.0.1/lib/linux/libclang_rt.asan-x86_64.so"


def main():

    os.chdir(PYTORCH_PATH)

#     with open(TRIGGERED_KERNEL_FILE, "r") as f:
#         lines = f.read().strip().split('\n')

#     triggered_kernels = {}

#     for line in lines:
#         kernel, test = line.split(' ')
#         triggered_kernels[kernel] = test

    triggered_kernels = {
        # "_convolution_double_backward": "/home/ivyusr/ivysyn/src/frameworks/pytorch-1.11-ivysyn/test/test_nn.py",
        "multi_margin_loss_cpu_backward": "/home/ivyusr/ivysyn/src/frameworks/pytorch-1.11-ivysyn/test/test_nn.py",
        "_batch_norm_impl_index_backward": "/home/ivyusr/ivysyn/src/frameworks/pytorch-1.11-ivysyn/test/test_jit_legacy.py",
    }

    for kernel, test in triggered_kernels.items():

        pos_file = glob(RESULTS_PATH + kernel + "*positive")
        if len(pos_file) > 0:
            os.remove(pos_file[0])
        print(kernel, test)

        args = ["python3", test]
        env = os.environ.copy()
        env['ASAN_OPTIONS'] = f"detect_leaks=0:symbolize=1:detect_odr_violation=0:allocator_may_return_null=1:log_path='{asan_out}{kernel}'"
        env['LD_PRELOAD'] = asan_rt

        p = subprocess.Popen(args, env=env,)
        p.wait()
        # p = subprocess.Popen(args, env=env,)
        # p.wait()


if __name__ == "__main__":
    main()
