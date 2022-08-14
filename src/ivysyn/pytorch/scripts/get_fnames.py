import os

import yaml

try:
    from yaml import CSafeLoader as Loader
except ImportError:
    from yaml import SafeLoader as Loader  # type: ignore

IVYSYN_PATH = "/home/ivyusr/ivysyn/"
PYTORCH_PATH = os.path.join(IVYSYN_PATH, "src/frameworks/pytorch-1.11-ivysyn/")
PYTORCH_IVYSYN_PATH = os.path.join(IVYSYN_PATH, "src/ivysyn/pytorch/")

native_yaml_path = os.path.join(PYTORCH_IVYSYN_PATH,
                                "native_functions_no_dups.yaml")


def get_fnames_and_dispatches():
    fnames = set()
    fnames_gpu = set()
    dispatches = {}

    with open(native_yaml_path, "r") as f:
        entries = yaml.load(f, Loader=Loader)

    for entry in entries:

        entry_fnames = []
        fname = entry.get("func")

        fname = fname.split("(")[0].split(".")[0]

        if "dispatch" in entry:
            for key, v in entry.get("dispatch").items():

                entry_fnames.append(v)
                if key in ["CUDA", "SparseCsrCUDA", "SparseCUDA"]:
                    if '_out' in v:
                        fnames_gpu.add(fname)
                    else:
                        fnames_gpu.add(v)
                dispatches[v] = fname
        else:
            entry_fnames.append(fname)

        for fname in entry_fnames:
            if "foreach" in fname:
                continue
            fnames.add(fname)

    return fnames, dispatches, fnames_gpu


def main():

    fnames, dispatches, fnames_gpu = get_fnames_and_dispatches()
    print(f"Total functions: {len(fnames)}")

    # fnames_gpu = [x for x in fnames if "cuda" in x]
    # fnames_gpu = [x.replace("_cuda_out", "") for x in fnames_gpu]
    # fnames_gpu = [x.replace("_out_cuda", "") for x in fnames_gpu]

    with open(os.path.join(PYTORCH_IVYSYN_PATH, "function_names.txt"), "w") as f:
        f.write("\n".join(fnames))

    with open(os.path.join(PYTORCH_IVYSYN_PATH, "function_names_gpu.txt"), "w") as f:
        f.write("\n".join(fnames_gpu))

    with open(os.path.join(PYTORCH_IVYSYN_PATH, "dispatches.txt"), "w") as f:
        f.write("\n".join([f"{k} {v}" for k, v in dispatches.items()]))


if __name__ == "__main__":
    main()
