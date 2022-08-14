import argparse
import glob
import hashlib
import inspect
import os

from tensorflow import raw_ops

IVYSYN_PATH = "/home/ivyusr/ivysyn/"
TF_SRC_PATH = os.path.join(
    IVYSYN_PATH, "src/frameworks/tensorflow-2.6-ivysyn/")
TF_IVYSYN_PATH = os.path.join(IVYSYN_PATH, "src/ivysyn/tensorflow/")
TF_FILES_PATH = os.path.join(IVYSYN_PATH, "src/ivysyn/tensorflow")
RESULTS_PATH = os.path.join(IVYSYN_PATH, "results/tensorflow/")

KERNEL_REGS_FILE = os.path.join(TF_FILES_PATH, "kernels_to_ops.txt")
CRASHFILES_PATH_BASE = os.path.join(RESULTS_PATH, "crashes/")
REPRODUCE_PATH_BASE = os.path.join(RESULTS_PATH, "synthesized/")
TYPES_PATH = os.path.join(RESULTS_PATH, "types/")
RES_PATH = os.path.join(TYPES_PATH, "to_ops/")
NOT_SYNTHED_PATH = os.path.join(CRASHFILES_PATH_BASE, "to-check/")

CRASH_DELIM = "--------------------------------------\n"


def get_tensor_type(dtype):
    if dtype == "DT_FLOAT":
        return "tf.float32"
    if dtype == "DT_DOUBLE":
        return "tf.float64"
    return dtype.replace("DT_", "tf.").lower()


def get_tf_type(ttype):
    if ttype == "half":
        return "float16"
    if ttype == "float":
        return "float32"
    if ttype == "double":
        return "float64"
    return ttype


def get_function_param_names(op_name):
    raw_op_fn = eval(f"raw_ops.{op_name}")
    params = inspect.signature(raw_op_fn).parameters.values()
    param_names = [param.name for param in params]
    return param_names


def handle_value_edge_cases(value):
    """Handle bad parsing"""

    value = value.replace("...", "")

    # Do negatives first, because of negative decimals
    if value.count("-") > 1:
        if 'e' not in value:
            value = "-" + value.split("-")[1]
            return value

    if value.count(".") > 1:
        value = "." + value.split(".")[1]

    if len(value) > 1 and value[0] == "0" and "." not in value:
        value = value[1:]

    return value


def parse_crash_argument(arg):
    attrs = arg.split(":")

    # This usually means unknown type (e.g., 'Resource' or 'Variant')
    # which is not printed as expected
    if len(attrs) < 3:
        return None

    tensor_type = attrs[1].replace(" shape", "").strip(" ")
    tensor_shape = attrs[2].replace(" values", "").strip(" ")
    tensor_values = attrs[3].strip(">").replace(
        "[", "").replace("]", "").split(" ")

    tensor_values = list(filter(None, tensor_values))
    if '?' in tensor_values:
        tensor_values = ['1']
    return tensor_type, tensor_shape, tensor_values


def split_attrs(attrs):

    if len(attrs) == 0:
        return []

    parsed_attrs = []
    attrs = attrs.split(", ")

    idx = 0
    for attr in attrs:

        if "=" in attr:
            parsed_attrs.append(attr)
            idx += 1
        else:
            if len(parsed_attrs) > 0:
                parsed_attrs[idx - 1] += ", " + attr

    return parsed_attrs


def parse_attrs(attrs):

    attrs = split_attrs(attrs)
    attrs_dict = {}

    for attr in attrs:
        attr = attr.split("=")
        attr_name = attr[0]
        attr_value = '='.join(attr[1:])

        if attr_name in ("dtype", "dt", "index_type", "output_dtype", "out_type",
                         "out_values_type", "out_row_splits_type", "out_idx", "output_idx_type"
                         "out_idx_type", "internal_type") or attr_name.startswith("T"):
            attr_value = get_tensor_type(attr_value)

        if attr_name in ("dtypes",):
            attr_values = attr_value[1:-1].split(", ")
            attr_value = "[" + ", ".join(get_tensor_type(x)
                                         for x in attr_values) + "]"

        if attr_name in ("cond", "body"):
            attr_value = "[]"

        if attr_value == "true":
            attr_value = "True"
        if attr_value == "false":
            attr_value = "False"

        # Edge case
        if attr_value.startswith("Tensor<") and attr_value.endswith(">"):
            continue
            # attr_value = synthesize_args([attr_value], "''", [])[
            #     0].split('=')[1:]
            # attr_value = '='.join(attr_value)[1:]
            # attr_value = 'tf.make_tensor_proto(' + attr_value
            # print(attr_value)

        attrs_dict[attr_name] = attr_value

    return attrs_dict


def synthesize_args(crashing_args, param_names, attrs, validate):

    input_args = []

    # Create attrs, if any
    for param_name in param_names:
        if param_name in attrs:
            attr_str = f"{param_name} = {attrs[param_name]}"
            input_args.append(attr_str)

    param_names = [x for x in param_names if x not in attrs]

    for idx, arg in enumerate(crashing_args):

        if validate:
            param_name = ""
        else:
            if idx >= len(param_names):
                break

            param_name = param_names[idx]

            if param_name == "name":
                break

        crash_args = parse_crash_argument(arg)

        # Will return none if it finds an arg in an unexpected format
        if crash_args is None:
            return None

        tensor_type, tensor_shape, tensor_values = crash_args

        if len(tensor_values) > 0:
            if tensor_shape == "[2]" or tensor_shape == "[3]":
                value = ",".join(tensor_values)
                if not validate:
                    value = "[" + value + "]"
            else:
                value = tensor_values[0]
                value = handle_value_edge_cases(value)
        else:
            value = "[]"

        if tensor_type == "string" and not validate:
            value = '"' + value + '"'

        if tensor_type == "bool":
            if value == "0":
                value = "False"
            else:
                value = "True"

        if tensor_type == "variant":
            return None

        if validate:
            tensor_shape = tensor_shape.replace('[', '').replace(']', '')
            fuzz_tensor = f"{value}\n{tensor_shape}\n{get_tf_type(tensor_type)}"
        else:
            fuzz_tensor = f"{param_name} = tf.constant({value}, shape={tensor_shape}, dtype=tf.{get_tf_type(tensor_type)})"
        input_args.append(fuzz_tensor)

    return input_args


def synthesize_file(crash, kernel_name, op_name, validate=False):

    if op_name is None:
        print(f"No op name registered for {kernel_name}")
        return -1

    synth_file = []

    if not validate:
        synth_file.append("# " + kernel_name + "\n")
        synth_file.append("import tensorflow as tf\n")

    crashing_args = crash.split("\n")
    attrs = parse_attrs(crashing_args[0])

    crashing_args = crashing_args[1:]

    if len(crashing_args) == 0:
        return -3

    if validate:
        param_names = []
    else:
        try:
            param_names = get_function_param_names(op_name)
        except AttributeError:
            # No raw op for this
            print(f"No raw op found for {kernel_name}, skipping")
            if not validate:
                return -1

    input_args = synthesize_args(crashing_args, param_names, attrs, validate)

    if input_args is None:
        # Contains unsupported type
        # print(f"unsupported type in {kernel_name}, skipping")
        return -2

    synth_file.extend(input_args)

    if not validate:
        kwargs = ["{}={}".format(param_names[idx], param_names[idx])
                  for idx in range(len(input_args))]

        synth_file.append(f"tf.raw_ops.{op_name}({', '.join(kwargs)})")

    return "\n".join(synth_file)


def get_kernel_name(filename, crashes_path, ext):
    crash_filename = filename.split("/")[-1]
    kernel_name = crash_filename.replace(crashes_path, "").replace(ext, "")
    return kernel_name


def save_validate_file(synth_file, kernel_name, reproduce_path):
    out_filename = reproduce_path + kernel_name + ".validate"

    # No duplicates
    if not os.path.isfile(out_filename):
        with open(out_filename, "w") as f:
            f.write(synth_file + "\n")
            f.close()


def save_synth_file(synth_file, op_name, reproduce_path, types=False):
    # Get the hash of the synthesized file to make sure we only
    # have unique reproduced files
    filehash = hashlib.md5(synth_file.encode()).hexdigest()
    out_filename = reproduce_path + op_name

    if not types:
        out_filename += "_" + filehash

    out_filename += ".py"

    # No duplicates
    if not os.path.isfile(out_filename):
        with open(out_filename, "w") as f:
            f.write(synth_file + "\n")
            f.close()


def main():

    successful = set()
    empty = set()
    successful_ops = set()
    all_kernels = set()
    no_raw_op = set()
    synth_failed = set()
    unsupported_type = set()
    other_errors = set()
    no_args = set()

    args_parser = argparse.ArgumentParser(
        description="Parse and transform Pytorch native files")

    arg_group = args_parser.add_mutually_exclusive_group(required=True)

    arg_group.add_argument(
        "--types", dest="types", action="store_true", default=False)

    arg_group.add_argument("--dir", dest="dir")

    args_parser.add_argument("--validate", dest="validate",
                             action="store_true", default=False)

    args = args_parser.parse_args()

    kernels_to_ops = {}
    with open(KERNEL_REGS_FILE, "r") as f:
        regs = f.read().strip().split("\n")
        for reg in regs:
            kernel_name, ops = reg.split(" ")
            ops = ops.split(',')
            kernels_to_ops[kernel_name] = ops

    if args.types:
        crashes_path = os.path.join(
            RESULTS_PATH, "atheris_comp/kernel_types/logged_types/")
    else:
        crashes_path = CRASHFILES_PATH_BASE + args.dir + "/"

    true_positives = glob.glob(crashes_path + "*.true_positive")
    true_positives = [x.split('/')[-1].replace(".true_positive", "")
                      for x in true_positives]

    ext = ".types" if args.types else "_crashes.log"
    logged_files = glob.glob(crashes_path + "*" + ext)

    if args.types:
        reproduce_path = os.path.join(
            RESULTS_PATH, "atheris_comp/kernel_types/synthed_ops/")
    elif args.validate:
        reproduce_path = REPRODUCE_PATH_BASE + args.dir + "/validate/"
    else:
        reproduce_path = REPRODUCE_PATH_BASE + args.dir + "/all/"

    if not os.path.isdir(reproduce_path):
        os.makedirs(reproduce_path)

    for crash_filename in logged_files:

        kernel_name = get_kernel_name(crash_filename, crashes_path, ext)

        # Only synthesize true crashes
        # if not args.validate and kernel_name not in true_positives:
        #     continue

        b_kernel_name = kernel_name
        if kernel_name.endswith("Base") or kernel_name.endswith("BaseOp"):
            b_kernel_name = kernel_name.replace("Base", "")

        cand_ops = kernels_to_ops[b_kernel_name]

        if args.validate:
            cand_ops = cand_ops[0]

        for op_name in cand_ops:

            # Ignore empty files
            if os.path.getsize(crash_filename) == 0:
                empty.add(kernel_name)
                continue

            all_kernels.add(kernel_name)

            with open(crash_filename, "r") as crash_file:
                try:
                    crashes = list(
                        filter(None, crash_file.read().split(CRASH_DELIM)))
                except UnicodeDecodeError as e:
                    other_errors.add(kernel_name)
                    continue

            if len(crashes) == 0:
                empty.add(kernel_name)
                continue

            if len(crashes) == 1 and crashes[0] == "\n\n":
                empty.add(kernel_name)
                continue

            for crash in crashes:

                crash = crash.rstrip()

                if len(crash) == 0:
                    print("Skipping empty crash for", kernel_name)
                    continue

                synth_file = synthesize_file(
                    crash, kernel_name, op_name, args.validate)

                if synth_file is None:
                    synth_failed.add(kernel_name)
                    continue
                if synth_file == -1:
                    no_raw_op.add(kernel_name)
                    # shutil.copy(crash_filename, NOT_SYNTHED_PATH)
                    continue
                if synth_file == -2:
                    unsupported_type.add(kernel_name)
                    continue
                if synth_file == -3:
                    no_args.add(kernel_name)
                    continue

                successful.add(kernel_name)
                successful_ops.add(op_name)
                if args.validate:
                    save_validate_file(synth_file, kernel_name, reproduce_path)
                else:
                    save_synth_file(synth_file, op_name,
                                    reproduce_path, args.types)

    no_raw_op = list([x for x in no_raw_op if x not in successful])
    unsupported_type = list(
        [x for x in unsupported_type if x not in successful])
    print("No raw op:")
    print("\n".join(no_raw_op))
    print("unsupported type:")
    print("\n".join(unsupported_type))
    print(f"Total synth_failed: {len(synth_failed)}")
    print(f"Total no raw op: {len(no_raw_op)}")
    print(f"Total empty files: {len(empty)}")
    print(f"Total no args: {len(no_args)}")
    print(f"Total unsupported type: {len(unsupported_type)}")
    print(f"Total other errors: {len(other_errors)}")
    print(f"Total successful: {len(successful)}")
    print(f"Total crash files: {len(all_kernels)}")
    print(f"Total successful (unique ops): {len(successful_ops)}")


if __name__ == "__main__":
    main()
