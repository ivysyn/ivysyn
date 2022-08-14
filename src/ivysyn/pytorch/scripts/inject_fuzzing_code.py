import argparse
import glob
import os
import re
from collections import Counter

from clang.cindex import *
from tqdm import tqdm

IVYSYN_PATH = "/home/ivyusr/ivysyn/"
PYTORCH_IVYSYN_PATH = os.path.join(IVYSYN_PATH, "src/ivysyn/pytorch/")
dirname = os.path.dirname(__file__)
PYTORCH_PATH = os.path.join(IVYSYN_PATH, "src/frameworks/pytorch-1.11-ivysyn/")
KERNELS_PATH = os.path.join(PYTORCH_PATH, "aten/src/ATen/native/")
STATS_PATH = os.path.join(IVYSYN_PATH, "results/pytorch/instrumentation/")
ATHERIS_COMMON_KERNELS = os.path.join(
    PYTORCH_IVYSYN_PATH, "one_to_one_kernels.txt")

build_include_path = glob.glob(os.path.join(
    PYTORCH_PATH + "build/lib.linux-x86_64*/torch/include"))[0]

clang_args = [
    "-I" + build_include_path,
    "-I" + PYTORCH_PATH + "aten/src",
    "-I" + PYTORCH_PATH + "aten/src/ATen",
    "-I" + PYTORCH_PATH + "aten/src/ATen/native",
    "-I" + PYTORCH_PATH + "aten/core",
    "-I" + PYTORCH_PATH + "torch/include",
    "-I" + PYTORCH_PATH + "c10/util",
]


class Parser:
    def __init__(self, filename, function_names, verbose=False, log_types=False, validate=False):
        if verbose:
            print(f"Parsing file {filename}")

        self.filename_abs = filename
        self.filename = filename.split("/")[-1]
        self.file_ext = filename.split(".")[-1]
        self.tmp_filename = filename + f".tmp.{self.file_ext}"
        self.function_names = function_names
        self.verbose = verbose
        self.validate = validate
        self.log_types = log_types
        self.fuzzed_argtypes = []
        self.unknown_types = {}
        self.created_wrappers = False

        self.index = Index.create()
        self.translation_unit = self.index.parse(
            self.filename_abs, args=clang_args)

        self.original_file = open(filename, "r").read().strip().split("\n")
        self.transformed_file = self.original_file

        self.transformed_funcs = []
        self.skipped_funcs = []

        self.cuda_funcs = []

        self.unknown_type_funcs = []
        self.failed_transform = []
        self.macro_funcs = []
        self.wrapped_funcs = []
        self.noarg_funcs = []
        self.out_funcs = []
        self.wrapper_funcs = []
        self.rnn = []
        self.inplace_unique = []
        self.inplace_dup = []
        self.structured = []

        self.helper_funcs = []
        self.indexing = []
        self.pointwise_ops = []
        self.binary_ops = []

        # Mappings between argument types and fuzzing functions
        self.fuzzer_typenames = {
            "int": "int",
            "int64_t": "long",
            "double": "double",
            "bool": "bool",
            "float": "double",
            "Tensor": "tensor",
            "at::Tensor": "tensor",
            "SparseTensor": "sparse_tensor",
            "at::sparse::SparseTensor": "sparse_tensor",
            "Scalar": "scalar",
            "c10::Scalar": "scalar",
            "TensorOptions": "tensor_options",
            "at::Scalar": "scalar",
            "ScalarType": "scalartype",
            "c10::ScalarType": "scalartype",
            "at::ScalarType": "scalartype",
            "std::string": "string",
            "at::IntArrayRef": "intarrayref",
            "c10::IntArrayRef": "intarrayref",
            "IntArrayRef": "intarrayref",
            "at::ArrayRef<double>": "doublearrayref",
            "ArrayRef<double>": "doublearrayref",
            "std::array<bool,3>": "boolarray",
            "std::array<bool,3ul>": "boolarray",
            "c10::optional<Tensor>": "c10opt_tensor",
            "optional<Tensor>": "c10opt_tensor",
            "optional<at::Tensor>": "c10opt_tensor",
            "c10::optional<IntArrayRef>": "c10opt_intarrayref",
            "optional<c10::IntArrayRef>": "c10opt_intarrayref",
            "optional<IntArrayRef>": "c10opt_intarrayref",
            "c10::optional<Scalar>": "c10opt_scalar",
            "optional<Scalar>": "c10opt_scalar",
            "optional<c10::Scalar>": "c10opt_scalar",
            "c10::optional<ScalarType>": "c10opt_scalartype",
            "optional<ScalarType>": "c10opt_scalartype",
            "optional<c10::ScalarType>": "c10opt_scalartype",
            "c10::optional<ArrayRef<double>>": "c10opt_doublearrayref",
            "c10::optional<c10::ArrayRef<double>>": "c10opt_doublearrayref",
            "optional<ArrayRef<double>>": "c10opt_doublearrayref",
            "c10::optional<int>": "c10opt_int",
            "optional<int>": "c10opt_int",
            "c10::optional<int64_t>": "c10opt_long",
            "optional<int64_t>": "c10opt_long",
            "c10::optional<double>": "c10opt_double",
            "optional<double>": "c10opt_double",
            "c10::optional<bool>": "c10opt_bool",
            "optional<bool>": "c10opt_bool",
            "c10::optional<std::string>": "c10opt_string",
            "optional<std::string>": "c10opt_string",
        }

    def transform_matching_funcs(self, node):
        """ Transform functions in the file which match one
        of the function names in the native function yaml file """

        # Found candidate function
        if (
            node.kind == CursorKind.FUNCTION_DECL
            and node.spelling in function_names
            and node.spelling not in self.transformed_funcs
            and node.spelling not in self.skipped_funcs
        ):

            source_file = node.location.file.name

            if source_file != self.filename_abs:
                return False

            # Skip RNN functions (too complicated)
            if self.filename == "RNN.cpp":
                self.rnn.append(node.spelling)
                self.skipped_funcs.append(node.spelling)
                return True

            # No arguments in function -- nothing to fuzz
            if len(list(node.get_arguments())) == 0:
                self.noarg_funcs.append(node.spelling)
                self.skipped_funcs.append(node.spelling)
                return True

            # Count cuda functions
            if "cuda" in node.spelling or "cudnn" in node.spelling:
                self.cuda_funcs.append(node.spelling)

            # Skip helper functions
            if "helper" in node.spelling:
                self.helper_funcs.append(node.spelling)
                self.skipped_funcs.append(node.spelling)
                return True

            # Skip out functions
            if (
                (len(node.spelling) > 4 and node.spelling[-4:] == "_out")
                or "_out_frame" in node.spelling
                or "_out_cpu" in node.spelling
                or "_out_cuda" in node.spelling
            ):
                self.out_funcs.append(node.spelling)
                self.skipped_funcs.append(node.spelling)
                return True

            if (
                node.spelling[-1] == "_"
                and node.spelling[-2] != "_"
                and all(x not in node.spelling for x in ["_out", "cuda", "cudnn"])
            ):
                if node.spelling[:-1] in self.function_names:
                    self.inplace_dup.append(node.spelling)
                else:
                    self.inplace_unique.append(node.spelling)
                self.skipped_funcs.append(node.spelling)
                return True

            # Get the types of the function arguments
            argtypes, _ = self.get_argument_names_and_types(node)

            self.fuzzed_argtypes = []
            # Make sure we can handle all the types
            all_known = True
            for i, argt in enumerate(argtypes):

                # Remove const keyword and references
                fuzz_type = argt.replace("const ", "").replace(
                    "&", "").replace(" ", "").strip()
                if fuzz_type not in self.fuzzer_typenames:

                    if fuzz_type in self.function_names:
                        self.macro_funcs.append(node.spelling)
                        self.skipped_funcs.append(node.spelling)

                    # Keep a dict to log how many times we encountered each
                    # unknown type
                    elif fuzz_type not in self.unknown_types:
                        # if fuzz_type not in self.unknown_types:
                        if self.verbose:
                            print(
                                f"Unknown type: {fuzz_type} in file {self.filename_abs}")
                            print(f"(Function {node.spelling})")
                        self.unknown_types[fuzz_type] = 1
                    else:
                        self.unknown_types[fuzz_type] += 1
                    all_known = False

                self.fuzzed_argtypes.append(fuzz_type)

            # Found a type we don't handle, don't transform this
            if not all_known:
                if node.spelling not in self.macro_funcs:
                    self.unknown_type_funcs.append(node.spelling)
                    self.skipped_funcs.append(node.spelling)
                if self.verbose:
                    print(argtypes)
                    print(
                        f"Function {node.spelling} in file {self.filename_abs} has" f" an unkonwn type, skipping")
                    print()
                return True

            # All checks passed, transform it and re-write the original file
            transformed = self.transform_function(node)
            if not transformed:
                if self.verbose:
                    print(f"Transformation for {node.spelling} failed")
                self.failed_transform.append(node.spelling)
                self.skipped_funcs.append(node.spelling)
                return True

            self.write_tmp_file()
            self.created_wrappers = True
            return True

        else:
            for c in node.get_children():
                source_file = c.location.file
                if source_file is not None:
                    source_file = source_file.name
                if source_file == self.filename_abs:
                    more_funcs = True
                    while more_funcs:
                        more_funcs = self.transform_matching_funcs(c)

        return False

    def get_line_toks(self, func, lineno):
        """ Extracts the words from a line """

        ret = " ".join([tok.spelling for tok in func.get_tokens()
                        if tok.location.line == lineno])

        # Remove comments in arguments
        ret = re.sub(r"/\*(?:[^*]|\*(?!/))*\*/", "", ret)

        # Remove the comments at the end of the line
        if "//" in ret:
            ret = ret[: ret.index("//")]

        return ret

    def get_argument_names_and_types(self, func):

        argnames = [x.spelling for x in func.get_arguments()]
        for i, x in enumerate(argnames):
            if len(x) == 0:
                argnames[i] = "dummyvar"

        argtypes = [x.type.spelling for x in func.get_arguments()]
        return argtypes, argnames

    def build_wrapper_func(self, func, argtypes, argnames):
        """ Create the function with the fuzzer code """

        lineno = func.location.line
        orig_sig = self.get_line_toks(func, lineno)

        if func.spelling not in orig_sig:
            return False

        # Find what type the function returns
        rettype = orig_sig[: orig_sig.index(func.spelling)]
        is_ref_ret = False
        # Function return type was in the previous line
        next_l = -1
        if rettype == "":
            rettype = self.get_line_toks(func, lineno - 1)
            # To also make it include the return type
            next_l -= 1
        rettype = rettype.replace(" ", "")

        if rettype[-1] == "&":
            # The function returns a reference
            is_ref_ret = True
            rettype = rettype[:-1]

        wrapper_func = []
        line = ""

        # Append the original signature of the function
        while "{" not in line:
            line = self.original_file[lineno + next_l]
            next_l += 1
            if "return " in line:
                line = line[: line.index("return ")]
            wrapper_func.append(line)

        # Handle arguments without names
        if "dummyvar" in argnames:
            idx = argnames.index("dummyvar")
            argt = argtypes[idx]
            if "bool" in argt:
                val = "true"
            elif "int" in argt:
                val = "0"
            elif "Tensor" in argt:
                val = "Tensor()"
            if argt == "IntArrayRef":
                val = "IntArrayRef({})"
            wrapper_func.append(f"\t{argt} dummyvar = {val};")

        # Create the arguments that will hold the fuzzed values
        fuzzed_argnames = []
        for i, argt in enumerate(argtypes):
            fuzz_type = self.fuzzed_argtypes[i]
            fuzz_arg = f"{argnames[i]}_fuzz"
            fuzzed_argnames.append(fuzz_arg)
            wrapper_func.append(f"\t{fuzz_type} {fuzz_arg};")

        # Call function with initial args first to avoid having bad seeds
        initial_funccall = f"do_{func.spelling}({', '.join(argnames)})"
        # wrapper_func.append("\n\t" + initial_funccall + ';')

        # Create the vector that holds the argument types represented
        # as strings, to be passed to the fuzzer
        types_as_str = ", ".join(
            '"' + "".join(argt.replace("const ", "").replace("&", "").split(" ")) + '"' for argt in argtypes
        )
        fuzzer_types_vec = f"\n\tstd::vector<std::string> types = {{{types_as_str}}};"
        wrapper_func.append(fuzzer_types_vec)

        # Store the arguments in a void * vector
        wrapper_func.append("\tstd::vector<void *> args{};\n")

        # Avoid nested fuzzing or re-fuzzing of the same function
        if not self.log_types and not self.validate:
            wrapper_func.append(
                f'\tif (!fuzzing::already_fuzzing && !fuzzing::was_fuzzed("{func.spelling}")) {{')

            # Mark that we are fuzzing to avoid nested fuzzing
            wrapper_func.append("\n\t\tfuzzing::already_fuzzing = true;\n")

        # Create the vector containing the original arguments
        arg_vector = []
        for i, argt in enumerate(self.fuzzed_argtypes):
            arg_vector.append(f"\t\targs.push_back((void *) &{argnames[i]});")

        # Initialize the fuzzer
        fuzzer_init = f'\n\t\tfuzzing::Fuzzer fuzzer = fuzzing::Fuzzer("{func.spelling}", types, args);\n'

        wrapper_func.append("\n".join(arg_vector))
        wrapper_func.append(fuzzer_init)

        if not self.log_types:
            # Keep fuzzing until we are out of mutations
            if self.validate:
                wrapper_func.append(
                    "\t\tif (fuzzer.should_validate()) {")
                for i, arg in enumerate(fuzzed_argnames):
                    fuzzed = f"\t\t\t{arg} = fuzzer.get_next_mut_" f"{self.fuzzer_typenames[self.fuzzed_argtypes[i]]}();"
                    wrapper_func.append(fuzzed)
                wrapper_func.append("\n\t\t\ttry {")
            else:
                wrapper_func.append(
                    "\t\twhile (fuzzer.has_more_mutations(true)) {")

                # Get the next mutation value for each argument
                for i, arg in enumerate(fuzzed_argnames):
                    fuzzed = f"\t\t\t{arg} = fuzzer.get_next_mut_" f"{self.fuzzer_typenames[self.fuzzed_argtypes[i]]}();"
                    wrapper_func.append(fuzzed)

                # To avoid exiting on exceptions
                wrapper_func.append("\n\t\t\ttry {")

                # For benchmarking
                wrapper_func.append("\n\t\t\t\tfuzzer.mut_start_time();")

            # Call the original function with the fuzzed arguments
            do_call = f"\t\t\t\tdo_{func.spelling}({', '.join(fuzzed_argnames)});"

            # Edge case (do_trapezoid and do_cumulative_trapezoid already exist)
            if func.spelling == "trapezoid" or func.spelling == "cumulative_trapezoid":
                do_call = do_call.replace("do_", "doo_")

            wrapper_func.append(do_call)
            if self.validate:
                wrapper_func.append("\t\t\t\tfuzzer.false_positive();\n")
                wrapper_func.append("\t\t\t} catch (...) {")
                wrapper_func.append("\t\t\t\tfuzzer.false_positive();\n")
                wrapper_func.append("\t\t\t}")
            else:
                wrapper_func.append("\t\t\t\tfuzzer.mut_end_time(false);\n")

                # # Ignore exceptions
                wrapper_func.append("\t\t\t} catch (...) {")
                wrapper_func.append("\t\t\t\tfuzzer.mut_end_time(true);")
                wrapper_func.append("\t\t\t} \n\t\t}")
                wrapper_func.append("\n\t\tfuzzing::already_fuzzing = false;")
            wrapper_func.append("\t}")

        if rettype != "void":
            wrapper_func.append(f"\n\treturn {initial_funccall};\n}}")
        else:
            wrapper_func.append("\n\treturn;\n}")

        # The original function that will be called on each iteration
        new_sig = self.transformed_file[lineno - 1].replace(
            func.spelling + "(", f"do_{func.spelling}(")

        if func.spelling == "trapezoid" or func.spelling == "cumulative_trapezoid":
            new_sig = new_sig.replace("do_", "doo_")

        self.transformed_file[lineno - 1] = new_sig
        self.wrapper_funcs.append(wrapper_func)
        return True

    def transform_function(self, func):
        """ Check if a function needs to be transformed and tansform it """

        argtypes, argnames = self.get_argument_names_and_types(func)
        if self.verbose:
            print(
                f"Found matching function: '{func.spelling}' with arguments:")
            print(*["".join(x + " " + y)
                    for x, y in zip(argtypes, argnames)], sep=", ")
            print(f"in file {self.filename_abs}")

        wrapper_created = self.build_wrapper_func(func, argtypes, argnames)
        if not wrapper_created:
            return False
        self.transformed_funcs.append(func.spelling)
        return True

    def insert_wrapper_funcs(self):
        """ Insert the wrapper functions containing the fuzzing code
        after the original functions """

        tr_funcs = self.transformed_funcs.copy()
        do_func_names = {x: f"do_{x}" for x in tr_funcs}
        if "trapezoid" in do_func_names:
            do_func_names["trapezoid"] = "doo_trapezoid"
        if "cumulative_trapezoid" in do_func_names:
            do_func_names["cumulative_trapezoid"] = "doo_cumulative_trapezoid"

        self.index = Index.create()
        self.translation_unit = self.index.parse(
            self.tmp_filename, args=clang_args)
        self.transformed_file = open(self.tmp_filename, "r").read().split("\n")
        nodes = self.translation_unit.cursor.walk_preorder()
        nodes = [
            x
            for x in nodes
            if x.location.file is not None
            and x.location.file.name == self.tmp_filename
            and x.kind == CursorKind.FUNCTION_DECL
        ]

        inserted_lines = 0
        for i, wfs in enumerate(tr_funcs):

            do_func_name = do_func_names[wfs]

            # Iterate until we find the original function
            found = False
            for node in nodes:
                if node.spelling == do_func_name:
                    func_end = node.extent.end.line

                    # Recursive edge case
                    if node.spelling == "do_matmul" and self.filename == "LinearAlgebra.cpp":
                        start = node.extent.start.line + inserted_lines
                        end = node.extent.end.line + inserted_lines
                        for ln in range(start, end + 1, 1):
                            self.transformed_file[ln] = self.transformed_file[ln].replace(
                                " matmul", " do_matmul")

                    if self.verbose:
                        print(f"Transformed function {node.spelling}\n")

                    self.transformed_file.insert(
                        func_end + inserted_lines, "\n".join(self.wrapper_funcs[i]))
                    inserted_lines += 1
                    found = True
                    break
            if not found:
                self.transformed_funcs.remove(wfs)
                print(f"Function {wfs} not found, not rewriting")

        self.write_transformed_file()
        if os.path.exists(self.tmp_filename):
            os.remove(self.tmp_filename)

    def print_transformed_file(self):
        print("\n".join(self.transformed_file))

    def write_tmp_file(self):
        with open(self.tmp_filename, "w") as r:
            r.write("\n".join(self.transformed_file))
            r.close()

    def write_transformed_file(self):
        """ Writes the file that was injected with the fuzzing code """
        with open(self.filename_abs, "w") as r:
            includes = [
                x for x in self.transformed_file if x.startswith("#include")]
            if "#include <ATen/ATen.h>" not in includes:
                self.transformed_file.insert(
                    0, "#include <ATen/core/fuzzing.h>")
            r.write("\n".join(self.transformed_file))
            r.close()


if __name__ == "__main__":

    parsed_files = []
    all_transformed = []
    all_skipped = set()

    all_cuda = set()
    all_unknown_type = set()
    all_macro_funcs = set()
    all_transform_failed = set()
    all_helper = set()
    all_wrapped = set()
    all_noarg = set()
    all_out = set()
    all_rnn = set()
    all_inplace_unique = set()
    all_inplace_dup = set()

    args_parser = argparse.ArgumentParser(
        description="Parse and transform Pytorch native files")
    args_parser.add_argument(
        "-v", "--verbose", dest="verbose", action="store_true", default=False, help="Print verbose info"
    )

    args_parser.add_argument(
        "--asan-extra",
        dest="asan_extra",
        action="store_true",
        default=False,
        help="Only modify extra asan crashing functions",
    )

    args_parser.add_argument(
        "--not-fuzzed-only",
        dest="not_fuzzed_only",
        action="store_true",
        default=False,
        help="Only modify non-fuzzed functions",
    )

    args_parser.add_argument(
        "--gpu", dest="instrument_gpu", action="store_true", default=False, help="Instrument GPU functions"
    )

    args_parser.add_argument(
        "--hip-only", dest="hip_only", action="store_true", default=False, help="Instrument only hip functions"
    )

    args_parser.add_argument(
        "--cuda-only", dest="cuda_only", action="store_true", default=False, help="Instrument only cuda functions"
    )

    args_parser.add_argument(
        "--types", dest="types", action="store_true", default=False, help="Instrument for type logging"
    )

    args_parser.add_argument(
        "--atheris-crashes", dest="atheris_crashes", action="store_true", default=False,
    )

    args_parser.add_argument(
        "--atheris-common", dest="atheris_common", action="store_true", default=False,
    )

    args_parser.add_argument("--validate", dest="validate",
                             action="store_true", default=False)

    args = args_parser.parse_args()

    # Need to use the hip modified libclang to parse .hip files, but native one for the rest
    if args.hip_only:
        Config.set_library_path("/opt/rocm-4.3.0/llvm/lib/")
    else:
        Config.set_library_path("/usr/lib/llvm-11/lib/")

    if args.asan_extra:
        with open("stats/asan_extra.txt", "r") as f:
            function_names = f.read().strip().split("\n")
    if args.atheris_crashes:
        with open("atheris_crashes.txt", "r") as f:
            function_names = f.read().strip().split("\n")
    elif args.atheris_common:
        with open(ATHERIS_COMMON_KERNELS, "r") as f:
            function_names = f.read().strip().split("\n")
    elif args.instrument_gpu:
        with open(os.path.join(PYTORCH_IVYSYN_PATH, "function_names_gpu.txt"), "r") as f:
            function_names = f.read().strip().split("\n")
    else:
        with open(os.path.join(PYTORCH_IVYSYN_PATH, "function_names.txt"), "r") as f:
            function_names = f.read().strip().split("\n")

    if args.not_fuzzed_only:
        with open("stats/cpu_fuzzed.txt", "r") as f:
            fuzzed = f.read().strip().split("\n")
        function_names = [x for x in function_names if x not in fuzzed]

    if args.hip_only:
        clang_args += ["-I" + "/opt/rocm/include",
                       "-I" + "/opt/rocm-4.3.0/hip/include"]

    basepath = os.path.join(PYTORCH_PATH, "aten/src/ATen/native/")

    files = glob.glob(KERNELS_PATH + "**/*.cpp", recursive=True)
    if args.instrument_gpu:
        files += glob.glob(KERNELS_PATH + "**/*.cu", recursive=True)

    # if args.instrument_gpu:
    #    if args.hip_only:
    #        files = glob.glob(KERNELS_PATH + "**/*.hip", recursive=True)
    #    elif args.cuda_only:
    #        files = glob.glob(KERNELS_PATH + "**/*.cu", recursive=True) + \
    #            glob.glob(KERNELS_PATH + "**/*.cpp", recursive=True)
    #    #files = [x for x in files if "hip/" in x]
    #    files = [x for x in files if "cuda/" in x or "cudnn/" in x]
    #    # if not args.hip_only:
    #    #     files += [KERNELS_PATH + "Activation.cpp"]
    # else:
    #    files = [x for x in files if not any(
    #        c in x for c in ["cuda/", "cudnn/", "hip/"])]

    # files = glob.glob(KERNELS_PATH + "LossCTC.cpp")
    files = [KERNELS_PATH + "/cuda/layer_norm_kernel.cu"]

    unknown_types = Counter()
    for filename in tqdm(files):

        if filename in parsed_files:
            continue

        parsed_files.append(filename)
        parser = Parser(filename, function_names, args.verbose,
                        args.types, args.validate)

        more_funcs = True
        while more_funcs:
            tu = parser.translation_unit
            more_funcs = parser.transform_matching_funcs(tu.cursor)

        if parser.created_wrappers:
            parser.insert_wrapper_funcs()
        unknown_types += Counter(parser.unknown_types)
        all_transformed.extend(parser.transformed_funcs)
        all_cuda.update(parser.cuda_funcs)

        all_unknown_type.update(
            {x for x in parser.unknown_type_funcs if x not in all_skipped})
        all_macro_funcs.update(
            {x for x in parser.macro_funcs if x not in all_skipped})
        all_transform_failed.update(
            {x for x in parser.failed_transform if x not in all_skipped})
        all_helper.update(
            {x for x in parser.helper_funcs if x not in all_skipped})
        all_noarg.update(
            {x for x in parser.noarg_funcs if x not in all_skipped})
        all_out.update({x for x in parser.out_funcs if x not in all_skipped})
        all_rnn.update({x for x in parser.rnn if x not in all_skipped})
        all_inplace_unique.update(
            {x for x in parser.inplace_unique if x not in all_skipped})
        all_inplace_dup.update(
            {x for x in parser.inplace_dup if x not in all_skipped})

        all_skipped.update(parser.skipped_funcs)

    print(f"Total unknown types: {len(unknown_types)}")
    for k, v in unknown_types.items():
        print(k, v)

    print("=" * 50)
    print(f"Total native functions read from file: {len(function_names)}")
    print(
        f"Transformed {len(all_transformed)} functions ({len(set(all_transformed))} unique)")
    print(f"{len(all_cuda)} cuda functions")
    print("-" * 50)

    all_transformed = set(all_transformed)
    all_skipped = {x for x in all_skipped if x not in all_transformed}

    # print(all_transformed)
    # print(all_skipped)

    all_unknown_type = {
        x for x in all_unknown_type if x not in all_transformed}
    all_transform_failed = {
        x for x in all_transform_failed if x not in all_transformed}
    all_macro_funcs = {x for x in all_macro_funcs if x not in all_transformed}
    all_helper = {x for x in all_helper if x not in all_transformed}
    all_noarg = {x for x in all_noarg if x not in all_transformed}
    all_out = {x for x in all_out if x not in all_transformed}
    all_rnn = {x for x in all_rnn if x not in all_transformed}
    all_inplace_unique = {
        x for x in all_inplace_unique if x not in all_transformed}
    all_inplace_dup = {x for x in all_inplace_dup if x not in all_transformed}

    print(f"Did not transform {len(all_skipped)} functions")
    print(f"{len(all_unknown_type)} functions with types we don't handle")
    print(f"{len(all_helper)} helper functions")
    print(f"{len(all_transform_failed)} functions where transformation failed")
    print(f"{len(all_macro_funcs)} macro functions")
    print(f"{len(all_inplace_unique)} unique inplace functions")
    print(f"{len(all_inplace_dup)} duplicate inplace functions")
    print(f"{len(all_noarg)} functions without arguments")
    print(f"{len(all_out)} out functions")
    print(f"{len(all_rnn)} RNN functions")

    print(len(set(all_transformed)) + len(all_skipped))
    not_found = [
        x for x in function_names if x not in all_transformed and x not in all_skipped]
    print(f"Did not find {len(not_found)} functions")

    out_prefix = "gpu_" if args.instrument_gpu else "cpu_"
    out_mode = "a+" if args.hip_only else "w"

    with open(STATS_PATH + f"{out_prefix}transformed.txt", out_mode) as f:
        f.write("\n".join(sorted(all_transformed)))

    with open(STATS_PATH + f"{out_prefix}skipped.txt", out_mode) as f:
        f.write("\n".join(sorted(all_skipped)))

    with open(STATS_PATH + f"{out_prefix}skipped_reasons.txt", out_mode) as f:
        f.write(f"Did not transform {len(all_skipped)} functions")
        f.write(f"{len(all_unknown_type)} functions with types we don't handle")
        f.write(f"{len(all_helper)} helper functions")
        f.write(
            f"{len(all_transform_failed)} functions where transformation failed")
        f.write(f"{len(all_macro_funcs)} macro functions")
        f.write(f"{len(all_inplace_unique)} unique inplace functions")
        f.write(f"{len(all_inplace_dup)} duplicate inplace functions")
        f.write(f"{len(all_noarg)} functions without arguments")
        f.write(f"{len(all_out)} out functions")
        f.write(f"{len(all_rnn)} RNN functions")

    # with open(STATS_PATH + f"{out_prefix}not_found.txt", out_mode) as f:
    #     f.write("\n".join(sorted(not_found)))
