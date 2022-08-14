
import argparse
import glob
import hashlib
import json
import os
import re
import sys

import yaml

IVYSYN_PATH = "/home/ivyusr/ivysyn/"
PYTORCH_PATH = os.path.join(IVYSYN_PATH, "src/frameworks/pytorch-1.11-ivysyn/")
sys.path.append(PYTORCH_PATH)

PYTORCH_IVYSYN_PATH = os.path.join(IVYSYN_PATH, "src/ivysyn/pytorch/")
RESULTS_PATH = os.path.join(IVYSYN_PATH, "results/pytorch/")
CRASHES_PATH_BASE = os.path.join(RESULTS_PATH, "crashes/")
REPRODUCE_PATH_BASE = os.path.join(RESULTS_PATH, "synthesized/")

try:
    from tools.autograd.gen_python_functions import load_signatures, load_signatures_synth, should_generate_py_binding
    from tools.codegen.gen import parse_native_yaml
    from tools.codegen.model import BaseTy, BaseType, ListType, OptionalType, Variant
except Exception as e:
    print(e)

native_functions_path = os.path.join(
    PYTORCH_IVYSYN_PATH, "native_functions.yaml")
depr_path = os.path.join(PYTORCH_PATH, "tools/autograd/deprecated.yaml")
derivatives_path = os.path.join(
    PYTORCH_PATH, "tools/autograd/derivatives.yaml")
autograd_dir = os.path.join(PYTORCH_PATH, "tools/autograd/")


CRASH_DELIM = "--------------------------------------\n"
SYNTH_IMPORTS = "import torch\n"
INIT_GPU = "torch.cuda.init()\n"
GPU_DEV = "gpu_dev = torch.device('cuda')\n"

dispatches = {}

dtypes = {'unsigned char': 'torch.uint8',
          'signed char': 'torch.int8',
          'short int': 'torch.int16',
          'int': 'torch.int32',
          'long': 'torch.int64',
          'long int': 'torch.int64',
          'c10::Half': 'torch.float16',
          'c10::BFloat16': 'torch.bfloat16',
          'float': 'torch.float32',
          'double': 'torch.float64',
          'c10::complex<float>': 'torch.cfloat',
          'c10::complex<double>': 'torch.cdouble',
          }


class TypeMismatch(Exception):
    pass


class BindingPairNotFound(Exception):
    pass


def parse_arg_type(label):
    global other_errs_list
    if label in ('Tensor', 'SparseTensor'):
        return BaseType(BaseTy.Tensor)
    elif label in ('Dimname',):
        return BaseType(BaseTy.Dimname)
    elif label in ('Generator',):
        return BaseType(BaseTy.Generator)
    elif label in ('Device',):
        return BaseType(BaseTy.Device)
    elif label in ('Stream',):
        return BaseType(BaseTy.Stream)
    elif label in ('Scalar',):
        return BaseType(BaseTy.Scalar)
    elif label in ('ScalarType',):
        return BaseType(BaseTy.ScalarType)
    elif label in ('Layout',):
        return BaseType(BaseTy.Layout)
    elif label in ('int64_t', 'long', 'int',):
        return BaseType(BaseTy.int)
    elif label in ('double',):
        return BaseType(BaseTy.float)
    elif label in ('bool',):
        return BaseType(BaseTy.bool)
    elif label in ('std::string', 'String',):
        return BaseType(BaseTy.str)
    elif re.match(r'TensorList\[?.*\]?', label):
        return ListType(BaseType(BaseTy.Tensor), size=None)
    elif re.match(r'IntArrayRef\[?.*\]?', label):
        return ListType(BaseType(BaseTy.int), size=None)
    elif re.match(r'DimnameList\[?.*\]?', label):
        return ListType(BaseType(BaseTy.Dimname), size=None)
    elif re.match(r'ScalarList\[?.*\]?', label):
        return ListType(BaseType(BaseTy.Scalar), size=None)
    elif label in ('std::array<bool,3>',):
        return ListType(BaseType(BaseTy.bool), size=None)
    elif label in ('Tensor?', 'OptionalTensor',):
        return OptionalType(BaseType(BaseTy.Tensor))
    elif label in ('int?', 'OptionalInt', 'long?', 'int64_t?', 'OptionalLong',):
        return OptionalType(BaseType(BaseTy.int))
    elif label in ('bool?', 'OptionalBool',):
        return OptionalType(BaseType(BaseTy.bool))
    elif label in ('double?',):
        return OptionalType(BaseType(BaseTy.float))
    elif label in ('Generator?',):
        return OptionalType(BaseType(BaseTy.Generator))
    elif label in ('Scalar?', 'OptionalScalar'):
        return OptionalType(BaseType(BaseTy.Scalar))
    elif label in ('ScalarType?', 'OptionalScalarType'):
        return OptionalType(BaseType(BaseTy.ScalarType))
    elif label in ('MemoryFormat?',):
        return OptionalType(BaseType(BaseTy.MemoryFormat))
    elif label in ('Layout?',):
        return OptionalType(BaseType(BaseTy.Layout))
    elif label in ('std::string?', 'OptionalString'):
        return OptionalType(BaseType(BaseTy.str))
    elif label in ('c10::List<c10::optional<Tensor>>',):
        return OptionalType(ListType(BaseType(BaseTy.Tensor), size=None))
    elif label in ('IntArrayRef?', 'OptionalIntArrayRef',):
        return OptionalType(ListType(BaseType(BaseTy.int), size=None))
    elif label in ('ArrayRef<double>?',):
        return OptionalType(ListType(BaseType(BaseTy.float), size=None))

    raise NotImplementedError(f"Unknown type {label}")


def get_python_pair_from_native_function(native_function_name, functions):

    py_pair = next(
        (x for x in functions if x.function.func.name.name.base == native_function_name), None)

    if py_pair is None:
        pre_dispatched = dispatches[native_function_name]
        py_pair = next(
            (x for x in functions if x.function.func.name.name.base == pre_dispatched), None)

    return py_pair


def get_interface(native_func):
    module = native_func.python_module

    if module is None:
        return 'torch.'

    interface = 'torch._C._'
    interface += module
    interface += '.'

    return interface


def extract_argtypes(str_arguments, is_python_sig=False):
    """Gets the types of the arguments from a function signature"""

    argtypes = []

    for str_arg in str_arguments:
        argtype = str_arg.split(' ')[0]
        if '\n' in argtype:
            argtype = argtype.split('\n')[0]
        argt = parse_arg_type(argtype)
        argtypes.append(argt)

    return argtypes


def parse_crash_args(crash, native_name, argnames, backward=False,
                     is_gpu=False, is_mkldnn=False, do_json=False,
                     validate=False):

    arguments = crash.split(';')[:-1]
    if len(arguments) == len(argnames):
        nnative_args_match = True
    else:
        nnative_args_match = False
        argnames = []

    # arguments = [x for x in arguments if not x.startswith('Duration')]
    argtypes = extract_argtypes(arguments)

    parsed_args = []
    total_arguments = 0

    for i, argt in enumerate(argtypes):
        arg = arguments[i]
        if nnative_args_match:
            argname = argnames[i]
        if argt == OptionalType(BaseType(BaseTy.Tensor)) and "nullopt" in arguments[i]:
            if not nnative_args_match:
                argname = f"opttensor_{total_arguments}"
                argnames.append(argname)
            if validate:
                parsed_arg = "opttensor"
            elif not do_json:
                parsed_arg = f"{argname} = None"
            parsed_args.append(parsed_arg)
        elif argt in (BaseType(BaseTy.Tensor), OptionalType(BaseType(BaseTy.Tensor))):
            tensor = arguments[i]
            contents_idx = tensor.index('Contents: ')
            sizes_idx = tensor.index('Sizes: ')
            device_idx = tensor.index('Device: ')
            if 'Dtype' in tensor:
                dtype_idx = tensor.index('Dtype: ')
            else:
                dtype_idx = None
            grad_idx = tensor.index('Requires grad: ')
            contents = tensor[contents_idx + len('Contents: '): sizes_idx]
            sz_end_idx = dtype_idx if dtype_idx is not None else device_idx
            sizes = tensor[sizes_idx + len('Sizes: '): sz_end_idx].strip()
            if dtype_idx is not None:
                dtype = dtypes[tensor[dtype_idx +
                                      len('Dtype: '): device_idx].strip()]
            else:
                dtype = 'torch.float64'
            device = tensor[device_idx + len('Device: '): grad_idx]
            req_grad = tensor[grad_idx + len('Requires grad: '):]

            # Tensors that will be used as input for backward() require grad
            if (backward or req_grad == '1') and dtype.startswith('torch.float'):
                req_grad = 'True'
            else:
                req_grad = 'False'

            if not nnative_args_match:
                argname = f"tensor_{total_arguments}"
                argnames.append(argname)

            if 'empty' in contents:
                if validate:
                    parsed_arg = f"tensor\n{dtype}\n\n{sizes.replace(' ', '').rstrip(',')}\nFalse"
                elif not do_json:
                    parsed_arg = f"{argname} = torch.empty(({sizes}), dtype={dtype})"
                else:
                    parsed_arg = {"argname": argname, "main_type": "tensor",
                                  "secondary_type": dtype, "to_mkldnn": False}
            elif len(contents) > 1:
                value = contents.strip()
                if validate:
                    parsed_arg = f"tensor\n{dtype}\n{value}\n{sizes.replace(' ', '').rstrip(',')}\n{req_grad}"
                elif not do_json:
                    parsed_arg = f"{argname} = torch.full(({sizes}), "
                    parsed_arg += f"{value}, dtype={dtype}, requires_grad={req_grad}"
                    if is_gpu:
                        parsed_arg += f", device=gpu_dev"
                    parsed_arg += ")"
                else:
                    parsed_arg = {"argname": argname, "main_type": "tensor",
                                  "secondary_type": dtype, "to_mkldnn": False}
            else:
                if not do_json:
                    parsed_arg = f"{argname} = torch.tensor(dtype={dtype}, requires_grad={req_grad})"
                else:
                    parsed_arg = {"argname": argname, "main_type": "tensor",
                                  "secondary_type": dtype, "to_mkldnn": False}

            if is_mkldnn and dtype in ('torch.float32', 'torch.bfloat16') and not validate:
                if not do_json:
                    parsed_arg = parsed_arg + ".to_mkldnn()"
                else:
                    parsed_arg["to_mkldnn"] = True
            parsed_args.append(parsed_arg)

        elif argt in (BaseType(BaseTy.Scalar), OptionalType(BaseType(BaseTy.Scalar))):
            contents = arg[arg.index(' ') + 1:]
            if not nnative_args_match:
                argname = f"scalar_{total_arguments}"
                argnames.append(argname)
            if validate:
                if '.' in contents or 'e' in contents:
                    scalar_type = 'float'
                else:
                    scalar_type = 'int'
                parsed_arg = f"scalar\n{scalar_type}\n{contents}"
                parsed_args.append(parsed_arg)
            elif not do_json:
                parsed_args.append(f"{argname} = torch.tensor({contents})")
            else:
                if 'e' in contents or '.' in contents:
                    dtype = 'torch.float64'
                else:
                    dtype = 'torch.int64'
                parsed_args.append({"argname": argname,
                                    "main_type": "scalar",
                                    "secondary_type": dtype})

        elif argt in (ListType(BaseType(BaseTy.int), size=None), OptionalType(ListType(BaseType(BaseTy.int), size=None))):
            contents = arg[arg.index('['):arg.index(']') + 1]
            if not nnative_args_match:
                argname = f"intarrayref_{total_arguments}"
                argnames.append(argname)
            if validate:
                contents = contents.replace(' ', '').rstrip(
                    ',').replace('[', '').replace(']', '')
                parsed_arg = f"intarray\n{contents}"
                parsed_args.append(parsed_arg)
            elif not do_json:
                parsed_args.append(f"{argname} = " + contents)
            else:
                parsed_args.append({"argname": argname,
                                    "main_type": "intarray",
                                    "secondary_type": None})

        elif argt in (BaseType(BaseTy.int), OptionalType(BaseType(BaseTy.int))):
            contents = arg[arg.index(' ') + 1:]
            if not nnative_args_match:
                argname = f"int_{total_arguments}"
                argnames.append(argname)
            if validate:
                parsed_arg = f"int\n{contents}"
                parsed_args.append(parsed_arg)
            elif not do_json:
                parsed_args.append(f"{argname} = " + contents)
            else:
                parsed_args.append({"argname": argname,
                                    "main_type": "int",
                                    "secondary_type": None})

        elif argt in (BaseType(BaseTy.float), OptionalType(BaseType(BaseTy.float))):
            contents = arg[arg.index(' ') + 1:]
            if not nnative_args_match:
                argname = f"double_{total_arguments}"
                argnames.append(argname)
            if validate:
                parsed_arg = f"double\n{contents}"
                parsed_args.append(parsed_arg)
            elif not do_json:
                parsed_args.append(f"{argname} = " + contents)
            else:
                parsed_args.append({"argname": argname,
                                    "main_type": "float",
                                    "secondary_type": None})

        elif argt in (BaseType(BaseTy.str), OptionalType(BaseType(BaseTy.str))):
            contents = '"' + arg[arg.index(' ') + 1:] + '"'
            if not nnative_args_match:
                argname = f"string_{total_arguments}"
                argnames.append(argname)
            if validate:
                parsed_arg = f"string\n{contents}"
                parsed_args.append(parsed_arg)
            elif not do_json:
                parsed_args.append(f"{argname} = " + contents)
            else:
                parsed_args.append({"argname": argname,
                                    "main_type": "str",
                                    "secondary_type": None})

        elif argt in (BaseType(BaseTy.bool), OptionalType(BaseType(BaseTy.bool))):
            contents = arg[arg.index(' ') + 1:]
            if not nnative_args_match:
                argname = f"bool_{total_arguments}"
                argnames.append(argname)
            if validate:
                parsed_arg = f"bool\n{contents}"
                parsed_args.append(parsed_arg)
            elif not do_json:
                if contents == '0':
                    parsed_args.append(f"{argname} = False")
                else:
                    parsed_args.append(f"{argname} = True")
            else:
                parsed_args.append({"argname": argname,
                                    "main_type": "bool",
                                    "secondary_type": None})

        elif argt in (ListType(BaseType(BaseTy.bool), size=None), OptionalType(ListType(BaseType(BaseTy.bool), size=None))):
            contents = arg[arg.index(' ') + 1:]
            if not nnative_args_match:
                argname = f"boolarr_{total_arguments}"
                argnames.append(argname)
            barr = []
            for c in contents:
                if c == '0':
                    barr.append(False)
                else:
                    barr.append(True)
            if validate:
                parsed_arg = f"boolarray\n{repr(barr)}"
                parsed_args.append(parsed_arg)
            elif not do_json:
                parsed_args.append(f"{argname} = {repr(barr)}")
            else:
                parsed_args.append({"argname": argname,
                                    "main_type": "boolarray",
                                    "secondary_type": None})

        else:
            raise NotImplementedError(
                f"Support for type {argt} not implemented")

        total_arguments += 1

    return argtypes, parsed_args, argnames


def has_self_arg(native_func):
    self_arg = native_func.func.arguments.self_arg
    return self_arg is not None


def has_preself_arg(native_func):
    preself_arg = native_func.func.arguments.pre_self_positional
    return preself_arg is not None


def has_out_arg(native_func):
    out_arg = native_func.func.arguments.out
    return len(out_arg) > 0


def get_self_arg_name_and_type(native_func):

    if has_self_arg(native_func):
        self_arg = native_func.func.arguments.self_arg.argument
        return [self_arg.name], [self_arg.type]

    return [], []


def get_preself_arg_names_and_types(native_func):

    if has_preself_arg(native_func):
        preself_args = native_func.func.arguments.pre_self_positional
        names = [x.name for x in preself_args]
        types = [x.type for x in preself_args]
        return names, types

    return [], []


def get_out_arg_name_and_type(native_func):

    if has_out_arg(native_func):
        out_arg = native_func.func.arguments.out[0]
        return [out_arg.name], [out_arg.type]

    return [], []


def get_native_positional_args_names_and_types(native_func):
    positional_args = native_func.func.arguments.post_self_positional
    names = [x.name for x in positional_args]
    types = [x.type for x in positional_args]
    return names, types


def get_python_positional_args_names_and_types(python_func):
    positional_args = python_func.input_args
    names = [x.name for x in positional_args]
    types = [x.type for x in positional_args]
    return names, types


def get_native_kwarg_only_names_and_types(native_func):
    kwarg_only = native_func.func.arguments.pre_tensor_options_kwarg_only
    kwarg_only += native_func.func.arguments.post_tensor_options_kwarg_only
    names = [x.name for x in kwarg_only]
    types = [x.type for x in kwarg_only]
    return names, types


def get_python_kwarg_only_names_and_types(python_func):
    kwarg_only = python_func.input_kwargs
    names = [x.name for x in kwarg_only]
    types = [x.type for x in kwarg_only]
    return names, types


def crashing_types_match(crashing_types, native_types):

    if len(crashing_types) != len(native_types):
        return False

    for crtype, nttype in zip(crashing_types, native_types):

        if isinstance(crtype, ListType):
            if isinstance(nttype, ListType):
                if crtype.elem != nttype.elem:
                    return False
            else:
                return False

    return True


def extract_arg_sequence(lamda_exprs):

    seq = []

    for expr in lamda_exprs.exprs:
        pos = expr[expr.find('(') + 1: expr.find(')')]
        seq.append(int(pos))

    return seq


def create_python_func_call_args(python_pos_names, python_kwarg_names, native_arg_names, main_argnames):

    python_call_args = []

    # print(python_pos_names + python_kwarg_names)
    # print(native_arg_names)
    # print(main_argnames)

    for pos_arg_name in python_pos_names:
        if pos_arg_name in ('grad_output', ):
            continue
        native_arg_idx = native_arg_names.index(pos_arg_name)
        crashing_arg = main_argnames[native_arg_idx]
        python_call_args.append(crashing_arg)

    for kwarg_name in python_kwarg_names:
        if pos_arg_name in ('grad_output', ):
            continue
        native_arg_idx = native_arg_names.index(kwarg_name)
        crashing_arg = main_argnames[native_arg_idx]
        python_call_args.append(f"{kwarg_name}={crashing_arg}")

    return python_call_args


def synthesize_file_validate(crash, native_name, is_gpu=False, is_mkldnn=False):

    is_mkldnn = False
    backward = False

    if '_backward' in native_name:
        backward = True

    if 'mkldnn' in native_name:
        is_mkldnn = True

    crashing_arg_types, synthesized_crashing_args, argnames = parse_crash_args(
        crash, native_name, [], backward, is_gpu,
        is_mkldnn, False, True)

    return '\n'.join(synthesized_crashing_args)


def synthesize_file(crash, native_name, functions, derivatives,
                    is_gpu=False, types=False, do_json=False):

    if do_json:
        synthesized_file = dict()
    else:
        synthesized_file = []

    is_mkldnn = False

    backward = False
    if '_backward' in native_name:
        backward = True

    if 'mkldnn' in native_name:
        is_mkldnn = True

    binding_python_pair = get_python_pair_from_native_function(
        native_name, functions)
    # print(binding_python_pair)

    forward_python_pair = None
    if backward:
        forward_name = derivatives[binding_python_pair.signature.name]
        if '.' in forward_name:
            forward_name = forward_name[:forward_name.index('.')]
        forward_python_pair = get_python_pair_from_native_function(
            forward_name, functions)
    else:
        forward_python_pair = binding_python_pair

    # print(forward_python_pair)
    # print(binding_python_pair)

    # Look for binding without _backward
    if binding_python_pair is None and forward_python_pair is not None:
        native_name = native_name.replace('_backward', '')
        forward_python_pair = get_python_pair_from_native_function(
            native_name, functions)

    if forward_python_pair is None:
        print(f"Binding not found for {native_name}")
        raise BindingPairNotFound(f"Pair not found for {native_name}")
        return None
    if binding_python_pair is None:
        print(f"Binding not found for {native_name}")
        raise BindingPairNotFound(f"Binding pair not found for {native_name}")
        return None

    python_func = forward_python_pair.signature
    native_func = forward_python_pair.function
    binding_python_func = binding_python_pair.signature
    binding_native_func = binding_python_pair.function

    # print(native_func)
    # print(python_func)
    # print(binding_native_func)
    # print(binding_python_func)

    if native_func.manual_cpp_binding:
        print(f"Manual cpp binding for {native_name}")
        return None

    native_preself_names, native_preself_types = get_preself_arg_names_and_types(
        binding_native_func)
    native_self_name, native_self_type = get_self_arg_name_and_type(
        binding_native_func)
    native_pos_names, native_pos_types = get_native_positional_args_names_and_types(
        binding_native_func)
    native_kwarg_names, native_kwarg_types = get_native_kwarg_only_names_and_types(
        binding_native_func)
    native_out_name, native_out_type = get_out_arg_name_and_type(
        binding_native_func)

    python_pos_names, python_pos_types = get_python_positional_args_names_and_types(
        binding_python_func)
    python_kwarg_names, python_kwarg_types = get_python_kwarg_only_names_and_types(
        binding_python_func)

    native_arg_names = native_preself_names + native_self_name + \
        native_pos_names + native_kwarg_names + native_out_name
    native_arg_types = native_preself_types + native_self_type + \
        native_pos_types + native_kwarg_types + native_out_type

    crashing_arg_types, synthesized_crashing_args, argnames = parse_crash_args(
        crash, native_name, native_arg_names, backward, is_gpu,
        is_mkldnn, do_json, False)

    # print(crashing_arg_types)
    # print(synthesized_crashing_args)
    # print(argnames)

    # print(python_pos_names + python_kwarg_names)
    # print(native_arg_names)
    # print(argnames)

    main_crashing_arg_types = crashing_arg_types
    main_crashing_args = synthesized_crashing_args
    main_argnames = argnames

    if not do_json:
        if types:
            synthesized_file.append("import os")
            synthesized_file.append("import glob")

        synthesized_file.append(SYNTH_IMPORTS)

        if is_gpu:
            synthesized_file.append(INIT_GPU)
            synthesized_file.append(GPU_DEV)

    if not do_json:
        for synth_arg in synthesized_crashing_args:
            synthesized_file.append(synth_arg)
    else:
        synthesized_file["args"] = synthesized_crashing_args

    ret_api_name = python_func.name

    native_self_name, native_self_type = get_self_arg_name_and_type(
        native_func)
    native_pos_names, native_pos_types = get_native_positional_args_names_and_types(
        native_func)
    native_kwarg_names, native_kwarg_types = get_native_kwarg_only_names_and_types(
        native_func)
    native_out_name, native_out_type = get_out_arg_name_and_type(native_func)
    python_pos_names, python_pos_types = get_python_positional_args_names_and_types(
        python_func)
    python_kwarg_names, python_kwarg_types = get_python_kwarg_only_names_and_types(
        python_func)

    interface = get_interface(native_func)
    try:
        py_func_call_args = create_python_func_call_args(
            python_pos_names, python_kwarg_names, native_arg_names, main_argnames)
    except ValueError as e:
        print(e)
        print(f"Matching arg name not found for {native_name}")
        return None

    is_method = any([x == Variant.method for x in native_func.variants])

    if not is_method:
        py_func_call_forward = interface + python_func.name + '('
        py_func_call_forward += ', '.join(py_func_call_args).strip(', ')
        py_func_call_forward += ')'
    else:
        if 'self' in py_func_call_args:
            self_arg = 'self'
            py_func_call_args.remove('self')
            rest = py_func_call_args
        else:
            self_arg = py_func_call_args[0]
            rest = py_func_call_args[1:]
        py_func_call_forward = self_arg + \
            '.' + python_func.name + '('
        py_func_call_forward += ', '.join(rest).strip(', ')
        py_func_call_forward += ')'

    if backward:
        py_func_call_forward = "res = " + py_func_call_forward

    if not do_json:
        # if types:
        #     synthesized_file.append(
        #         "for filename in glob.glob('/mnt/pytorch-ivysyn/*'):")
        # synthesized_file.append("\tos.remove(filename)")

        synthesized_file.append(py_func_call_forward)
    else:
        synthesized_file["binding_call"] = py_func_call_forward

    if backward:
        synthesized_file.append("grad_out = torch.zeros_like(res)")
        synthesized_file.append(
            "torch.autograd.backward(res, grad_tensors=grad_out)")

    if backward:
        ret_api_name += '_backward'
    if not do_json:
        return '\n'.join(synthesized_file), ret_api_name
    else:
        return synthesized_file, ret_api_name


def get_der_value(kw, entry):

    return [value for key, value in entry.items() if kw in key][0]


def parse_derivatives():

    derivatives = {}

    with open(derivatives_path, 'r') as f:
        derivatives_yaml = yaml.safe_load(f)

    for der in derivatives_yaml:
        vals = []
        if any('name' in x for x in der.keys()):
            if any('self' in x for x in der.keys()):
                vals.append(get_der_value('self', der))
            if any('input' in x for x in der.keys()):
                vals.append(get_der_value('input', der))
            if any('weight' in x for x in der.keys()):
                vals.append(get_der_value('weight', der))
            if any('log_probs' in x for x in der.keys()):
                vals.append(get_der_value('log_probs', der))
            if any('per_sample_weights' in x for x in der.keys()):
                vals.append(get_der_value('per_sample_weights', der))
            if len(vals) == 0:
                continue
            frw = get_der_value('name', der)
            frw = frw.split('(')[0]
            for val in vals:
                for v in val.split(' '):
                    if 'backward' in val:
                        bkw = v.split('(')[0]
                        derivatives[bkw] = frw

    return derivatives


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser(
        description="Synthesize pytorch crashes")

    arg_group = args_parser.add_mutually_exclusive_group(required=True)

    arg_group.add_argument(
        "--dir", dest="dir"
    )

    arg_group.add_argument(
        "--types", dest="types", action="store_true", default=False,
    )

    args_parser.add_argument(
        "--json", dest="json", action="store_true", default=False, help="Synthesize as json file"
    )

    args_parser.add_argument(
        "--extra-asan", dest="extra_asan", action="store_true", default=False, help="Synthesize extra asan crash files"
    )

    args_parser.add_argument(
        "--gpu", dest="gpu", action="store_true", default=False, help="Synthesize gpu crashes"
    )

    args_parser.add_argument("--validate", dest="validate",
                             action="store_true", default=False)

    args = args_parser.parse_args()

    if args.types:
        native_functions = parse_native_yaml(
            native_functions_path).native_functions
        native_functions = list(
            filter(should_generate_py_binding, native_functions))
        functions = load_signatures(
            native_functions, depr_path, method=False)
    else:
        functions = load_signatures_synth(
            native_functions_path, depr_path, method=False)

    derivatives = parse_derivatives()

    with open(os.path.join(PYTORCH_IVYSYN_PATH, "dispatches.txt"), "r") as f:
        disps = f.read().strip().split("\n")

    for disp in disps:
        disp = disp.split(" ")
        dispatches[disp[0]] = disp[1]

    reproduce_path = REPRODUCE_PATH_BASE

    if args.gpu:
        reproduce_path += "gpu/all/"
        crashes_path = CRASHES_PATH_BASE + args.dir + "/"
    elif args.types:
        crashes_path = os.path.join(
            RESULTS_PATH, "atheris_comp/kernel_types/logged_types/")
        if not args.json:
            reproduce_path = os.path.join(
                RESULTS_PATH, "atheris_comp/kernel_types/synthed_ops/")
        else:
            reproduce_path = os.path.join(
                RESULTS_PATH, "atheris_comp/kernel_types/synthed_json/")
    elif args.validate:
        reproduce_path += args.dir + "/validate/"
        crashes_path = CRASHES_PATH_BASE + args.dir + "/"
    else:
        reproduce_path += args.dir + "/all/"
        crashes_path = CRASHES_PATH_BASE + args.dir + "/"

    if not os.path.isdir(reproduce_path):
        os.makedirs(reproduce_path)

    ext = ".types" if args.types else "_crashes.log"

    successful = set()
    errors = set()
    for crash_filename in glob.glob(crashes_path + '*' + ext):

        crash_fname = crash_filename.replace(crashes_path, '')
        native_name = crash_fname.replace(ext, '')

        # if native_name != 'ctc_loss':
        #     continue

        with open(crash_filename, 'r') as crash_file:
            crashes = crash_file.read().strip()

        if crashes == '':
            continue

        # print(crashes)

        crashes = crashes.split(CRASH_DELIM)

        for crash in crashes:
            try:
                if not args.validate:
                    ret = synthesize_file(crash, native_name,
                                          functions, derivatives,
                                          args.gpu, args.types,
                                          args.json)
                    if ret is None:
                        errors.add(native_name)
                        continue
                    if not args.json:
                        synth_file, api_name = ret
                    else:
                        synth_file = ret[0]
                        api_name = ret[1]
                    out_filename = reproduce_path + api_name
                else:
                    synth_file = synthesize_file_validate(
                        crash, native_name, args.gpu)
                    out_filename = reproduce_path + native_name
                if not args.types and not args.validate:
                    filehash = hashlib.md5(synth_file.encode()).hexdigest()
                    out_filename += '_' + filehash
                if args.validate:
                    out_filename += '.validate'
                elif not args.json:
                    out_filename += '.py'
                else:
                    out_filename += '.json'
                if not os.path.isfile(out_filename):
                    with open(out_filename, 'w') as f:
                        if not args.json:
                            if not args.validate:
                                f.write('# ' + native_name + '\n')
                            # f.write('# ' + run_date + '\n')
                            f.write(synth_file + '\n')
                        else:
                            f.write(json.dumps(synth_file) + '\n')
                if not args.validate:
                    successful.add(api_name)
                else:
                    successful.add(native_name)
            except Exception as e:
                print(e)
                errors.add(native_name)
                print(f"Error {native_name}")
                # print(native_name)
                # print('=' * 50)

    print(f"{len(successful)} successfully synthesized, {len(errors)} errors")
    # if args.dir:
    #     with open(os.path.join(RESULTS_PATH, args.dir + "/synth_errors.txt"), "w") as f:
    #         f.write('\n'.join(errors))
    # print('\n'.join(errors))
