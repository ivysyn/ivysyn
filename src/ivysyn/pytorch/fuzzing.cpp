
#include "fuzzing.h"
#include <ATen/core/fuzzing.h>

/* #define IVYSYN_VALIDATE */

namespace fuzzing {

  const char* results_dir = "/mnt/pytorch-ivysyn";

#if !defined(IVYSYN_VALIDATE) && !defined(IVYSYN_COLLECT_TYPES)
  bool already_fuzzing = false;
  const int TIMEOUT_SECS = 1200;
  const int RAND_SEED = 123;
  const int NMUT_UPPER_BOUND_MID = 1000000;
  const int CRASHES_BOUND = 1;
  const int MUTFILE_TRIES = 5;
  const int MAX_EMPTY_LOG_FILES = 5;
  const int TIME_THRESH_SECS = 30;
  const at::DeviceType tensor_dev = c10::kCPU;
  std::string cur_fname_glob = {};

  static std::fstream mutations_file;
  static std::fstream mutations_restore;
  static std::fstream last_timestamp_file;
  static std::fstream timestamp_restore;
  static std::fstream crashes_file;
  static std::fstream unknown_type_file;
  static std::fstream num_crashes_file;
  static std::fstream time_file;
  static std::fstream except_file;
  static std::fstream start_file;
  static std::fstream done_file;
  static std::fstream crash_found_file;
  static std::fstream overflow_file;
  static std::fstream nofuzz_file;
  static std::fstream zero_muts_file;

  static struct timespec start_time;
  static struct timespec end_time;
#endif

#if defined(IVYSYN_COLLECT_TYPES)
  static std::fstream types_file;
#endif

  void create_file(const std::string& filename, std::fstream &file, std::ios_base::openmode fflags)
  {
      std::ofstream file_stream(filename);

      /* std::cout << "Creating file " << filename << std::endl; */

      if (file.is_open()) {
        file.close();
      }

      file.clear();
      file.open(filename, fflags);
      if (file.fail()) {
        std::cout << "Failed to open " << filename << ":" << strerror(errno) << std::endl;
      }

      /* std::cout << "Created file " << filename << std::endl; */
  }

#if !defined(IVYSYN_VALIDATE) && !defined(IVYSYN_COLLECT_TYPES)
  struct timespec time_diff(struct timespec start, struct timespec end)
  {
    struct timespec res;
    if ((end.tv_nsec - start.tv_nsec) < 0) {
      res.tv_sec = end.tv_sec - start.tv_sec - 1;
      res.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
    }
    else {
      res.tv_sec = end.tv_sec - start.tv_sec;
      res.tv_nsec = end.tv_nsec - start.tv_nsec;
    }
    return res;
  }


  bool was_fuzzed(const std::string& fname)
  {
    struct stat stat_buffer = {};
    std::string filename = std::string(results_dir) + "/" + fname + ".done";
    return stat(filename.c_str(), &stat_buffer) == 0;
  }

  bool zero_muts_crashed(const std::string& fname) {
    struct stat stat_buffer = {};
    std::string zero_mut_filename;

    zero_mut_filename = std::string(results_dir) + "/" + fname + ".zero_muts";

    return stat(zero_mut_filename.c_str(), &stat_buffer) == 0;
  }

  bool was_killed(const std::string& fname)
  {
    struct stat stat_buffer = {};
    int killed_exists = 0, timeout_exists = 0;

    std::string killed_filename = std::string(results_dir) + "/" + fname + ".killed";
    std::string timeout_filename = std::string(results_dir) + "/" + fname + ".timeout";

    killed_exists = stat(killed_filename.c_str(), &stat_buffer) == 0;

    stat_buffer = {};
    timeout_exists = stat(timeout_filename.c_str(), &stat_buffer) == 0;

    return killed_exists | timeout_exists;
  }

  void handle_timeout(int)
  {

    /* std::cout << "Kernel " << cur_fname_glob << " timed out, stopping fuzzing" << std::endl; */

    struct timespec ts = {};
    clock_gettime(CLOCK_MONOTONIC, &ts);

    std::string timeout_filename = std::string(results_dir) + "/" + cur_fname_glob + ".timeout";

    /* std::ofstream timeout_file(timeout_filename); */
    /* timeout_file << ts.tv_sec << std::endl; */
    /* timeout_file.close(); */

    int fd = open(timeout_filename.c_str(), O_WRONLY | O_CREAT, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    char logbuf[0x20];
    memset(logbuf, 0, 0x20);
    sprintf(logbuf, "%ld", ts.tv_sec);
    write(fd, logbuf, 0x20);
    close(fd);

    _Exit(-SIGALRM);
  }

#endif

Fuzzer::~Fuzzer() {

  /* if (mutations_file.is_open()) { */
  /*     mutations_file.close(); */
  /* } */
  /* if (mutations_restore.is_open()) { */
  /*     mutations_restore.close(); */
  /* } */
  /* if (crashes_file.is_open()) { */
  /*     crashes_file.close(); */
  /* } */
  /* if (num_crashes_file.is_open()) { */
  /*     num_crashes_file.close(); */
  /* } */

  alarm(0);

}


#if defined(IVYSYN_COLLECT_TYPES)
Fuzzer::Fuzzer(char *fname, std::vector<std::string> types_vec, std::vector<void *> args)
  : cur_fname(fname) {

    struct stat stat_buffer = {};
    std::string out_str;
    std::string types_filename;

    types_filename = std::string(results_dir) + "/" + fname + ".types";

    int exists = 0;
    exists = stat(types_filename.c_str(), &stat_buffer) == 0;

    if (exists) {
      return;
    }

    std::ios_base::openmode fflags = std::ios::out | std::ios::in | std::ios::trunc;

    create_file(types_filename, types_file, fflags);

    at::Tensor tensor;
    at::Tensor sparse_tensor;
    at::TensorOptions tensor_opts;
    bool boolean;
    bool a,b,c;
    int integer;
    double doublenum;
    int64_t longint;
    at::IntArrayRef intarrayref;
    at::Scalar scalar;
    at::ScalarType scalartype;
    at::ArrayRef<double> doublearrayref;
    std::array<bool,3> boolarray;
    std::string string;
    c10::optional<at::Tensor> tensor_opt;
    c10::optional<at::IntArrayRef> intarrayref_opt;
    c10::optional<at::Scalar> scalar_opt;
    c10::optional<at::ScalarType> scalartype_opt;
    c10::optional<at::ArrayRef<double>> doublearrayref_opt;
    c10::optional<int> int_opt;
    c10::optional<int64_t> long_opt;
    c10::optional<double> double_opt;
    c10::optional<bool> bool_opt;
    c10::optional<std::string> string_opt;
    int total_args, i;
    fuzzing::TorchType type_enum;
    std::string type;
    void *arg;

    at::TensorOptions default_opts = c10::TensorOptions().dtype(c10::kDouble).layout(c10::kStrided).device(tensor_dev);
    total_args = types_vec.size();

    for (int i = 0; i < total_args; i++) {

      type = types_vec.at(i);

      auto it = map_str_enum.find(type);
      if (it != map_str_enum.end()) {
        indices.push_back(0);
        type_enum = it->second;
      } else {
        mark_unknown_type(type);
      }

      arg = args.at(i);
      switch (type_enum) {
        case fuzzing::FUZZ_INT:
          integer = *(int*) arg;
          types_file << "int " << integer << ";";
          break;
        case fuzzing::FUZZ_LONG:
          longint = *(long*) arg;
          types_file << "int64_t " << longint << ";";
          break;
        case fuzzing::FUZZ_FLOAT:
        case fuzzing::FUZZ_DOUBLE:
          doublenum = *(double*) arg;
          types_file << "double " << doublenum << ";";
          break;
        case fuzzing::FUZZ_BOOLEAN:
          boolean = *(bool*) arg;
          types_file << "bool " << boolean << ";";
          break;
        case fuzzing::FUZZ_SCALAR:
          scalar = *(at::Scalar*) arg;
          if (scalar.isFloatingPoint()) {
            types_file << "Scalar " << scalar.to<double>() << ";";
          }
          else if (scalar.isIntegral(false)) {
            types_file << "Scalar " << scalar.to<int64_t>() << ";";
          } else if (scalar.isBoolean()) {
            types_file << "Scalar " << scalar.to<bool>() << ";";
          }
          break;
        case fuzzing::FUZZ_SCALARTYPE:
          scalartype = *(at::ScalarType*) arg;
          types_file << "ScalarType " << scalartype << ";";
          break;
        case fuzzing::FUZZ_TENSOR:
          tensor = *(at::Tensor*) arg;
          if  (!tensor.defined()) {
            tensor = at::empty({0}, default_opts);
          }
          types_file << "Tensor " << "\n";
          types_file << "Contents: 0\n";
          types_file << "Sizes: ";
          for (auto &sz : tensor.sizes()) {
            types_file << sz << ", ";
          }
          types_file << "\n";
          types_file << "Dtype: " << tensor.dtype() << "\n";
          types_file << "Device: " << tensor.device() << "\n";
          types_file << "Requires grad: " << tensor.requires_grad();
          types_file << ";";
          break;
        case fuzzing::FUZZ_SPARSE_TENSOR:
          tensor = *(at::Tensor*) arg;
          if  (!tensor.defined()) {
            tensor = at::empty({0}, default_opts);
          }
          types_file << "SparseTensor " << tensor << ";";
          break;
        case fuzzing::FUZZ_INTARRAY_REF:
          intarrayref = *(at::IntArrayRef*) arg;
          types_file << "IntArrayRef " << intarrayref << ";";
          break;
        case fuzzing::FUZZ_DOUBLEARRAYREF:
          doublearrayref = *(at::ArrayRef<double>*) arg;
          types_file << "ArrayRef<double> " << doublearrayref << ";";
          break;
        case fuzzing::FUZZ_BOOLARRAY:
          // Will create all aftewards
          types_file << "std::array<bool,3>;";
          break;
        case fuzzing::FUZZ_STRING:
          string = *(std::string*) arg;
          types_file << "String " << string << ";";
          break;
        case fuzzing::FUZZ_C10OPTIONAL_TENSOR:
          tensor_opt = *(c10::optional<at::Tensor>*) arg;
          tensor = c10::value_or_else(tensor_opt, [] {return at::Tensor();});
          if  (!tensor.defined()) {
            tensor = at::empty({0}, default_opts);
          }
          types_file << "OptionalTensor " << "\n";
          types_file << "Contents: 0" << "\n";
          types_file << "Sizes: ";
          for (auto &sz : tensor.sizes()) {
            types_file << sz << ", ";
          }
          types_file << "\n";
          types_file << "Dtype: " << tensor.dtype() << "\n";
          types_file << "Device: " << tensor.device() << "\n";
          types_file << "Requires grad: " << tensor.requires_grad();
          types_file << ";";
          break;
        case fuzzing::FUZZ_C10OPTIONAL_INTARRAYREF:
          {
            intarrayref_opt = *(c10::optional<at::IntArrayRef>*) arg;
            if (intarrayref_opt) {
              intarrayref = intarrayref_opt.value();
            } else {
              intarrayref = at::IntArrayRef({});
            }
            types_file << "OptionalIntArrayRef " << intarrayref << ";";
            break;
          }
        case fuzzing::FUZZ_C10OPTIONAL_DOUBLEARRAYREF:
          doublearrayref_opt = *(c10::optional<at::ArrayRef<double>>*) arg;
          if (doublearrayref_opt) {
            doublearrayref = *doublearrayref_opt->data();
          } else {
            doublearrayref = at::ArrayRef<double>({});
          }
          types_file << "OptionalArrayRef<double> " << doublearrayref << ";";
          break;
        case fuzzing::FUZZ_C10OPTIONAL_INT:
          int_opt = *(c10::optional<int>*) arg;
          if (int_opt.has_value()) {
            integer = int_opt.value();
          } else {
            integer = 0;
          }
          types_file << "OptionalInt " << integer << ";";
          break;
        case fuzzing::FUZZ_C10OPTIONAL_LONG:
          long_opt = *(c10::optional<int64_t>*) arg;
          if (long_opt.has_value()) {
            longint = long_opt.value();
          } else {
            longint = 0;
          }
          types_file << "OptionalLong " << longint << ";";
          break;
        case fuzzing::FUZZ_C10OPTIONAL_BOOL:
          bool_opt = (bool) *(c10::optional<bool>*) arg;
          if (bool_opt.has_value())  {
            boolean = bool_opt.value();
          } else {
            boolean = false;
          }
          types_file << "OptionalBool " << boolean << ";";
          break;
        case fuzzing::FUZZ_C10OPTIONAL_DOUBLE:
          double_opt = *(c10::optional<double>*) arg;
          if (double_opt.has_value()) {
            doublenum = double_opt.value();
          } else {
            doublenum = 0.0;
          }
          types_file << "OpiontalDouble " << doublenum << ";";
          break;
        case fuzzing::FUZZ_C10OPTIONAL_SCALAR:
          scalar_opt = *(c10::optional<at::Scalar>*) arg;
          if (scalar_opt.has_value()) {
            scalar = scalar_opt.value();
          } else {
            scalar = at::Scalar(0);
          }
          types_file << "OptionalScalar " << scalar.toLong() << ";";
          break;
        case fuzzing::FUZZ_C10OPTIONAL_SCALARTYPE:
          scalartype_opt = *(c10::optional<at::ScalarType>*) arg;
          if (scalartype_opt.has_value()) {
            scalartype = scalartype_opt.value();
          } else {
            scalartype_opt = at::ScalarType::Int;
          }
          types_file << "OptionalScalarType " << scalartype << ";";
          break;
        case fuzzing::FUZZ_C10OPTIONAL_STRING:
          string_opt = *(c10::optional<std::string>*) arg;
          if (string_opt.has_value()) {
            string = string_opt.value();
          }
          types_file << "OptionalString " << string << ";";
          break;
        case fuzzing::FUZZ_TENSOR_OPTIONS:
          tensor_opts = *(at::TensorOptions*) arg;
          types_file << "TensorOptions " << tensor_opts << ";";
          break;
        /* case fuzzing::FUZZ_DIMNAME: */
        /* case fuzzing::FUZZ_DIMNAME_LIST: */
        /* case fuzzing::FUZZ_TENSOR_LIST: */
      default:
        break;
    }

    }
    /* out_str += "\n--------------------------------------\n"; */
    types_file << std::flush;
    types_file.close();

  }

#endif

#if defined(IVYSYN_VALIDATE)

Fuzzer::Fuzzer(char *fname, std::vector<std::string> types_vec, std::vector<void *> args)
  : cur_fname(fname)
  {
    int total_args;
    int exists_validate, exists_check, exists_true_pos, exists_false_pos;
    struct stat stat_buffer = {};
    fuzzing::TorchType type_enum;
    std::string type;
    void *arg;

    /* std::cout << "In fuzzer for " << fname << std::endl; */

    cur_fname = fname;

    std::string validate_filename = std::string(results_dir) + "/" + cur_fname + ".validate";
    std::string check_filename = std::string(results_dir) + "/" + cur_fname + ".check";
    std::string true_positive_filename = std::string(results_dir) + "/" + cur_fname + ".true_positive";
    std::string false_positive_filename = std::string(results_dir) + "/" + cur_fname + ".false_positive";

    exists_validate = stat(validate_filename.c_str(), &stat_buffer) == 0;
    exists_true_pos = stat(true_positive_filename.c_str(), &stat_buffer) == 0;

    /* We don't need to validate this kernel */
    if (!exists_validate || exists_true_pos) {
      _should_validate = false;
      return;
    }

    exists_check = stat(check_filename.c_str(), &stat_buffer) == 0;

    /* If a check file exists, it means the kernel crashed
     * when we checked it, log it as correctly validated */
    if (exists_check) {
      std::fstream true_positive_file;
      create_file(true_positive_filename, true_positive_file, std::ios::out);
      std::remove(check_filename.c_str());
      exists_false_pos = stat(false_positive_filename.c_str(), &stat_buffer) == 0;
      if (exists_false_pos) {
        std::remove(false_positive_filename.c_str());
      }
      _should_validate = false;
      return;
    } else {
      std::fstream check_file;
      create_file(check_filename, check_file, std::ios::out);
    }

    /* We should check if the crash for this kernel is a true positive */
    _should_validate = true;

    std::cout << "Will validate " << fname << std::endl;

    total_args = types_vec.size();

    std::vector<TorchType> orig_types = {};
    std::vector<TorchType> orig_types_check = {};
    for (int i = 0; i < total_args; i++) {

      type = types_vec.at(i);

      auto it = map_str_enum.find(type);
      if (it != map_str_enum.end()) {
        indices.push_back(0);
        type_enum = it->second;
      }
      /* else { */
      /*   mark_unknown_type(type); */
      /* } */

      arg = args.at(i);
      switch (type_enum) {
        case fuzzing::FUZZ_INT:
        case fuzzing::FUZZ_C10OPTIONAL_INT:
          orig_types.push_back(fuzzing::FUZZ_INT);
          orig_types_check.push_back(fuzzing::FUZZ_INT);
          break;
        case fuzzing::FUZZ_LONG:
        case fuzzing::FUZZ_C10OPTIONAL_LONG:
          orig_types.push_back(fuzzing::FUZZ_LONG);
          orig_types_check.push_back(fuzzing::FUZZ_INT);
          break;
        case fuzzing::FUZZ_FLOAT:
          orig_types.push_back(fuzzing::FUZZ_FLOAT);
          orig_types_check.push_back(fuzzing::FUZZ_DOUBLE);
          break;
        case fuzzing::FUZZ_DOUBLE:
        case fuzzing::FUZZ_C10OPTIONAL_DOUBLE:
          orig_types.push_back(fuzzing::FUZZ_DOUBLE);
          orig_types_check.push_back(fuzzing::FUZZ_DOUBLE);
          break;
        case fuzzing::FUZZ_BOOLEAN:
        case fuzzing::FUZZ_C10OPTIONAL_BOOL:
          orig_types.push_back(fuzzing::FUZZ_BOOLEAN);
          orig_types_check.push_back(fuzzing::FUZZ_BOOLEAN);
          break;
        case fuzzing::FUZZ_SCALAR:
        case fuzzing::FUZZ_C10OPTIONAL_SCALAR:
          orig_types.push_back(fuzzing::FUZZ_SCALAR);
          orig_types_check.push_back(fuzzing::FUZZ_SCALAR);
          break;
        case fuzzing::FUZZ_SCALARTYPE:
        case fuzzing::FUZZ_C10OPTIONAL_SCALARTYPE:
          orig_types.push_back(fuzzing::FUZZ_SCALARTYPE);
          orig_types_check.push_back(fuzzing::FUZZ_SCALARTYPE);
          break;
        case fuzzing::FUZZ_TENSOR:
        case fuzzing::FUZZ_C10OPTIONAL_TENSOR:
        case fuzzing::FUZZ_SPARSE_TENSOR:
          orig_types.push_back(fuzzing::FUZZ_TENSOR);
          orig_types_check.push_back(fuzzing::FUZZ_TENSOR);
          break;
        case fuzzing::FUZZ_INTARRAY_REF:
        case fuzzing::FUZZ_C10OPTIONAL_INTARRAYREF:
          orig_types.push_back(fuzzing::FUZZ_INTARRAY_REF);
          orig_types_check.push_back(fuzzing::FUZZ_INTARRAY_REF);
          break;
        case fuzzing::FUZZ_DOUBLEARRAYREF:
        case fuzzing::FUZZ_C10OPTIONAL_DOUBLEARRAYREF:
          orig_types.push_back(fuzzing::FUZZ_DOUBLEARRAYREF);
          orig_types_check.push_back(fuzzing::FUZZ_DOUBLEARRAYREF);
          break;
        case fuzzing::FUZZ_BOOLARRAY:
          orig_types.push_back(fuzzing::FUZZ_BOOLARRAY);
          orig_types_check.push_back(fuzzing::FUZZ_BOOLARRAY);
          break;
        case fuzzing::FUZZ_STRING:
        case fuzzing::FUZZ_C10OPTIONAL_STRING:
          orig_types.push_back(fuzzing::FUZZ_STRING);
          orig_types_check.push_back(fuzzing::FUZZ_STRING);
          break;
        case fuzzing::FUZZ_TENSOR_OPTIONS:
          orig_types.push_back(fuzzing::FUZZ_TENSOR_OPTIONS);
          orig_types_check.push_back(fuzzing::FUZZ_TENSOR_OPTIONS);
          break;
        default:
          std::cout << "Can't validate " << fname << ", unknown type" << std::endl;
          _should_validate = false;
          std::remove(check_filename.c_str());
      }

    }

    int tensor_idx = 0, scalar_idx = 0, intarray_idx = 0, double_idx = 0, float_idx = 0,
        int_idx = 0, long_idx = 0, bool_idx = 0, boolarray_idx = 0, string_idx = 0;
    std::string contents_str;
    std::string shape_str;
    std::string type_str;
    std::string dtype_str;
    std::string req_grad_str;

    std::string int_str;

    std::ifstream validate_file(validate_filename);

    auto options = c10::TensorOptions();
    c10::ScalarType dtype;
    at::Tensor tensor;
    at::Scalar scalar;
    at::IntArrayRef intarray, dims_intarray;
    long long_t, dim;
    int int_t;
    std::array<bool,3> boolarray = {};
    std::string string_t;
    bool bool_t, bool0, bool1, bool2;
    double double_t;
    float float_t;
    bool req_grad, is_empty;
    long *dims, *ints;
    std::vector<long> dims_vec = {};
    std::vector<long> int_vec = {};

    int inputs_read = 0;
    while (std::getline(validate_file, type_str)) {
      std::cout << "Reading input " << inputs_read << std::endl;
      if (inputs_read > total_args) {
          std::cout << "Read more inpus than expected, not validating" << std::endl;
          _should_validate = false;
          std::remove(check_filename.c_str());
          return;
      }
      std::cout << type_str << std::endl;
      inputs_read++;
      if (type_str.compare("opttensor") == 0) {
        nullopt_indices.push_back(inputs_read - 1);
      } else if (type_str.compare("tensor") == 0) {
        dims_vec = {};
        if (orig_types_check.at(inputs_read - 1) != fuzzing::FUZZ_TENSOR) {
          std::cout << "Input type #" << inputs_read - 1 << " did not match, not validating" << std::endl;
          std::cout << "expected tensor (" << fuzzing::FUZZ_TENSOR << "), got " << orig_types_check.at(inputs_read - 1) << std::endl;
          _should_validate = false;
          std::remove(check_filename.c_str());
          return;
        }

        std::getline(validate_file, dtype_str);
        std::cout << dtype_str << std::endl;
        if (dtype_str.compare("torch.float32") == 0) {
          dtype = c10::kFloat;
        } else if (dtype_str.compare("torch.float64") == 0) {
          dtype = c10::kDouble;
        } else if (dtype_str.compare("torch.int32") == 0) {
          dtype = c10::kInt;
        } else if (dtype_str.compare("torch.int64") == 0) {
          dtype = c10::kLong;
        } else {
          std::cout << "Input type #" << inputs_read - 1 << " has unknown tensor type, not validating" << std::endl;
          _should_validate = false;
          std::remove(check_filename.c_str());
          return;
        }

        std::getline(validate_file, contents_str);
        std::cout << contents_str << std::endl;
        std::getline(validate_file, shape_str);
        std::cout << shape_str << std::endl;
        std::stringstream shape_sstream(shape_str);
        std::string dim_str;
        if (shape_str.compare("") != 0) {
          while (shape_sstream.good()) {
            std::getline(shape_sstream, dim_str, ',');
            std::cout << dim_str << std::endl;
            dim = std::stol(dim_str);
            dims_vec.push_back(dim);
          }
          size_t sz = dims_vec.size();
          dims = new long int[sz];
          int idx = 0;
          for (auto dimv : dims_vec) {
            dims[idx++] = dimv;
          }
          dims_intarray = at::IntArrayRef(dims, sz);
          is_empty = false;
        } else {
          std::cout << "Empty tensor" << std::endl;
          is_empty = true;
        }

        std::getline(validate_file, req_grad_str);
        std::cout << req_grad_str << std::endl;
        if (req_grad_str.compare("False") == 0) {
          req_grad = false;
        } else {
          req_grad = true;
        }

        options = at::TensorOptions().dtype(dtype).requires_grad(req_grad);

        if (is_empty) {
          tensor = at::empty(dims_intarray, options);
          tensor_contents.push_back(123123123);
        } else {
          switch (dtype) {
            case c10::kFloat:
              float_t = std::stof(contents_str);
              tensor = at::full(dims_intarray, float_t, options);
              tensor_contents.push_back((double) float_t);
              break;
            case c10::kDouble:
              double_t = std::stod(contents_str);
              tensor = at::full(dims_intarray, double_t, options);
              tensor_contents.push_back(double_t);
              break;
            case c10::kInt:
              int_t = std::stoi(contents_str);
              tensor = at::full(dims_intarray, int_t, options);
              tensor_contents.push_back((double) int_t);
              break;
            case c10::kLong:
              long_t = std::stol(contents_str);
              tensor = at::full(dims_intarray, long_t, options);
              tensor_contents.push_back((double) long_t);
              break;
            default:
              std::cout << "Unknown dtype " << dtype << std::endl;
              break;
          }
        }

        std::cout << "Adding tensor mutation" << std::endl;
        if (cur_fname.find("mkldnn") != std::string::npos && (tensor.dtype() == c10::kFloat || tensor.dtype() == c10::kBFloat16)) {
          tensor = tensor.to_mkldnn();
        }
        tensor_mutations.push_back(tensor);
        indices[inputs_read - 1] = tensor_idx;
        tensor_idx++;

      } else if (type_str.compare("scalar") == 0) {
        if (orig_types_check.at(inputs_read - 1) != fuzzing::FUZZ_SCALAR) {
          std::cout << "Input type #" << inputs_read - 1 << " did not match, not validating" << std::endl;
          std::cout << "expected scalar (" << fuzzing::FUZZ_SCALAR << "), got " << orig_types_check.at(inputs_read - 1) << std::endl;
          _should_validate = false;
          std::remove(check_filename.c_str());
          return;
        }

        std::getline(validate_file, dtype_str);
        std::cout << dtype_str << std::endl;
        std::getline(validate_file, contents_str);
        std::cout << contents_str << std::endl;
        if (dtype_str.compare("float") == 0) {
          float_t = std::stof(contents_str);
          scalar = at::Scalar(float_t);
        } else if (dtype_str.compare("int") == 0) {
          int_t = std::stod(contents_str);
          scalar = at::Scalar(int_t);
        } else {
          std::cout << "Input type #" << inputs_read - 1 << " has unknown scalar type, not validating" << std::endl;
          _should_validate = false;
          std::remove(check_filename.c_str());
          return;
        }

        std::cout << "Adding scalar mutation" << std::endl;
        scalar_mutations.push_back(scalar);
        indices[inputs_read - 1] = scalar_idx;
        scalar_idx++;
      } else if (type_str.compare("intarray") == 0) {
        int_vec = {};

        if (orig_types_check.at(inputs_read - 1) != fuzzing::FUZZ_INTARRAY_REF) {
          std::cout << "Input type #" << inputs_read - 1 << " did not match, not validating" << std::endl;
          std::cout << "expected intarray (" << fuzzing::FUZZ_INTARRAY_REF << "), got " << orig_types_check.at(inputs_read - 1) << std::endl;
          _should_validate = false;
          std::remove(check_filename.c_str());
          return;
        }

        std::getline(validate_file, contents_str);
        std::cout << contents_str << std::endl;
        std::stringstream contents_sstream(contents_str);
        if (contents_str.compare("") != 0) {
          while (contents_sstream.good()) {
            std::getline(contents_sstream, int_str, ',');
            std::cout << int_str << std::endl;
            long_t = std::stol(int_str);
            int_vec.push_back(long_t);
          }
          ints = new long int[int_vec.size()];
          int idx = 0;
          for (auto intv : int_vec) {
            ints[idx++] = intv;
          }
          size_t sz = int_vec.size();
          intarray = at::IntArrayRef(ints, sz);
        } else {
          intarray = at::IntArrayRef({});
        }

        std::cout << "Adding intarray mutation" << std::endl;
        intarrayref_mutations.push_back(intarray);
        indices[inputs_read - 1] = intarray_idx;
        intarray_idx++;
      } else if (type_str.compare("int") == 0) {
        if (orig_types_check.at(inputs_read - 1) != fuzzing::FUZZ_INT) {
          std::cout << "Input type #" << inputs_read - 1 << " did not match, not validating" << std::endl;
          std::cout << "expected int (" << fuzzing::FUZZ_INT << "), got " << orig_types_check.at(inputs_read - 1) << std::endl;
          _should_validate = false;
          std::remove(check_filename.c_str());
          return;
        }
        std::getline(validate_file, contents_str);
        std::cout << contents_str << std::endl;
        long_t = std::stol(contents_str);
        if (orig_types.at(inputs_read - 1) == fuzzing::FUZZ_INT) {
          std::cout << "Adding int mutation" << std::endl;
          int_mutations.push_back((int) long_t);
          indices[inputs_read - 1] = int_idx;
          int_idx++;
        } else {
          std::cout << "Adding long mutation" << std::endl;
          long_mutations.push_back(long_t);
          indices[inputs_read - 1] = long_idx;
          long_idx++;
        }

      } else if (type_str.compare("double") == 0) {
        if (orig_types_check.at(inputs_read - 1) != fuzzing::FUZZ_DOUBLE) {
          std::cout << "Input type #" << inputs_read - 1 << " did not match, not validating" << std::endl;
          std::cout << "expected double (" << fuzzing::FUZZ_DOUBLE << "), got " << orig_types_check.at(inputs_read - 1) << std::endl;
          _should_validate = false;
          std::remove(check_filename.c_str());
          return;
        }
        std::getline(validate_file, contents_str);
        std::cout << contents_str << std::endl;
        double_t = std::stod(contents_str);
        if (orig_types.at(inputs_read - 1) == fuzzing::FUZZ_FLOAT) {
          std::cout << "Adding float mutation" << std::endl;
          float_mutations.push_back((float) double_t);
          indices[inputs_read - 1] = float_idx;
          float_idx++;
        } else {
          std::cout << "Adding double mutation" << std::endl;
          double_mutations.push_back(double_t);
          indices[inputs_read - 1] = double_idx;
          double_idx++;
        }
      } else if (type_str.compare("string") == 0) {
        if (orig_types_check.at(inputs_read - 1) != fuzzing::FUZZ_STRING) {
          std::cout << "Input type #" << inputs_read - 1 << " did not match, not validating" << std::endl;
          std::cout << "expected string (" << fuzzing::FUZZ_STRING << "), got " << orig_types_check.at(inputs_read - 1) << std::endl;
          _should_validate = false;
          std::remove(check_filename.c_str());
          return;
        }
        std::getline(validate_file, contents_str);
        std::cout << contents_str << std::endl;
        string_t.assign(contents_str);

        std::cout << "Adding string mutation" << std::endl;
        string_mutations.push_back(string_t);
        indices[inputs_read - 1] = string_idx;
        string_idx++;
      } else if (type_str.compare("bool") == 0) {
        if (orig_types_check.at(inputs_read - 1) != fuzzing::FUZZ_BOOLEAN) {
          std::cout << "Input type #" << inputs_read - 1 << " did not match, not validating" << std::endl;
          std::cout << "expected boolean (" << fuzzing::FUZZ_BOOLEAN << "), got " << orig_types_check.at(inputs_read - 1) << std::endl;
          _should_validate = false;
          std::remove(check_filename.c_str());
          return;
        }
        std::getline(validate_file, contents_str);
        std::cout << contents_str << std::endl;
        if (contents_str.compare("1") == 0) {
          bool_t = true;
        } else {
          bool_t = false;
        }
        std::cout << "Adding bool mutation" << std::endl;
        bool_mutations.push_back(bool_t);
        indices[inputs_read - 1] = bool_idx;
        bool_idx++;
      } else if (type_str.compare("boolarray") == 0) {
        if (orig_types_check.at(inputs_read - 1) != fuzzing::FUZZ_BOOLARRAY) {
          std::cout << "Input type #" << inputs_read - 1 << " did not match, not validating" << std::endl;
          std::cout << "expected boolarray (" << fuzzing::FUZZ_BOOLARRAY << "), got " << orig_types_check.at(inputs_read - 1) << std::endl;
          _should_validate = false;
          std::remove(check_filename.c_str());
          return;
        }
        std::getline(validate_file, contents_str);
        std::cout << contents_str << std::endl;
        std::stringstream contents_sstream(contents_str);
        std::getline(contents_sstream, int_str, ',');
        if (int_str.compare("1") == 0) {
          bool0 = true;
        } else {
          bool0 = false;
        }
        std::getline(contents_sstream, int_str, ',');
        if (int_str.compare("1") == 0) {
          bool1 = true;
        } else {
          bool1 = false;
        }
        std::getline(contents_sstream, int_str, ',');
        if (int_str.compare("1") == 0) {
          bool2 = true;
        } else {
          bool2 = false;
        }
        boolarray = std::array<bool,3>{bool0, bool1, bool2};
        std::cout << "Adding boolarray mutation" << std::endl;
        bool_arrays.push_back(boolarray);
        indices[inputs_read - 1] = boolarray_idx;
        boolarray_idx++;
      }
    }

    std::cout << "Read all args" << std::endl;

    if (inputs_read != total_args) {
      std::cout << "Got different number of args for " << fname << ", not validating" << std::endl;
      _should_validate = false;
      std::remove(check_filename.c_str());
      return;
    }
  }

  bool Fuzzer::should_validate()
  {
    return _should_validate;
  }

  /* This will only be reached if the mutation run without crashing */
  void Fuzzer::false_positive()
  {
    std::string false_positive_filename = std::string(results_dir) + "/" + cur_fname + ".false_positive";
    std::string check_filename = std::string(results_dir) + "/" + cur_fname + ".check";
    std::fstream false_positive_file;
    struct stat stat_buffer = {};

    int exists_check = stat(check_filename.c_str(), &stat_buffer) == 0;

    if (!exists_check) {
      return;
    }

    create_file(false_positive_filename, false_positive_file, std::ios::out);
    std::remove(check_filename.c_str());
  }

#endif

#if !defined(IVYSYN_VALIDATE) && !defined(IVYSYN_COLLECT_TYPES)
  Fuzzer::Fuzzer(char *fname, std::vector<std::string> types_vec, std::vector<void *> args)
    : cur_fname(fname) {

      std::cout << "In fuzzer for " << cur_fname << std::endl;

      main_pool_done = false;

      original_args = args;

      // printf("Initializing fuzzer...\n");
      cur_fname_glob.assign(cur_fname);

      std::string mut_filename;
      std::string last_timestamp_filename;
      std::string time_filename;
      std::string except_filename;
      std::string total_filename;
      std::string start_filename;
      std::string mutfile_pattern;
      std::string mutfile_prefix;
      std::string proc_filename;
      std::string nofuzz_filename;

      glob_t glob_result = {};
      struct stat stat_buffer = {};
      struct timespec ts = {};
      std::fstream total_file;

      bool restore = false, do_resume = false;
      long long last_mutation = -1, last_timestamp = -1;
      bool has_tensor = false, has_intarrayref = false, has_scalar = false,
           has_doublearrayref = false, has_sparse_tensor = false, has_tensor_options = false;

      int glob_ret = 0;
      pid_t mypid = 0;

      char *existing_pid;

      int total_args, i;
      fuzzing::TorchType type_enum;
      std::string type;

      bool boolean = false;
      int integer = 0;
      float floatnum = 0.0;
      double doublenum = 0.0;
      int64_t longint = 0;
      std::array<bool,3> boolarray = {};
      at::Tensor tensor;
      at::Tensor sparse_tensor;
      at::TensorOptions tensor_opts;
      std::string string;
      at::IntArrayRef intarrayref, zero_dims;
      at::IntArrayRef *intarray_ptr;
      at::IntArrayRef tmp;
      at::Scalar scalar;
      at::ScalarType scalartype;
      at::ArrayRef<double> doublearrayref;
      c10::optional<at::Tensor> tensor_opt;
      c10::optional<at::IntArrayRef> intarrayref_opt;
      c10::optional<at::Scalar> scalar_opt;
      c10::optional<at::ScalarType> scalartype_opt;
      c10::optional<at::ArrayRef<double>> doublearrayref_opt;
      c10::optional<int> int_opt;
      c10::optional<int64_t> long_opt;
      c10::optional<double> double_opt;
      c10::optional<bool> bool_opt;
      c10::optional<std::string> string_opt;

      at::TensorOptions default_opts = c10::TensorOptions().dtype(c10::kDouble).layout(c10::kStrided).device(tensor_dev);

      mypid = ::getpid();

      mutfile_pattern = std::string(results_dir) + "/" + cur_fname + "_mutations.log.*";
      mutfile_prefix = std::string(results_dir) + "/" + cur_fname + "_mutations.log";

      mut_filename = std::string(results_dir) + "/" + cur_fname + "_mutations.log." + std::to_string(mypid);
      last_timestamp_filename = std::string(results_dir) + "/" + cur_fname + ".last_timestamp." + std::to_string(mypid);
      time_filename = std::string(results_dir) + "/" + cur_fname + ".time." + std::to_string(mypid);
      except_filename = std::string(results_dir) + "/" + cur_fname + ".failed." + std::to_string(mypid);
      start_filename = std::string(results_dir) + "/" + cur_fname + ".start";
      total_filename = std::string(results_dir) + "/totals.txt";
      nofuzz_filename = std::string(results_dir) + "/" + cur_fname + ".nofuzz";

      mutations_logger_filename = mut_filename;
      timestamp_logger_filename = last_timestamp_filename;

      std::ios_base::openmode fflags = std::ios::out | std::ios::in | std::ios::trunc;

      /*
       * Seed it with the pid such that two processes will
       * get different values in the next iterations
       */
      std::mt19937_64 rng(std::random_device{}());
      std::uniform_int_distribution<std::mt19937_64::result_type> dist(0, 2000);

      int slp = dist(rng);
      std::this_thread::sleep_for(std::chrono::milliseconds(slp));

      glob_ret = glob(mutfile_pattern.c_str(), 0, NULL, &glob_result);
      // A mutation file for the same function exists
      if (glob_ret != GLOB_NOMATCH && !restore) {
      /* if (glob_ret != GLOB_NOMATCH) { */

        // A lot of empty mutation files, probably deadlock or bug, stop
        // fuzzing this function
        if (glob_result.gl_pathc > MAX_EMPTY_LOG_FILES) {
          mark_fuzzing_done();
          std::cout << mypid << ": " << cur_fname << "has a lot of empty mutation files, skip" << std::endl;
          return;
        }

        // Check if the existing mutation files belong to a running process
        for(size_t i = 0; i < glob_result.gl_pathc && !restore; ++i) {

          // Grab the pid suffix from the mutation filename
          existing_pid = glob_result.gl_pathv[i] + mutfile_prefix.length() + 1;
          proc_filename = "/proc/" + std::string(existing_pid);

          // Check if a process with that pid exists right now
          if (stat(proc_filename.c_str(), &stat_buffer) == 0){

            // The mutations file belongs to a running process, skip
            /* printf("%d: %s belongs to a running process, skipping\n", mypid, glob_result.gl_pathv[i]); */
            std::cout << mypid << ": " << glob_result.gl_pathv[i] << "belongs to a running process, skipping" << std::endl;
            total_mutations = 0;
            is_running = true;
            globfree(&glob_result);
            return;

          } else {

            // The mutations file doesn't belong to any running process, something crashed or it was killed
            mutations_restore_filename = glob_result.gl_pathv[i];
            size_t mut_str_idx = mutations_restore_filename.find("_mutations.log");
            timestamp_restore_filename = mutations_restore_filename;
            timestamp_restore_filename.replace(mut_str_idx, std::string("_mutations.log").length(), ".last_timestamp");

            restore = true;

            // Was killed by the watchdog and didn't crash, don't log a crash
            if (was_killed(cur_fname)) {
              do_resume = true;
            }

          }
        }

      }
      globfree(&glob_result);

      if (do_resume) {
        std::cout << mypid << ": " << cur_fname << " was killed, will resume from " << mutations_restore_filename << std::endl;
      } else if (restore) {
        std::cout << mypid << ": " << cur_fname << " crashed, will resume from " << mutations_restore_filename << std::endl;
      } else {
        create_file(time_filename, time_file, fflags);
        time_file.rdbuf()->pubsetbuf(nullptr, 0);
        create_file(except_filename, except_file, fflags);
        except_file.rdbuf()->pubsetbuf(nullptr, 0);
      }

      std::cout << mypid << ": Fuzzing function " << cur_fname << std::endl;

      /* Disable buffering else program might crash before writing to logger */
      create_file(mutations_logger_filename.c_str(), mutations_file, fflags);
      mutations_file.rdbuf()->pubsetbuf(nullptr, 0);
      create_file(last_timestamp_filename, last_timestamp_file, fflags);
      last_timestamp_file.rdbuf()->pubsetbuf(nullptr, 0);

      if (!restore) {

        if (stat(start_filename.c_str(), &stat_buffer) == 0) {
            std::remove(mutations_restore_filename.c_str());
            std::remove(timestamp_restore_filename.c_str());
            total_mutations = 0;
            is_running = true;
            return;
        }

        create_file(start_filename, start_file, fflags);
        clock_gettime(CLOCK_MONOTONIC, &ts);
        /* Log start time in seconds */
        start_file << ts.tv_sec << std::endl;
        start_file.close();
      }


      total_args = types_vec.size();

      std::cout << "Total args: " << total_args << std::endl;

      void *arg;
      for (i = 0; i < total_args; i++) {

        type = types_vec.at(i);

        auto it = map_str_enum.find(type);
        if (it != map_str_enum.end()) {
          indices.push_back(0);
          type_enum = it->second;
        } else {
          mark_unknown_type(type);
        }

        /* Extract the argument from the va list */
        func_types.push_back(type_enum);
        arg = args.at(i);
        switch (type_enum) {
          case fuzzing::FUZZ_INT:
            integer = *(int*) arg;
            int_mutations.push_back(integer);
            break;
          case fuzzing::FUZZ_LONG:
            longint = *(long*) arg;
            long_mutations.push_back(longint);
            break;
          case fuzzing::FUZZ_FLOAT:
          case fuzzing::FUZZ_DOUBLE:
            doublenum = *(double*) arg;
            double_mutations.push_back(doublenum);
            break;
          case fuzzing::FUZZ_BOOLEAN:
            boolean = *(bool*) arg;
            break;
          case fuzzing::FUZZ_SCALAR:
            scalar = *(at::Scalar*) arg;
            scalar_mutations.push_back(scalar);
            has_scalar = true;
            break;
          case fuzzing::FUZZ_SCALARTYPE:
            scalartype = *(at::ScalarType*) arg;
            break;
          case fuzzing::FUZZ_TENSOR:
            tensor = *(at::Tensor*) arg;
            /* std::cout << "Original tensor: " << tensor << std::endl; */
            std::cout << "Original tensor: at idx " << i << std::endl;
            std::cout << "Sizes: ";
            for (auto &sz : tensor.sizes()) {
              std::cout << sz << ", ";
            }
            std::cout << "\n";
            std::cout << "Dtype: " << tensor.dtype() << "\n";
            std::cout << "Device: " << tensor.device() << "\n";
            std::cout << "Requires grad: " << tensor.requires_grad();
            if (tensor.is_mkldnn()) {
              have_mkldnn_tensors = true;
            }
            if (tensor.is_quantized()) {
              mark_fuzzing_done();
              create_file(nofuzz_filename, nofuzz_file, std::ios::out);
              return;
            }
            if (!tensor.defined()) {
              std::cout << "Tensor at idx " << i << " is undefined" << std::endl;
              tensor_dims.push_back(0);
              original_tensor_types.push_back(c10::kDouble);
            } else {
              tensor_dims.push_back(tensor.sizes());
              std::cout << "Tensor at idx " << i << " sizes: ";
              for (auto n : tensor.sizes()) {
                std::cout << n << ", ";
              }
              std::cout << std::endl;
              original_tensor_types.push_back(tensor.scalar_type());
            }
            has_tensor = true;
            break;
          case fuzzing::FUZZ_SPARSE_TENSOR:
            tensor = *(at::Tensor*) arg;
            if  (!tensor.defined()) {
              tensor = at::empty({0}, default_opts);
            }
            sparse_tensor_dims.push_back(sparse_tensor.sizes());
            has_sparse_tensor = true;
            break;
          case fuzzing::FUZZ_INTARRAY_REF:
            intarrayref = *(at::IntArrayRef*) arg;
            /* std::cout << "Original intarrayref: " << intarrayref << std::endl; */
            intarrayref_sizes.insert(intarrayref.size());
            intarrayref_sizes_vec.push_back(intarrayref.size());
            has_intarrayref = true;
            break;
          case fuzzing::FUZZ_DOUBLEARRAYREF:
            doublearrayref = *(at::ArrayRef<double>*) arg;
            doublearrayref_sizes.insert(doublearrayref.size());
            has_doublearrayref = true;
            break;
          case fuzzing::FUZZ_BOOLARRAY:
            /* Will create all aftewards */
            break;
          case fuzzing::FUZZ_STRING:
            string = *(std::string*) arg;
            string_mutations.push_back(string);
            break;
          case fuzzing::FUZZ_C10OPTIONAL_TENSOR:
            tensor_opt = *(c10::optional<at::Tensor>*) arg;
            if (!tensor_opt.has_value()) {
              nullopt_indices.push_back(i);
              std::cout << "Original tensor: nullopt" << std::endl;
            } else {
              tensor = c10::value_or_else(tensor_opt, [] {return at::Tensor();});
              /* std::cout << "Original tensor: " << tensor << std::endl; */
              std::cout << "Original tensor: at idx " << i << std::endl;
              if (tensor.is_mkldnn()) {
                have_mkldnn_tensors = true;
              }
              if (!tensor.defined()) {
                tensor_dims.push_back(0);
                original_tensor_types.push_back(c10::kDouble);
              } else {
                tensor_dims.push_back(tensor.sizes());
                original_tensor_types.push_back(tensor.scalar_type());
              }
            }
            has_tensor = true;
            break;
          case fuzzing::FUZZ_C10OPTIONAL_INTARRAYREF:
            {
              intarrayref_opt = *(c10::optional<at::IntArrayRef>*) arg;
              if (intarrayref_opt) {
                intarrayref = intarrayref_opt.value();
                intarrayref_sizes.insert(intarrayref.size());
                intarrayref_sizes_vec.push_back(intarrayref.size());
              }
              has_intarrayref = true;
              break;
            }
          case fuzzing::FUZZ_C10OPTIONAL_DOUBLEARRAYREF:
            doublearrayref_opt = *(c10::optional<at::ArrayRef<double>>*) arg;
            if (doublearrayref_opt) {
              doublearrayref = *doublearrayref_opt->data();
              doublearrayref_sizes.insert(doublearrayref.size());
            }
            has_doublearrayref = true;
            break;
          case fuzzing::FUZZ_C10OPTIONAL_INT:
            int_opt = *(c10::optional<int>*) arg;
            break;
          case fuzzing::FUZZ_C10OPTIONAL_LONG:
            long_opt = *(c10::optional<int64_t>*) arg;
            break;
          case fuzzing::FUZZ_C10OPTIONAL_BOOL:
            bool_opt = (bool) *(c10::optional<bool>*) arg;
            break;
          case fuzzing::FUZZ_C10OPTIONAL_DOUBLE:
            double_opt = *(c10::optional<double>*) arg;
            break;
          case fuzzing::FUZZ_C10OPTIONAL_SCALAR:
            scalar_opt = *(c10::optional<at::Scalar>*) arg;
            has_scalar = true;
            break;
          case fuzzing::FUZZ_C10OPTIONAL_SCALARTYPE:
            scalartype_opt = *(c10::optional<at::ScalarType>*) arg;
            break;
          case fuzzing::FUZZ_C10OPTIONAL_STRING:
            string_opt = *(c10::optional<std::string>*) arg;
            if (string_opt.has_value()) {
              string = string_opt.value();
              string_mutations.push_back(string);
            }
            break;
          case fuzzing::FUZZ_TENSOR_OPTIONS:
            tensor_opts = *(at::TensorOptions*) arg;
            tensor_options_mutations.push_back(tensor_opts);
            has_tensor_options = true;
            break;
          case fuzzing::FUZZ_DIMNAME:
          case fuzzing::FUZZ_DIMNAME_LIST:
          case fuzzing::FUZZ_TENSOR_LIST:
          default:
            break;
        }

      }

      /*
       * Avoid initializing certain pools if we don't have any arguments
       * of that type
       */
      auto shuf_rng = std::default_random_engine(RAND_SEED);
      if (has_tensor) {
        initialize_tensor_pool();
        std::shuffle(std::begin(tensor_mutations), std::end(tensor_mutations), shuf_rng);
        /* Reset it so contents get shuffle accordingly */
        shuf_rng = std::default_random_engine(RAND_SEED);
        std::shuffle(std::begin(tensor_contents), std::end(tensor_contents), shuf_rng);
      }
      if (has_intarrayref) {
        initialize_intarrayref_pool();
        std::shuffle(std::begin(intarrayref_mutations), std::end(intarrayref_mutations), shuf_rng);
      }
      if (has_scalar) {
        initialize_scalar_pool();
        std::shuffle(std::begin(scalar_mutations), std::end(scalar_mutations), shuf_rng);
      }
      if (has_doublearrayref) {
        initialize_doublearrayref_pool();
        std::shuffle(std::begin(doublearrayref_mutations), std::end(doublearrayref_mutations), shuf_rng);
      }
      if (has_sparse_tensor) {
        initialize_sparse_tensor_pool();
        std::shuffle(std::begin(sparse_tensor_mutations), std::end(sparse_tensor_mutations), shuf_rng);
      }
      if (has_tensor_options) {
        initialize_tensor_options_pool();
        std::shuffle(std::begin(tensor_options_mutations), std::end(tensor_options_mutations), shuf_rng);
      }
      initialize_boolarrays();
      std::shuffle(std::begin(bool_arrays), std::end(bool_arrays), shuf_rng);

      calculate_total_mutations();

      /* File to log total number of mutations */
      total_file.clear();
      total_file.open(total_filename, std::ios::app);
      if (total_file.fail()) {
        std::cout << "Failed to open " << total_filename << std::endl;
        std::cout << "Error: " << strerror(errno) << std::endl;
      }
      total_file << cur_fname << ":" << all_mutations << std::endl << std::flush;
      total_file.close();

      if (restore) {

        mutations_restore.open(mutations_restore_filename, std::ios::out | std::ios::in);
        timestamp_restore.open(timestamp_restore_filename, std::ios::out | std::ios::in);

        std::string last_line;

        getline(mutations_restore, last_line);
        if (last_line.length() > 0) {
          last_mutation = std::stoll(last_line);
        } else {
          printf("Error: reading %s (got %s)...\n", mutations_restore_filename.c_str(), last_line.c_str());
        }

        getline(timestamp_restore, last_line);
        if (last_line.length() <= 0) {
          printf("Error: reading %s (got %s)...\n", timestamp_restore_filename.c_str(), last_line.c_str());
        } else {
          last_timestamp = std::stoll(last_line);
        }
        if (last_mutation >= 0) {
            restore_last_mutation(last_mutation, last_timestamp, do_resume);
            /* Delete the file since we already logged the crash */
            std::remove(mutations_restore_filename.c_str());
            std::remove(timestamp_restore_filename.c_str());
        }
      } else {
        indices[0] = -1;
      }

      struct sigaction timeout_sigaction = {};
      timeout_sigaction.sa_handler = handle_timeout;
      sigaction(SIGALRM, &timeout_sigaction, NULL);
      alarm(TIMEOUT_SECS);
    }

  void Fuzzer::log_current_mutation(std::fstream &file) {

    at::Tensor tensor;
    at::Tensor sparse_tensor;
    at::TensorOptions tensor_opts;
    bool boolean;
    bool a,b,c;
    int integer;
    double doublenum, contents;
    int64_t longint;
    at::IntArrayRef intarrayref;
    at::Scalar scalar;
    at::ScalarType scalartype;
    at::ArrayRef<double> doublearrayref;
    std::array<bool,3> boolarray;
    std::string string;
    c10::optional<at::Tensor> tensor_opt;
    c10::optional<at::IntArrayRef> intarrayref_opt;
    c10::optional<at::Scalar> scalar_opt;
    c10::optional<at::ScalarType> scalartype_opt;
    c10::optional<at::ArrayRef<double>> doublearrayref_opt;
    c10::optional<int> int_opt;
    c10::optional<int64_t> long_opt;
    c10::optional<double> double_opt;
    c10::optional<bool> bool_opt;
    c10::optional<std::string> string_opt;

    int idx = 0;
    for (auto &type_enum : func_types) {
      std::cout << "Logging index " << idx++ << std::endl;

      switch (type_enum) {
        case fuzzing::FUZZ_INT:
          integer = get_next_mut_int();
          std::cout << "int " << integer << ";";
          file << "int " << integer << ";";
          break;
        case fuzzing::FUZZ_LONG:
          longint = get_next_mut_long();
          std::cout << "int64_t " << longint << ";";
          file << "int64_t " << longint << ";";
          break;
        case fuzzing::FUZZ_FLOAT:
        case fuzzing::FUZZ_DOUBLE:
          doublenum = get_next_mut_long();
          std::cout << "double " << doublenum << ";";
          file << "double " << doublenum << ";";
          break;
        case fuzzing::FUZZ_BOOLEAN:
          boolean = get_next_mut_bool();
          std::cout << "bool " << boolean << ";";
          file << "bool " << boolean << ";";
          break;
        case fuzzing::FUZZ_SCALAR:
          scalar = get_next_mut_scalar();
          if (scalar.isFloatingPoint()) {
            std::cout << "Scalar " << scalar.to<double>() << ";";
            file << "Scalar " << scalar.to<double>() << ";";
          }
          else if (scalar.isIntegral(false)) {
            std::cout << "Scalar " << scalar.to<int64_t>() << ";";
            file << "Scalar " << scalar.to<int64_t>() << ";";
          } else if (scalar.isBoolean()) {
            std::cout << "Scalar " << scalar.to<bool>() << ";";
            file << "Scalar " << scalar.to<bool>() << ";";
          }
          break;
        case fuzzing::FUZZ_SCALARTYPE:
          scalartype = get_next_mut_scalartype();
          std::cout << "ScalarType " << scalartype << ";";
          file << "ScalarType " << scalartype << ";";
          break;
        case fuzzing::FUZZ_TENSOR:
          tensor = get_next_mut_tensor();
          contents = get_tensor_contents();
          std::cout << "Tensor " << "\n";
          std::cout << "Contents: " << contents << "\n";
          std::cout << "Sizes: ";
          for (auto &sz : tensor.sizes()) {
            std::cout << sz << ", ";
          }
          std::cout << "\n";
          std::cout << "Dtype: " << tensor.dtype() << "\n";
          std::cout << "Device: " << tensor.device() << "\n";
          std::cout << "Requires grad: " << tensor.requires_grad();
          std::cout << ";";
          file << "Tensor " << "\n";
          file << "Contents: " << contents << "\n";
          file << "Sizes: ";
          for (auto &sz : tensor.sizes()) {
            file << sz << ", ";
          }
          file << "\n";
          file << "Dtype: " << tensor.dtype() << "\n";
          file << "Device: " << tensor.device() << "\n";
          file << "Requires grad: " << tensor.requires_grad();
          file << ";";
          break;
        case fuzzing::FUZZ_SPARSE_TENSOR:
          sparse_tensor = get_next_mut_sparse_tensor();
          std::cout << "SparseTensor " << sparse_tensor << ";";
          file << "SparseTensor " << sparse_tensor << ";";
          break;
        case fuzzing::FUZZ_TENSOR_OPTIONS:
          tensor_opts = get_next_mut_tensor_options();
          std::cout << "TensorOptions " << tensor_opts << ";";
          file << "TensorOptions " << tensor_opts << ";";
          break;
        case fuzzing::FUZZ_INTARRAY_REF:
          intarrayref = get_next_mut_intarrayref();
          std::cout << "IntArrayRef " << intarrayref << ";";
          file << "IntArrayRef " << intarrayref << ";";
          break;
        case fuzzing::FUZZ_DOUBLEARRAYREF:
          doublearrayref = get_next_mut_doublearrayref();
          std::cout << "ArrayRef<double> " << doublearrayref << ";";
          file << "ArrayRef<double> " << doublearrayref << ";";
          break;
        case fuzzing::FUZZ_BOOLARRAY:
          boolarray = get_next_mut_boolarray();
          a = boolarray.at(0);
          b = boolarray.at(1);
          c = boolarray.at(2);
          std::cout << "std::array<bool,3> " << a << b << c << ";";
          file << "std::array<bool,3> " << a << b << c << ";";
          break;
        case fuzzing::FUZZ_STRING:
          string = get_next_mut_string();
          std::cout << "String " << string << ";";
          file << "String " << string << ";";
          break;
        case fuzzing::FUZZ_C10OPTIONAL_TENSOR:
          tensor_opt = get_next_mut_c10opt_tensor();
          if (!tensor_opt.has_value()) {
            file << "OptionalTensor " << "\n";
            file << "nullopt;";
            break;
          }
          tensor = c10::value_or_else(tensor_opt, [] {return at::Tensor();});
          std::cout << "OptionalTensor " << ";\n";
          file << "OptionalTensor " << "\n";
          file << "Contents: " << get_tensor_contents() << "\n";
          file << "Sizes: ";
          for (auto &sz : tensor.sizes()) {
            file << sz << ", ";
          }
          file << "\n";
          file << "Dtype: " << tensor.dtype() << "\n";
          file << "Device: " << tensor.device() << "\n";
          file << "Requires grad: " << tensor.requires_grad();
          file << ";";
          break;
        case fuzzing::FUZZ_C10OPTIONAL_SCALAR:
          scalar_opt = get_next_mut_c10opt_scalar();
          scalar = scalar_opt.value();
          std::cout << "OptionalScalar " << scalar.toLong() << ";";
          file << "OptionalScalar " << scalar.toLong() << ";";
          break;
        case fuzzing::FUZZ_C10OPTIONAL_INT:
          int_opt = get_next_mut_c10opt_int();
          integer = int_opt.value();
          std::cout << "OptionalInt " << integer << ";";
          file << "OptionalInt " << integer << ";";
          break;
        case fuzzing::FUZZ_C10OPTIONAL_LONG:
          long_opt = get_next_mut_c10opt_long();
          longint = long_opt.value();
          std::cout << "OptionalLong " << longint << ";";
          file << "OptionalLong " << longint << ";";
          break;
        case fuzzing::FUZZ_C10OPTIONAL_INTARRAYREF:
          intarrayref_opt = get_next_mut_c10opt_intarrayref();
          intarrayref = intarrayref_opt.value();
          std::cout << "OptionalIntArrayRef " << intarrayref << ";";
          file << "OptionalIntArrayRef " << intarrayref << ";";
          break;
        case fuzzing::FUZZ_C10OPTIONAL_DOUBLEARRAYREF:
          doublearrayref_opt = get_next_mut_c10opt_doublearrayref();
          doublearrayref = *doublearrayref_opt->data();
          std::cout << "OptionalArrayRef<double> " << doublearrayref << ";";
          file << "OptionalArrayRef<double> " << doublearrayref << ";";
          break;
        case fuzzing::FUZZ_C10OPTIONAL_BOOL:
          bool_opt = get_next_mut_c10opt_bool();
          boolean = bool_opt.value();
          std::cout << "OptionalBool " << boolean << ";";
          file << "OptionalBool " << boolean << ";";
          break;
        case fuzzing::FUZZ_C10OPTIONAL_DOUBLE:
          double_opt = get_next_mut_c10opt_double();
          doublenum = double_opt.value();
          std::cout << "OpiontalDouble " << doublenum << ";";
          file << "OpiontalDouble " << doublenum << ";";
          break;
        case fuzzing::FUZZ_C10OPTIONAL_STRING:
          string_opt = get_next_mut_c10opt_string();
          string = string_opt.value();
          std::cout << "OptionalString " << string << ";";
          file << "OptionalString " << string << ";";
          break;
        case fuzzing::FUZZ_C10OPTIONAL_SCALARTYPE:
          scalartype_opt = get_next_mut_c10opt_scalartype();
          scalartype = scalartype_opt.value();
          std::cout << "OptionalScalarType " << scalartype << ";";
          file << "OptionalScalarType " << scalartype << ";";
          break;
        default:
          break;
      }
    }

    file << "\n--------------------------------------" << std::endl;
  }

  void Fuzzer::increase_num_crashes()
  {

    long long num_crashes = 0; // Used to bound number of crashes
    struct stat stat_buffer = {};
    std::ios_base::openmode fflags = std::ios::out | std::ios::in;
    std::string crashes_num_filename;
    std::fstream run_file;
    std::string run_filename;
    std::string last_line;

    char logbuf[LOGBUFSZ];
    memset(logbuf, 0, LOGBUFSZ);

    crashes_num_filename = std::string(results_dir) + "/" + cur_fname + "_crashes_num.log";

    if (stat(crashes_num_filename.c_str(), &stat_buffer) == 0){
      num_crashes_file.open(crashes_num_filename, fflags);
      getline(num_crashes_file, last_line);
      if (last_line.length() > 0) {
        num_crashes = std::stoll(last_line);
      } else {
        std::cout << "Error while reading file with number of crashes..." << std::flush;
      }
    } else {
      num_crashes = 0;
      fflags |= std::ios::trunc;
      create_file(crashes_num_filename, num_crashes_file,  fflags);
    }
    num_crashes++;

    num_crashes_file.seekp(0, std::ios::beg);
    num_crashes_file << num_crashes;
    num_crashes_file.flush();
    /* num_crashes_file.close(); */

    if (num_crashes >= CRASHES_BOUND) {
      std::cout << "Function " << cur_fname << " crashed " << CRASHES_BOUND << " times, skipping rest of fuzzing" << std::endl;

      run_filename = std::string(results_dir) + "/" + cur_fname + ".run";
      create_file(run_filename, run_file, fflags);

      sprintf(logbuf, "%llu", total_mutations);
      run_file.write(logbuf, LOGBUFSZ);
      run_file << std::flush;
      run_file.close();
      mark_fuzzing_done();
    }

    return;
  }

  void Fuzzer::restore_last_mutation(long long last_mutation, long long last_timestamp, bool do_resume)
  {

    std::string crashes_filename;
    std::string crash_found_filename;

    /*
     * Handle the case where mutations were already done for this test
     * by just giving back one mutation so that the test doesn't crash
     */
    if (last_mutation < 0) {
      total_mutations = 1;
      return;
    }

    std::cout << "Resuming from mutation " << last_mutation << std::endl;
    if (!zero_muts_crashed(cur_fname)) {
      while (total_mutations != last_mutation) {
        if (total_mutations < last_mutation) {
          std::cout << "\033[1;31mError: didn't match last mutation, aborting\n\033[0m " << cur_fname << std::endl << std::flush;
          mark_fuzzing_done();
          return;
        }
        next_mutations_indices(false);
      }
    } else {
      main_pool_done = true;
      total_mutations = last_mutation;
    }

    /* If we weren't killed, also log the crash */
    if (!do_resume) {
      crashes_filename = std::string(results_dir) + "/" + cur_fname + "_crashes.log";
      crashes_logger_filename = crashes_filename;
      crashes_file.open(crashes_logger_filename, std::ios::out | std::ios::app);
      crashes_file.rdbuf()->pubsetbuf(nullptr, 0);
      log_current_mutation(crashes_file);

      crash_found_filename = std::string(results_dir) + "/" + cur_fname + ".crash_found";
      create_file(crash_found_filename, crash_found_file, std::ios::out | std::ios::app);
      /* Log crash time in seconds */
      crash_found_file << last_timestamp << std::endl;
      crash_found_file.close();
    }

    increase_num_crashes();

    next_mutations_indices(true);
    std::cout << "Mutations left: " << total_mutations << std::endl;
  }

  void Fuzzer::calculate_total_mutations() {

    fuzzing::TorchType type_enum;
    long long nmut_fuzz;
    int overflow = 0;

    for (auto &type : func_types) {

      switch (type) {
        case fuzzing::FUZZ_INT:
        case fuzzing::FUZZ_C10OPTIONAL_INT:
          overflow |= __builtin_smulll_overflow(total_mutations, int_mutations.size(), &total_mutations);
          pool_sizes.push_back(int_mutations.size());
          break;
        case fuzzing::FUZZ_LONG:
        case fuzzing::FUZZ_C10OPTIONAL_LONG:
          overflow |= __builtin_smulll_overflow(total_mutations, long_mutations.size(), &total_mutations);
          pool_sizes.push_back(long_mutations.size());
          break;
        case fuzzing::FUZZ_FLOAT:
        case fuzzing::FUZZ_DOUBLE:
        case fuzzing::FUZZ_C10OPTIONAL_DOUBLE:
          overflow |= __builtin_smulll_overflow(total_mutations, double_mutations.size(), &total_mutations);
          pool_sizes.push_back(double_mutations.size());
          break;
        case fuzzing::FUZZ_BOOLEAN:
        case fuzzing::FUZZ_C10OPTIONAL_BOOL:
          overflow |= __builtin_smulll_overflow(total_mutations, bool_mutations.size(), &total_mutations);
          pool_sizes.push_back(bool_mutations.size());
          break;
        case fuzzing::FUZZ_SCALAR:
        case fuzzing::FUZZ_C10OPTIONAL_SCALAR:
          overflow |= __builtin_smulll_overflow(total_mutations, scalar_mutations.size(), &total_mutations);
          pool_sizes.push_back(scalar_mutations.size());
          break;
        case fuzzing::FUZZ_SCALARTYPE:
        case fuzzing::FUZZ_C10OPTIONAL_SCALARTYPE:
          overflow |= __builtin_smulll_overflow(total_mutations, scalar_types.size(), &total_mutations);
          pool_sizes.push_back(scalar_types.size());
          break;
        case fuzzing::FUZZ_TENSOR:
        case fuzzing::FUZZ_C10OPTIONAL_TENSOR:
          overflow |= __builtin_smulll_overflow(total_mutations, tensor_mutations.size(), &total_mutations);
          pool_sizes.push_back(tensor_mutations.size());
          break;
        case fuzzing::FUZZ_SPARSE_TENSOR:
          overflow |= __builtin_smulll_overflow(total_mutations, sparse_tensor_mutations.size(), &total_mutations);
          pool_sizes.push_back(sparse_tensor_mutations.size());
          break;
        case fuzzing::FUZZ_TENSOR_OPTIONS:
          overflow |= __builtin_smulll_overflow(total_mutations, tensor_options_mutations.size(), &total_mutations);
          pool_sizes.push_back(tensor_options_mutations.size());
          break;
        case fuzzing::FUZZ_INTARRAY_REF:
        case fuzzing::FUZZ_C10OPTIONAL_INTARRAYREF:
          overflow |= __builtin_smulll_overflow(total_mutations, intarrayref_mutations.size(), &total_mutations);
          pool_sizes.push_back(intarrayref_mutations.size());
          break;
        case fuzzing::FUZZ_DOUBLEARRAYREF:
        case fuzzing::FUZZ_C10OPTIONAL_DOUBLEARRAYREF:
          overflow |= __builtin_smulll_overflow(total_mutations, doublearrayref_mutations.size(), &total_mutations);
          pool_sizes.push_back(doublearrayref_mutations.size());
          break;
        case fuzzing::FUZZ_BOOLARRAY:
          overflow |= __builtin_smulll_overflow(total_mutations, bool_arrays.size(), &total_mutations);
          pool_sizes.push_back(bool_arrays.size());
          break;
        case fuzzing::FUZZ_STRING:
        case fuzzing::FUZZ_C10OPTIONAL_STRING:
          overflow |= __builtin_smulll_overflow(total_mutations, string_mutations.size(), &total_mutations);
          pool_sizes.push_back(string_mutations.size());
          break;
        case fuzzing::FUZZ_TENSOR_LIST:
        case fuzzing::FUZZ_DIMNAME:
        case fuzzing::FUZZ_DIMNAME_LIST:
        case fuzzing::FUZZ_LAYOUT:
        case fuzzing::FUZZ_DEVICE:
        default:
          break;
      }
    }

    if (overflow != 0) {
      std::cout << "Total mutations for "  << cur_fname << " overflowed, maxing out at bound" << std::endl;
      total_mutations = NMUT_UPPER_BOUND_MID * 2;
      std::string overflow_filename = std::string(results_dir) + "/" + cur_fname + ".overflow";
      create_file(overflow_filename, overflow_file, std::ios::out | std::ios::in | std::ios::trunc);
    }

    nmut_fuzz = total_mutations;
    num_mut_skip = 1;

    /* Try at most UPPER_BOUND_MID * 2 mutations */
    if (nmut_fuzz > NMUT_UPPER_BOUND_MID) {
      nmut_fuzz = NMUT_UPPER_BOUND_MID;
      num_mut_skip = total_mutations / nmut_fuzz;
    }

    all_mutations = total_mutations;
    std::cout << cur_fname << ": Total mutations: " << total_mutations << std::endl;
    /* std::cout << "Will run with (at least): " << nmut_fuzz << " mutations" << std::endl; */
    std::cout << "Nmut skip: " << num_mut_skip << std::endl;

    /* To avoid off by one on first mutation */
    total_mutations += num_mut_skip;

    for (auto &ndims : tensor_dims) {
      zero_dim_mutations += ndims.size();
    }

    extra_tensor_mutations.reserve(zero_dim_mutations);

    int extra_intarr_muts = 0;
    for (auto &sz : intarrayref_sizes_vec) {
      zero_dim_mutations += sz;
      extra_intarr_muts += sz;
    }

    extra_intarrayref_mutations.reserve(extra_intarr_muts);

    std::cout << "Zero dim mutations: " << zero_dim_mutations << std::endl;

  }

  /* Creates all the tensor mutations */
  void Fuzzer::initialize_tensor_pool(){

    /* std::cout << "Creating tensor pool" << std::endl; */

    at::Tensor tensor;
    at::IntArrayRef tdim;
    at::ScalarType ttype;
    std::vector<int64_t> fuzz_dims_vec = {};
    int64_t rand_dim = 0;
    long fuzzval = 0;
    double dfuzzval = 0;

    /* Random generators */
    std::mt19937 rngenerator(RAND_SEED);
    std::uniform_int_distribution<> dims_distr(0, MAX_TENSOR_DIMS_FUZZ);
    std::uniform_int_distribution<> long_distr(0, long_mutations.size() - 1);
    std::uniform_int_distribution<> double_distr(0, double_mutations.size() - 1);
    std::uniform_int_distribution<> flip(0, 1);

    auto options = c10::TensorOptions();

    /*
     * Create tensors with the same dimensions as the originals:
     * - One same type as original one full of 1s
     * - One double one full of value 0.5
     */
    /* std::cout << "Creating tensors with same dims as originals" << std::endl; */
    for (int i = 0; i < tensor_dims.size(); i++) {
      /* std::cout << "Original type" << std::endl; */
      tdim = tensor_dims.at(i);
      ttype = original_tensor_types.at(i);
      options = c10::TensorOptions()
        .device(tensor_dev)
        .dtype(ttype);
      tensor = at::ones(tdim, options);
      if (have_mkldnn_tensors && (ttype == c10::kFloat || ttype == c10::kBFloat16)) {
        tensor = tensor.to_mkldnn();
      }
      tensor_mutations.push_back(tensor);
      tensor_contents.push_back(1);

      /* std::cout << "Double type" << std::endl; */
      options = c10::TensorOptions()
        .device(tensor_dev)
        .dtype(c10::kDouble);
      tensor = at::full(tdim, 0.5, options);
      tensor_mutations.push_back(tensor);
      tensor_contents.push_back(0.5);
    }

    /*
     * Long and double tensors with increasingly more random-sized dimensions
     * and containing random values
     */
    /* std::cout << "Creating increasing size tensors" << std::endl; */
    for (int i = 0; i < TENSOR_NUM_DIMS_FUZZ; i++) {
      fuzzval = long_mutations.at(long_distr(rngenerator));
      /* fuzzval = long_mutations.at(rnd_idx++ % long_mutations.size()); */
      /* std::cout << fuzzval << std::endl; */
      options = c10::TensorOptions()
        .device(tensor_dev)
        .dtype(c10::kLong);
      tensor = at::full(fuzz_dims_vec, fuzzval, options);
      tensor_mutations.push_back(tensor);
      tensor_contents.push_back(fuzzval);

      dfuzzval = double_mutations.at(double_distr(rngenerator));
      /* dfuzzval = double_mutations.at(rnd_idx++ % double_mutations.size()); */
      /* std::cout << dfuzzval << std::endl; */
      options = c10::TensorOptions()
        .device(tensor_dev)
        .dtype(c10::kDouble);
        /* .requires_grad(true); */
      tensor = at::full(fuzz_dims_vec, dfuzzval, options);
      tensor_mutations.push_back(tensor);
      tensor_contents.push_back(dfuzzval);

      /* Turn some dims to 0 */
      for (int j = 0; j < TENSOR_DIM_SIZE_FUZZ; j++) {
        rand_dim = dims_distr(rngenerator);
        if (flip(rngenerator)) {
          fuzz_dims_vec.push_back(rand_dim);
        } else {
          fuzz_dims_vec.push_back(0);
        }
      }
    }

    // Two deep tensors
    /* std::cout << "Creating deep tensors" << std::endl; */
    fuzz_dims_vec = {};
    for (int cur_ndims = 0; cur_ndims < MEDIUM_TENSOR_DIMS_FUZZ; cur_ndims++) {
      fuzz_dims_vec.push_back(1);
    }
    options = c10::TensorOptions()
      .device(tensor_dev)
      .dtype(c10::kLong);
    fuzzval = long_mutations.at(long_distr(rngenerator));
    tensor = at::full(fuzz_dims_vec, fuzzval, options);
    tensor_mutations.push_back(tensor);
    tensor_contents.push_back(fuzzval);
    options = c10::TensorOptions()
      .device(tensor_dev)
      .dtype(c10::kDouble);
    tensor = at::full(fuzz_dims_vec, LARGE_FLOAT_FUZZ, options);
    tensor_mutations.push_back(tensor);
    tensor_contents.push_back((double) LARGE_FLOAT_FUZZ);

    // Large double tensor
    /* std::cout << "Creating large double tensor" << std::endl; */
    fuzz_dims_vec = {};
    for (int i = 0; i < TENSOR_NUM_DIMS_FUZZ; i++) {
      fuzz_dims_vec.push_back(TENSOR_DIM_SIZE_FUZZ);
    }
    options = c10::TensorOptions()
      .dtype(c10::kDouble)
      .layout(c10::kStrided)
      .device(tensor_dev)
      /* .requires_grad(flip(rngenerator) ? true : false); */
      .requires_grad(true);
    tensor = at::full(fuzz_dims_vec, LARGE_FLOAT_FUZZ ,options);
    tensor_mutations.push_back(tensor);
    tensor_contents.push_back( (double) LARGE_FLOAT_FUZZ);

    /* std::cout << "Created tensor pool" << std::endl; */
  }

  void Fuzzer::initialize_sparse_tensor_pool(){

    /* std::cout << "Creating sparse tensor pool" << std::endl; */

    at::Tensor sparse_tensor;
    std::vector<int64_t> fuzz_dims_vec = {};

    // Random generators
    std::mt19937 rngenerator(RAND_SEED);
    std::uniform_int_distribution<> dims_distr(0, MAX_TENSOR_DIMS_FUZZ);
    std::uniform_int_distribution<> flip(0, 1);

    c10::TensorOptions sparse_opts = c10::TensorOptions()
      .dtype(c10::kFloat)
      .layout(c10::kSparse)
      .device(tensor_dev)
      .requires_grad(false);

    // Create sparse tensors with the same dimensions as the originals
    for (auto &tdim : sparse_tensor_dims) {
      sparse_tensor = at::sparse_coo_tensor(tdim, sparse_opts);
      sparse_tensor_mutations.push_back(sparse_tensor);
    }

    for (int i = 0; i < TENSOR_NUM_DIMS_FUZZ; i++) {
      sparse_tensor = at::sparse_coo_tensor(fuzz_dims_vec, sparse_opts);
      sparse_tensor_mutations.push_back(sparse_tensor);
      for (int j = 0; j < TENSOR_DIM_SIZE_FUZZ; j++) {
        fuzz_dims_vec.push_back(2);
      }
    }

  }

  void Fuzzer::initialize_tensor_options_pool(){

    at::TensorOptions tensor_opts;
    std::mt19937 rngenerator(RAND_SEED);
    std::uniform_int_distribution<> flip(0, 1);

    for (auto &dtype : original_tensor_types) {
      tensor_opts = at::TensorOptions()
        .dtype(dtype)
        .layout(c10::kStrided)
        .device(tensor_dev)
        .requires_grad(flip(rngenerator) ? true : false);
      tensor_options_mutations.push_back(tensor_opts);
    }
  }

  void Fuzzer::initialize_scalar_pool() {

    at::Scalar scalar;

    for (auto &l : long_mutations) {
      scalar = at::Scalar(l);
      scalar_mutations.push_back(scalar);
    }

    for (auto &d : double_mutations) {
      scalar = at::Scalar(d);
      scalar_mutations.push_back(scalar);
    }

  }

  void Fuzzer::initialize_intarrayref_pool(){

    long int *data;
    at::IntArrayRef arr_ref;
    std::mt19937 rngenerator(RAND_SEED);
    std::uniform_int_distribution<> long_distr(0, long_mutations.size() - 1);
    std::uniform_int_distribution<> flip(0, 1);
    long fuzzval = 0;

    /* Same sizes as originals, all long values */
    for (auto &sz : intarrayref_sizes) {
      for (auto &fuzzval : long_mutations) {
        data = new long int[sz];
        for (int l = 0; l < sz; l++) {
          data[l] = fuzzval;
        }
        arr_ref = *(new at::IntArrayRef(data, sz));
        intarrayref_mutations.push_back(arr_ref);
      }
    }

    /* Arrays of increasing dimensions, random values, some 0 */
    for (int dim = 0; dim < MAX_TENSOR_DIMS_FUZZ; dim += TENSOR_NUM_DIMS_FUZZ) {
      fuzzval = long_mutations.at(long_distr(rngenerator));
      data = new long int[dim];
      for (int l = 0; l < dim; l++) {
        if (flip(rngenerator)) {
          data[l] = fuzzval;
        } else {
          data[l] = 0;
        }
      }
      arr_ref = *(new at::IntArrayRef(data, dim));
      intarrayref_mutations.push_back(arr_ref);
    }

  }

  void Fuzzer::initialize_doublearrayref_pool() {

    double *data;
    at::ArrayRef<double> arr_ref;

    std::mt19937 rngenerator(RAND_SEED);
    std::uniform_int_distribution<> flip(0, 1);

    for (auto &sz : doublearrayref_sizes) {
      for (auto &fuzzval : double_mutations) {
        data = new double[sz];
        for (int l = 0; l < sz; l++) {
          data[l] = fuzzval;
        }
        arr_ref = *(new at::ArrayRef<double>(data, sz));
        doublearrayref_mutations.push_back(arr_ref);
      }
    }

    for (auto &fuzzval : double_mutations) {
      data = new double[ARRAYREF_LEN];
      for (int l = 0; l < ARRAYREF_LEN; l++) {
        if (flip(rngenerator)) {
          data[l] = fuzzval;
        } else {
          data[l] = 0.0;
        }
      }
      arr_ref = *(new at::ArrayRef<double>(data, ARRAYREF_LEN));
      doublearrayref_mutations.push_back(arr_ref);
    }
  }

  void Fuzzer::initialize_boolarrays() {
    std::array<bool,3> boolarray;
    for (int a = 0; a <= 1; a++) {
      for (int b = 0; b <= 1; b++) {
        for (int c = 0; c <= 1; c++) {
          boolarray = std::array<bool,3>{(bool)a, (bool)b, (bool)c};
          bool_arrays.push_back(boolarray);
        }
      }

    }
  }

  /* Get the next mutation indices and check if we are done fuzzing */
  bool Fuzzer::has_more_mutations(bool reset)
  {

    bool has_more;
    /*
     * Another thread/proc is fuzzing this right now,
     * return false such that the current thread doesn't
     * also try to fuzz
     */
    if (is_running)
      return false;

    has_more = total_mutations > 0;

    if (has_more && reset) {
      cur_idx = 0;
      next_mutations_indices(true);
    }

    if (!has_more) {
      if (!main_pool_done) {
        /* std::cout << "Main pool done for " << cur_fname << ", creating secondary pool" << std::endl << std::flush; */
        if (zero_dim_mutations == 0) {
          mark_fuzzing_done();
          std::remove(mutations_logger_filename.c_str());
          std::remove(timestamp_logger_filename.c_str());
        } else {
          std::ios_base::openmode fflags = std::ios::out | std::ios::in | std::ios::trunc;
          std::string zero_muts_filename = std::string(results_dir) + "/" + cur_fname + ".zero_muts";
          create_file(zero_muts_filename, zero_muts_file, fflags);
          has_more = true;
          main_pool_done = true;
          cur_idx = 0;
          /* Total mutations will be decreased by next_mutations_indices, so set
           * it to zero_dim_mutations here */
          total_mutations = zero_dim_mutations;
          /* Call this to log the first mutations number (needs to be called
           * with main_pool_done = false) */
          next_mutations_indices(true);
        }
      } else {
        mark_fuzzing_done();
        std::remove(mutations_logger_filename.c_str());
        std::remove(timestamp_logger_filename.c_str());
      }
    }

    return has_more;
  }

  /* Skips ahead num_mut_skip mutations to bound the total mutations */
  void Fuzzer::next_mutations_indices(bool log)
  {

    pid_t mypid = 0;
    std::string mut_filename;
    std::string last_timestamp_filename;
    std::ios_base::openmode fflags = std::ios::out | std::ios::in | std::ios::trunc;
    char logbuf[LOGBUFSZ];
    struct timespec ts = {};
    memset(logbuf, 0, LOGBUFSZ);

    if (!main_pool_done) {
      total_mutations -= num_mut_skip;

      long long passed = all_mutations - total_mutations;
      for (int i = 0; i < pool_sizes.size(); i++) {
        indices[i] = passed % pool_sizes[i];
        passed = passed / pool_sizes[i];
      }
    } else {
      total_mutations--;
    }

    if (log) {

      if (!mutations_file.is_open()) {
        mypid = ::getpid();
        mut_filename = std::string(results_dir) + "/" + cur_fname + "_mutations.log." + std::to_string(mypid);
        create_file(mut_filename, mutations_file, fflags);
      }
      mutations_file.seekp(0, std::ios::beg);
      sprintf(logbuf, "%llu", total_mutations);
      mutations_file.clear();
      mutations_file.write(logbuf, LOGBUFSZ);
      mutations_file.flush();

      if (!last_timestamp_file.is_open()) {
        mypid = ::getpid();
        last_timestamp_filename = std::string(results_dir) + "/" + cur_fname + ".last_timestamp." + std::to_string(mypid);
        create_file(last_timestamp_filename, last_timestamp_file, fflags);
      }

      last_timestamp_file.seekp(0, std::ios::beg);
      memset(logbuf, 0, LOGBUFSZ);
      clock_gettime(CLOCK_MONOTONIC, &ts);
      sprintf(logbuf, "%lld", (long long) ts.tv_sec);
      last_timestamp_file.write(logbuf, LOGBUFSZ);
      last_timestamp_file.flush();
    }
  }

  void Fuzzer::mark_unknown_type(std::string &ttype)
  {
    std::string filename;

    std::cout << "\033[1;31mUnknown type:\033[0m " << ttype << std::endl << std::flush;

    // Indicates a type that isn't handled in the fuzzer
    filename = std::string(results_dir) + "/" + cur_fname + ".unknown";

    std::ios_base::openmode fflags = std::ios::out | std::ios::in;
    unknown_type_file.open(filename, fflags);
    unknown_type_file << ttype << std::flush;
    unknown_type_file.close();

    abort();
  }

  void Fuzzer::mark_fuzzing_done()
  {
    std::string done_filename;
    int exists = 0;
    struct timespec ts = {};
    struct stat stat_buffer = {};
    std::ios_base::openmode fflags = std::ios::out | std::ios::in | std::ios::trunc;

    main_pool_done = true;
    /* This file indicates to the fuzzer that this function has been already fuzzed */
    done_filename = std::string(results_dir) + "/" + cur_fname + ".done";

    exists = stat(done_filename.c_str(), &stat_buffer) == 0;
    if (exists) {
      return;
    }

    create_file(done_filename, done_file, fflags);

    std::cout << cur_fname << ": finished fuzzing" << std::endl;

    /* Log end time in seconds */
    clock_gettime(CLOCK_MONOTONIC, &ts);
    done_file << ts.tv_sec << std::endl;
    done_file.close();

    /* Set mutations to zero to stop fuzzing */
    total_mutations = 0;
  }

  void Fuzzer::mut_start_time()
  {

    if (main_pool_done) {
      return;
    }

    clock_gettime(CLOCK_MONOTONIC, &start_time);
  }

  void Fuzzer::mut_end_time(bool failed)
  {

    struct timespec duration_ts = {};
    struct stat stat_buffer = {};
    std::ios_base::openmode fflags = std::ios::out | std::ios::in;
    /* std::string duration_filename; */
    /* std::fstream duration_file; */
    /* char logbuf[LOGBUFSZ]; */
    /* memset(logbuf, 0, LOGBUFSZ); */

    clock_gettime(CLOCK_MONOTONIC, &end_time);

    duration_ts = time_diff(start_time, end_time);
    int64_t duration = duration_ts.tv_sec * NS_PER_SEC + duration_ts.tv_nsec;

    /* Log successful and failed mutations in different files */
    if (!failed) {
      /* time_file.write(logbuf, LOGBUFSZ) << ":" << duration << std::endl << std::flush; */
      time_file << total_mutations << ":" << duration << std::endl << std::flush;
    } else {
      /* except_file.write(logbuf, LOGBUFSZ) << ":" << duration << std::endl << std::flush; */
      except_file << total_mutations << ":" << duration << std::endl << std::flush;
    }

    ///* Log mutations that took more than THRESH seconds to finish */
    //if (duration_secs > TIME_THRESH_SECS) {
    //  /*
    //   * We want to log the mutation that just run, so this will achieve it:
    //   * 1. Move to the mutation BEFORE the mutation that just run
    //   * 2. Call next_mutations_indices to set up the indices array correctly
    //   * for the mutation after that, which is the mutation that just run
    //   * 3. Call log_current_mutation to log it
    //   * 4. When the fuzzed function calls has_more_mutations,
    //   * next_mutations_indices will be called again and will move to the next
    //   * mutation
    //   */
    //  total_mutations += (num_mut_skip * 2);
    //  next_mutations_indices(false);

    //  duration_filename = std::string(results_dir) + "/" + cur_fname + ".duration";
    //  if (stat(duration_filename.c_str(), &stat_buffer) != 0) {
    //    fflags |= std::ios::trunc;
    //    create_file(duration_filename, duration_file, fflags);
    //  } else {
    //    duration_file.open(duration_filename, std::ios::app);
    //    if (duration_file.fail()) {
    //      std::cout << "Failed to open " << duration_filename << std::endl;
    //      std::cout << "Error: " << strerror(errno) << std::endl;
    //    }
    //  }

    //  duration_file << "Duration (secs): " << duration_secs << std::endl;
    //  log_current_mutation(duration_file);
    //  duration_file << std::flush;
    //  duration_file.close();
    //}

  }

  double Fuzzer::get_tensor_contents() {
    if (!main_pool_done) {
      return tensor_contents.at(indices[cur_idx - 1]);
    } else {
      return 1;
    }
  }

#endif


#if !defined(IVYSYN_COLLECT_TYPES)

  int Fuzzer::get_next_mut_int() {
#if defined(IVYSYN_VALIDATE)
  std::cout << "Returning int mutation at index " << indices[cur_idx] << std::endl;
  std::cout << "Int mutations size: " << int_mutations.size() << std::endl;
  int integer = int_mutations.at(indices[cur_idx]);
  std::cout << "int " << integer << ";" << std::flush;
  return int_mutations.at(indices[cur_idx++]);
#else
  if (!main_pool_done) {
    return int_mutations.at(indices[cur_idx++]);
  } else {
    return *(int *) original_args.at(cur_idx++);
  }
#endif
  }

  int64_t Fuzzer::get_next_mut_long() {
#if defined(IVYSYN_VALIDATE)
  std::cout << "Returning long mutation at index " << indices[cur_idx] << std::endl;
  std::cout << "long mutations size: " << long_mutations.size() << std::endl;
  long long_t = long_mutations.at(indices[cur_idx]);
  std::cout << "int64_t " << long_t << ";" << std::flush;
  return long_mutations.at(indices[cur_idx++]);
#else
  if (!main_pool_done) {
    return long_mutations.at(indices[cur_idx++]);
  } else {
    return *(long *) original_args.at(cur_idx++);
  }
#endif
  }

  bool Fuzzer::get_next_mut_bool() {
#if defined(IVYSYN_VALIDATE)
  std::cout << "Returning bool mutation at index " << indices[cur_idx] << std::endl;
  std::cout << "bool mutations size: " << bool_mutations.size() << std::endl;
  bool bool_t = bool_mutations.at(indices[cur_idx]);
  std::cout << "bool " << bool_t << ";" << std::flush;
  return bool_mutations.at(indices[cur_idx++]);
#else
  if (!main_pool_done) {
    return bool_mutations.at(indices[cur_idx++]);
  } else {
    return *(bool *) original_args.at(cur_idx++);
  }
#endif
  }

  double Fuzzer::get_next_mut_double() {
#if defined(IVYSYN_VALIDATE)
  std::cout << "Returning double mutation at index " << indices[cur_idx] << std::endl;
  std::cout << "double mutations size: " << double_mutations.size() << std::endl;
  double double_t = double_mutations.at(indices[cur_idx]);
  std::cout << "double " << double_t << ";" << std::flush;
  return double_mutations.at(indices[cur_idx++]);
#else
  if (!main_pool_done) {
    return double_mutations.at(indices[cur_idx++]);
  } else {
    return *(double *) original_args.at(cur_idx++);
  }
#endif
  }

  std::string Fuzzer::get_next_mut_string(){
#if defined(IVYSYN_VALIDATE)
  std::cout << "Returning string mutation at index " << indices[cur_idx] << std::endl;
  std::cout << "string mutations size: " << string_mutations.size() << std::endl;
  std::string string_t = string_mutations.at(indices[cur_idx]);
  std::cout << "string " << string_t << ";" << std::flush;
  return string_mutations.at(indices[cur_idx++]);
#else
  if (!main_pool_done) {
    return string_mutations.at(indices[cur_idx++]);
  } else {
    return *(std::string *) original_args.at(cur_idx++);
  }
#endif
  }

  at::IntArrayRef Fuzzer::get_next_mut_intarrayref() {
#if defined(IVYSYN_VALIDATE)
  std::cout << "Returning intarray mutation at index " << indices[cur_idx] << ": " << std::endl;
  std::cout << "intarray mutations size: " << intarrayref_mutations.size() << std::endl;
  at::IntArrayRef intarrayref = intarrayref_mutations.at(indices[cur_idx]);
  std::cout << "IntArrayRef " << intarrayref << ";" << std::flush;
  return intarrayref_mutations.at(indices[cur_idx++]);
#else
  if (!main_pool_done) {
    return intarrayref_mutations.at(indices[cur_idx++]);
  } else {
    int dim_idx = 0, arg_idx = 0;
    at::IntArrayRef orig_intarr;
    at::IntArrayRef zero_dim_intarr;
    at::Tensor orig_tensor;
    c10::optional<at::Tensor> opt_tensor;
    c10::optional<at::IntArrayRef> opt_intarr;
    int64_t *intarr_data;
    int idx = 0;
    for (auto orig_type : func_types) {
      if (orig_type == fuzzing::FUZZ_TENSOR || orig_type == fuzzing::FUZZ_C10OPTIONAL_TENSOR) {
        if (orig_type == fuzzing::FUZZ_TENSOR) {
          /* std::cout << "Tensor original arg" << std::endl; */
          orig_tensor = *((at::Tensor *) original_args.at(arg_idx));
        } else {
          /* std::cout << "Opt Tensor original arg" << std::endl; */
          opt_tensor = *(c10::optional<at::Tensor> *) original_args.at(arg_idx);
          if (opt_tensor.has_value()) {
            orig_tensor = opt_tensor.value();
          } else {
            arg_idx++;
            continue;
          }
        }
        if (orig_tensor.defined()) {
          dim_idx += orig_tensor.dim();
        }
      } else if (orig_type == fuzzing::FUZZ_INTARRAY_REF || orig_type == fuzzing::FUZZ_C10OPTIONAL_INTARRAYREF) {
        orig_intarr = *(at::IntArrayRef *) original_args.at(arg_idx);
        /* std::cout << "Orig intarray at idx " << arg_idx << " sizes: "; */
        /* for (auto n : orig_intarr) { */
        /*   std::cout << n << ", "; */
        /* } */
        /* std::cout << std::endl; */
        std::cout << "Dim idx before: " << dim_idx << std::endl;
        for (int tdim_idx = 0; tdim_idx < orig_intarr.size(); tdim_idx++) {
          std::cout << "Dim idx: " << dim_idx << std::endl;
          if (dim_idx == total_mutations &&
              arg_idx == cur_idx) {
            intarr_data = new int64_t[orig_intarr.size()];
            idx = 0;
            for (auto n : orig_intarr) {
              if (idx == tdim_idx) {
                intarr_data[idx++] = 0;
              } else {
                intarr_data[idx++] = n;
              }
            }
            zero_dim_intarr = at::IntArrayRef(intarr_data, orig_intarr.size());
            extra_intarrayref_mutations.push_back(zero_dim_intarr);
            std::cout << "Returning zero dim intarr with sizes:";
            for (auto n : zero_dim_intarr) {
              std::cout << n << ", ";
            }
            std::cout << std::endl;
            /* delete zero_dim_intarr; */
            cur_idx++;
            return extra_intarrayref_mutations.at(extra_intarrayref_mutations.size() - 1);
          }
          dim_idx++;
        }
      }
      arg_idx++;
    }
    orig_intarr = *((at::IntArrayRef *) original_args.at(cur_idx++));
    std::cout << "Returning original IntArrayRef from arg index " << cur_idx - 1 << std::endl;
    return orig_intarr;
  }
#endif
  }

  at::ArrayRef<double> Fuzzer::get_next_mut_doublearrayref() {
#if defined(IVYSYN_VALIDATE)
    return doublearrayref_mutations.at(indices[cur_idx++]);
#else
    if (!main_pool_done) {
      return doublearrayref_mutations.at(indices[cur_idx++]);
    } else {
      return *(at::ArrayRef<double> *) original_args.at(cur_idx++);
    }
#endif
  }

  at::Tensor Fuzzer::get_next_mut_tensor() {
#if defined(IVYSYN_VALIDATE)
    std::cout << "Returning tensor mutation at index " << indices[cur_idx] << std::endl;
    std::cout << "tensor mutations size: " << tensor_mutations.size() << std::endl;
    at::Tensor tensor = tensor_mutations.at(indices[cur_idx]);
    if (tensor.defined()) {
      double contents = tensor_contents.at(indices[cur_idx]);
      std::cout << "Tensor " << "\n";
      std::cout << "Contents: " << contents << "\n";
      std::cout << "Sizes: ";
      for (auto &sz : tensor.sizes()) {
        std::cout << sz << ", ";
      }
      std::cout << "\n";
      std::cout << "Dtype: " << tensor.dtype() << "\n";
      std::cout << "Device: " << tensor.device() << "\n";
      std::cout << "Requires grad: " << tensor.requires_grad();
      std::cout << ";" << std::flush;
    } else {
      std::cout << "Undefined tensor" << std::endl;
    }
    return tensor_mutations.at(indices[cur_idx++]);
#else
    if (!main_pool_done) {
      return tensor_mutations.at(indices[cur_idx++]);
    } else {
      /* std::cout << "Cur idx: " << cur_idx << std::endl; */
      int dim_idx = 0, arg_idx = 0;
      at::Tensor orig_tensor;
      at::TensorOptions tensor_opts;
      at::Tensor zero_dim_tensor;
      at::IntArrayRef zero_dims;
      at::IntArrayRef orig_tensor_dims;
      c10::optional<at::Tensor> opt_tensor;
      c10::optional<at::IntArrayRef> opt_intarr;
      c10::ScalarType dtype;
      int64_t *intarr_data;
      int idx;
      for (auto orig_type : func_types) {
        if (orig_type == fuzzing::FUZZ_INTARRAY_REF) {
          dim_idx += ((at::IntArrayRef *) original_args.at(arg_idx))->size();
        } else if (orig_type == fuzzing::FUZZ_C10OPTIONAL_INTARRAYREF) {
          opt_intarr = *((c10::optional<at::IntArrayRef> *) original_args.at(arg_idx));
          if (opt_intarr) {
            dim_idx += opt_intarr.value().size();
          }
        } else if (orig_type == fuzzing::FUZZ_TENSOR || orig_type == fuzzing::FUZZ_C10OPTIONAL_TENSOR) {
          if (orig_type == fuzzing::FUZZ_TENSOR) {
            orig_tensor = *((at::Tensor *) original_args.at(arg_idx));
          } else {
            opt_tensor = *(c10::optional<at::Tensor> *) original_args.at(arg_idx);
            if (opt_tensor.has_value()) {
              orig_tensor = opt_tensor.value();
            } else {
              arg_idx++;
              continue;
            }
          }
          if (!orig_tensor.defined()) {
            arg_idx++;
            continue;
          }
          for (int tdim_idx = 0; tdim_idx < orig_tensor.dim(); tdim_idx++) {
            if (dim_idx == total_mutations &&
                arg_idx == cur_idx &&
                orig_tensor.dim() > 1) {
              orig_tensor_dims = orig_tensor.sizes();
              intarr_data = new int64_t[orig_tensor_dims.size()];
              idx = 0;
              for (auto n : orig_tensor_dims) {
                if (idx == tdim_idx) {
                  intarr_data[idx++] = 0;
                } else {
                  intarr_data[idx++] = n;
                }
              }
              zero_dims = at::IntArrayRef(intarr_data, orig_tensor_dims.size());
              dtype = orig_tensor.scalar_type();
              zero_dim_tensor = at::ones(zero_dims, dtype);
              extra_tensor_mutations.push_back(zero_dim_tensor);
              cur_idx++;
              return extra_tensor_mutations.at(extra_tensor_mutations.size() - 1);
            }
            dim_idx++;
          }
        }
        arg_idx++;
      }
      orig_tensor = *((at::Tensor *) original_args.at(cur_idx++));
      return orig_tensor;
    }
#endif
  }

  at::Tensor Fuzzer::get_next_mut_sparse_tensor() {
#if defined(IVYSYN_VALIDATE)
    return sparse_tensor_mutations.at(indices[cur_idx++]);
#else
    if (!main_pool_done) {
      return sparse_tensor_mutations.at(indices[cur_idx++]);
    } else {
      return *(at::Tensor *) original_args.at(cur_idx++);
    }
#endif
  }

  at::TensorOptions Fuzzer::get_next_mut_tensor_options() {
#if defined(IVYSYN_VALIDATE)
    return tensor_options_mutations.at(indices[cur_idx++]);
#else
    if (!main_pool_done) {
      return tensor_options_mutations.at(indices[cur_idx++]);
    } else {
      return *(at::TensorOptions *) original_args.at(cur_idx++);
    }
#endif
  }

  at::Scalar Fuzzer::get_next_mut_scalar() {
#if defined(IVYSYN_VALIDATE)
  std::cout << "Returning scalar mutation at index " << indices[cur_idx] << std::endl;
  std::cout << "scalar mutations size: " << scalar_mutations.size() << std::endl;
  at::Scalar scalar = scalar_mutations.at(indices[cur_idx]);
  if (scalar.isFloatingPoint()) {
    std::cout << "Scalar " << scalar.to<double>() << ";" << std::flush;
  }
  else if (scalar.isIntegral(false)) {
    std::cout << "Scalar " << scalar.to<int64_t>() << ";" << std::flush;
  } else if (scalar.isBoolean()) {
    std::cout << "Scalar " << scalar.to<bool>() << ";" << std::flush;
  }
  return scalar_mutations.at(indices[cur_idx++]);
#else
  if (!main_pool_done) {
    return scalar_mutations.at(indices[cur_idx++]);
  } else {
    return *(at::Scalar *) original_args.at(cur_idx++);
  }
#endif
  }

  at::ScalarType Fuzzer::get_next_mut_scalartype() {
#if defined(IVYSYN_VALIDATE)
    return scalar_types.at(indices[cur_idx++]);
#else
    if (!main_pool_done) {
      return scalar_types.at(indices[cur_idx++]);
    } else {
      return *(at::ScalarType *) original_args.at(cur_idx++);
    }
#endif
  }

  std::array<bool,3> Fuzzer::get_next_mut_boolarray() {
#if defined(IVYSYN_VALIDATE)
    std::cout << "Returning boolarray mutation at index " << indices[cur_idx] << std::endl;
    std::cout << "std::array<bool,3>;" << std::flush;
    return bool_arrays.at(indices[cur_idx++]);
#else
    if (!main_pool_done) {
      return bool_arrays.at(indices[cur_idx++]);
    } else {
      return *(std::array<bool, 3> *) original_args.at(cur_idx++);
    }
#endif
  }

  c10::optional<at::Tensor> Fuzzer::get_next_mut_c10opt_tensor() {
#if defined(IVYSYN_VALIDATE)
    std::cout << "Returning tensor mutation at index " << indices[cur_idx] << std::endl;
    std::cout << "tensor mutations size: " << tensor_mutations.size() << std::endl;
    if (std::find(nullopt_indices.begin(), nullopt_indices.end(), cur_idx) != nullopt_indices.end()) {
      cur_idx++;
      return c10::nullopt;
    }
    at::Tensor tensor = tensor_mutations.at(indices[cur_idx]);
    if (tensor.defined()) {
      double contents = tensor_contents.at(indices[cur_idx]);
      std::cout << "Tensor " << "\n";
      std::cout << "Contents: " << contents << "\n";
      std::cout << "Sizes: ";
      for (auto &sz : tensor.sizes()) {
        std::cout << sz << ", ";
      }
      std::cout << "\n";
      std::cout << "Dtype: " << tensor.dtype() << "\n";
      std::cout << "Device: " << tensor.device() << "\n";
      std::cout << "Requires grad: " << tensor.requires_grad();
      std::cout << ";" << std::flush;
    } else {
      std::cout << "Undefined tensor" << std::endl;
    }
    return c10::make_optional(tensor_mutations.at(indices[cur_idx++]));
#else
    if (!main_pool_done) {
      return c10::make_optional(tensor_mutations.at(indices[cur_idx++]));
    } else {
      /* std::cout << "Cur idx: " << cur_idx << std::endl; */
      int dim_idx = 0, arg_idx = 0;
      at::Tensor orig_tensor;
      at::TensorOptions tensor_opts;
      at::Tensor zero_dim_tensor;
      at::IntArrayRef zero_dims;
      c10::optional<at::Tensor> opt_tensor;
      c10::optional<at::IntArrayRef> opt_intarr;
      c10::ScalarType dtype;
      if (std::find(nullopt_indices.begin(), nullopt_indices.end(), cur_idx) != nullopt_indices.end()) {
        /* std::cout << "Returning original opt tensor at idx " << cur_idx << std::endl; */
        opt_tensor = *(c10::optional<at::Tensor> *) original_args.at(cur_idx++);
        return opt_tensor;
      }
      /* else { */
      /*   std::cout << "Tensor at index " << cur_idx << " has value" << std::endl; */
      /* } */
      for (auto orig_type : func_types) {
        /* std::cout << "Arg idx: " << arg_idx << std::endl; */
        if (orig_type == fuzzing::FUZZ_INTARRAY_REF) {
          dim_idx += ((at::IntArrayRef *) original_args.at(arg_idx))->size();
        } else if (orig_type == fuzzing::FUZZ_C10OPTIONAL_INTARRAYREF) {
          opt_intarr = *((c10::optional<at::IntArrayRef> *) original_args.at(arg_idx));
          if (opt_intarr) {
            dim_idx += opt_intarr.value().size();
            /* std::cout << "Dim idx: " << dim_idx << std::endl; */
          }
        } else if (orig_type == fuzzing::FUZZ_TENSOR || orig_type == fuzzing::FUZZ_C10OPTIONAL_TENSOR) {
          if (orig_type == fuzzing::FUZZ_TENSOR) {
            /* std::cout << "Tensor original arg at idx " << arg_idx << std::endl; */
            orig_tensor = *(at::Tensor *) original_args.at(arg_idx);
          } else {
            /* std::cout << "Opt Tensor original arg at idx " << arg_idx << std::endl; */
            /* std::cout << "Opt Tensor original arg" << std::endl; */
            opt_tensor = *(c10::optional<at::Tensor> *) original_args.at(arg_idx);
            if (opt_tensor.has_value()) {
              orig_tensor = opt_tensor.value();
            } else {
              arg_idx++;
              continue;
            }
          }
          /* std::cout << "Test 1" << std::endl; */
          if (!orig_tensor.defined()) {
            /* std::cout << "Undefined tensor" << std::endl; */
            arg_idx++;
            continue;
          }
          /* std::cout << "Test 2" << std::endl; */
          /* std::cout << "Orig tensor dims: " << orig_tensor.dim() << std::endl; */
          for (int tdim_idx = 0; tdim_idx < orig_tensor.dim(); tdim_idx++) {
            /* std::cout << "tdim idx: " << tdim_idx << std::endl; */
            /* std::cout << "dim idx: " << dim_idx << std::endl; */
            /* std::cout << "cur idx: " << cur_idx << std::endl; */
            /* std::cout << "arg idx: " << arg_idx << std::endl; */
            /* std::cout << "total mutations: " << total_mutations << std::endl; */
            if (dim_idx == total_mutations &&
                arg_idx == cur_idx &&
                orig_tensor.dim() > 1) {
              /* std::cout << "Dim idx: " << dim_idx << std::endl; */
              std::vector<int64_t> zero_dims_vec = at::IntArrayRef(orig_tensor.sizes()).vec();
              /* std::cout << "Zero dims vec before: " << std::endl; */
              /* for (auto n : zero_dims_vec) { */
              /*   std::cout << n << ", "; */
              /* } */
              /* std::cout << std::endl; */
              /* std::cout << "Test 3" << std::endl; */
              zero_dims_vec[tdim_idx] = 0;
              /* std::cout << "Test 4" << std::endl; */
              /* std::cout << "Zero dims vec after: " << std::endl; */
              /* for (auto n : zero_dims_vec) { */
              /*   std::cout << n << ", "; */
              /* } */
              /* std::cout << std::endl; */
              zero_dims = at::IntArrayRef(zero_dims_vec);
              /* std::cout << "Zero dims intarray: " << std::endl; */
              for (auto n : zero_dims) {
                std::cout << n << ", ";
              }
              std::cout << std::endl;
              /* std::cout << "Test 5" << std::endl; */
              /* tensor_opts = at::TensorOptions() */
              /*   .dtype(orig_tensor.dtype()) */
              /*   .device(orig_tensor.device()) */
              /*   .requires_grad(orig_tensor.requires_grad()); */
              /* std::cout << "Test 6" << std::endl; */
              /* std::cout << "Orig tensor dtype: " << orig_tensor.dtype() << std::endl; */
              dtype = orig_tensor.scalar_type();
              zero_dim_tensor = at::ones(zero_dims, dtype);
              extra_tensor_mutations.push_back(zero_dim_tensor);
              /* std::cout << "Test 7" << std::endl; */
              /* /1* std::cout << "Zero dim tensor: " << zero_dim_tensor << std::endl; *1/ */
              cur_idx++;
              /* std::cout << "Returning zero dim tensor" << std::endl; */
              return c10::make_optional(extra_tensor_mutations.at(extra_tensor_mutations.size() - 1));
            }
            dim_idx++;
          }
          /* std::cout << "Checked tensor at idx " << arg_idx << std::endl; */
        }
        /* std::cout << "Test 8" << std::endl; */
        arg_idx++;
      }
      /* std::cout << "Test 9" << std::endl; */
      /* std::cout << "Returning original optional tensor from arg index " << cur_idx << std::endl; */
      opt_tensor = *(c10::optional<at::Tensor> *) original_args.at(cur_idx++);
      /* std::cout << "Test 10" << std::endl; */
      return opt_tensor;
    }
#endif
  }

  c10::optional<at::IntArrayRef> Fuzzer::get_next_mut_c10opt_intarrayref() {
#if defined(IVYSYN_VALIDATE)
  std::cout << "Returning intarray mutation at index " << indices[cur_idx] << std::endl;
  std::cout << "intarray mutations size: " << intarrayref_mutations.size() << std::endl;
  at::IntArrayRef intarrayref = intarrayref_mutations.at(indices[cur_idx]);
  std::cout << "IntArrayRef " << intarrayref << ";" << std::flush;
  return c10::make_optional(intarrayref_mutations.at(indices[cur_idx++]));
#else
  if (!main_pool_done) {
    return c10::make_optional(intarrayref_mutations.at(indices[cur_idx++]));
  } else {
    int dim_idx = 0, arg_idx = 0;
    at::IntArrayRef orig_intarr;
    at::IntArrayRef zero_dim_intarr;
    at::Tensor orig_tensor;
    c10::optional<at::Tensor> opt_tensor;
    c10::optional<at::IntArrayRef> opt_intarr;
    int64_t *intarr_data;
    int idx = 0;
    for (auto orig_type : func_types) {
      if (orig_type == fuzzing::FUZZ_TENSOR || orig_type == fuzzing::FUZZ_C10OPTIONAL_TENSOR) {
        if (orig_type == fuzzing::FUZZ_TENSOR) {
          /* std::cout << "Tensor original arg" << std::endl; */
          orig_tensor = *((at::Tensor *) original_args.at(arg_idx));
        } else {
          /* std::cout << "Opt Tensor original arg" << std::endl; */
          opt_tensor = *(c10::optional<at::Tensor> *) original_args.at(arg_idx);
          orig_tensor = c10::value_or_else(opt_tensor, [] {return at::Tensor();});
        }
        if (orig_tensor.defined()) {
          dim_idx += orig_tensor.dim();
        }
      } else if (orig_type == fuzzing::FUZZ_INTARRAY_REF) {
        dim_idx += ((at::IntArrayRef *) original_args.at(arg_idx))->size();
      } else if (orig_type == fuzzing::FUZZ_C10OPTIONAL_INTARRAYREF) {
        opt_intarr = *((c10::optional<at::IntArrayRef> *) original_args.at(arg_idx));
        if (opt_intarr) {
          orig_intarr = opt_intarr.value();
        } else {
          arg_idx++;
          continue;
        }
        for (int tdim_idx = 0; tdim_idx < orig_intarr.size(); tdim_idx++) {
          if (dim_idx == total_mutations && arg_idx == cur_idx) {
            intarr_data = new int64_t[orig_intarr.size()];
            idx = 0;
            for (auto n : orig_intarr) {
              if (idx == tdim_idx) {
                intarr_data[idx++] = 0;
              } else {
                intarr_data[idx++] = n;
              }
            }
            zero_dim_intarr = at::IntArrayRef(intarr_data, orig_intarr.size());
            extra_intarrayref_mutations.push_back(zero_dim_intarr);
            /* std::cout << "Returning zero dim intarr with sizes:"; */
            for (auto n : zero_dim_intarr) {
              std::cout << n << ", ";
            }
            std::cout << std::endl;
            cur_idx++;
            return c10::make_optional(extra_intarrayref_mutations.at(extra_intarrayref_mutations.size() - 1));
          }
          dim_idx++;
        }
      }
      arg_idx++;
    }
    /* std::cout << "IntArrayRef " << *(at::IntArrayRef *) original_args.at(cur_idx) << ";" << std::flush; */
    opt_intarr = *((c10::optional<at::IntArrayRef> *) original_args.at(cur_idx++));
    return opt_intarr;
  }
#endif
  }

  c10::optional<at::ArrayRef<double>> Fuzzer::get_next_mut_c10opt_doublearrayref() {
#if defined(IVYSYN_VALIDATE)
    return c10::make_optional(doublearrayref_mutations.at(indices[cur_idx++]));
#else
    if (!main_pool_done) {
      return c10::make_optional(doublearrayref_mutations.at(indices[cur_idx++]));
    } else {
      return c10::make_optional(*(at::ArrayRef<double> *) original_args.at(cur_idx++));
    }
#endif
  }

  c10::optional<int> Fuzzer::get_next_mut_c10opt_int() {
#if defined(IVYSYN_VALIDATE)
  std::cout << "Returning int mutation at index " << indices[cur_idx] << std::endl;
  std::cout << "Int mutations size: " << int_mutations.size() << std::endl;
  int integer = int_mutations.at(indices[cur_idx]);
  std::cout << "int " << integer << ";" << std::flush;
  return c10::make_optional(int_mutations.at(indices[cur_idx++]));
#else
  if (!main_pool_done) {
    return c10::make_optional(int_mutations.at(indices[cur_idx++]));
  } else {
    return c10::make_optional(*(int *) original_args.at(cur_idx++));
  }
#endif
  }

  c10::optional<int64_t> Fuzzer::get_next_mut_c10opt_long() {
#if defined(IVYSYN_VALIDATE)
  std::cout << "Returning long mutation at index " << indices[cur_idx] << std::endl;
  std::cout << "long mutations size: " << long_mutations.size() << std::endl;
  long long_t = long_mutations.at(indices[cur_idx]);
  std::cout << "int64_t " << long_t << ";" << std::flush;
  return c10::make_optional(long_mutations.at(indices[cur_idx++]));
#else
  if (!main_pool_done) {
    return c10::make_optional(long_mutations.at(indices[cur_idx++]));
  } else {
    return c10::make_optional(*(int64_t *) original_args.at(cur_idx++));
  }
#endif
  }

  c10::optional<double> Fuzzer::get_next_mut_c10opt_double() {
#if defined(IVYSYN_VALIDATE)
  std::cout << "Returning double mutation at index " << indices[cur_idx] << std::endl;
  std::cout << "double mutations size: " << double_mutations.size() << std::endl;
  double double_t = double_mutations.at(indices[cur_idx]);
  std::cout << "double " << double_t << ";" << std::flush;
  return c10::make_optional(double_mutations.at(indices[cur_idx++]));
#else
  if (!main_pool_done) {
    return c10::make_optional(double_mutations.at(indices[cur_idx++]));
  } else {
    return c10::make_optional(*(double *) original_args.at(cur_idx++));
  }
#endif
  }

  c10::optional<bool> Fuzzer::get_next_mut_c10opt_bool() {
#if defined(IVYSYN_VALIDATE)
  std::cout << "Returning bool mutation at index " << indices[cur_idx] << std::endl;
  std::cout << "bool mutations size: " << bool_mutations.size() << std::endl;
  bool bool_t = bool_mutations.at(indices[cur_idx]);
  std::cout << "bool " << bool_t << ";" << std::flush;
  return c10::make_optional((bool)bool_mutations.at(indices[cur_idx++]));
#else
  if (!main_pool_done) {
    return c10::make_optional((bool)bool_mutations.at(indices[cur_idx++]));
  } else {
    return c10::make_optional(*(bool *) original_args.at(cur_idx++));
  }
#endif
  }

  c10::optional<std::string> Fuzzer::get_next_mut_c10opt_string() {
#if defined(IVYSYN_VALIDATE)
  std::cout << "Returning string mutation at index " << indices[cur_idx] << std::endl;
  std::cout << "string mutations size: " << string_mutations.size() << std::endl;
  std::string string_t = string_mutations.at(indices[cur_idx]);
  std::cout << "string " << string_t << ";" << std::flush;
  return c10::make_optional(string_mutations.at(indices[cur_idx++]));
#else
  if (!main_pool_done) {
    return c10::make_optional(string_mutations.at(indices[cur_idx++]));
  } else {
    return c10::make_optional(*(std::string *) original_args.at(cur_idx++));
  }
#endif
  }

  c10::optional<at::Scalar> Fuzzer::get_next_mut_c10opt_scalar() {
#if defined(IVYSYN_VALIDATE)
  std::cout << "Returning scalar mutation at index " << indices[cur_idx] << std::endl;
  std::cout << "scalar mutations size: " << scalar_mutations.size() << std::endl;
  at::Scalar scalar = scalar_mutations.at(indices[cur_idx]);
  if (scalar.isFloatingPoint()) {
    std::cout << "Scalar " << scalar.to<double>() << ";" << std::flush;
  }
  else if (scalar.isIntegral(false)) {
    std::cout << "Scalar " << scalar.to<int64_t>() << ";" << std::flush;
  } else if (scalar.isBoolean()) {
    std::cout << "Scalar " << scalar.to<bool>() << ";" << std::flush;
  }
  return c10::make_optional(scalar_mutations.at(indices[cur_idx++]));
#else
  if (!main_pool_done) {
    return c10::make_optional(scalar_mutations.at(indices[cur_idx++]));
  } else {
    return c10::make_optional(*(at::Scalar *) original_args.at(cur_idx++));
  }
#endif
  }

  c10::optional<at::ScalarType> Fuzzer::get_next_mut_c10opt_scalartype() {
#if defined(IVYSYN_VALIDATE)
    return c10::make_optional(scalar_types.at(indices[cur_idx++]));
#else
    if (!main_pool_done) {
      return c10::make_optional(scalar_types.at(indices[cur_idx++]));
    } else {
      return c10::make_optional(*(at::ScalarType *) original_args.at(cur_idx++));
    }
#endif
  }

#endif

}
