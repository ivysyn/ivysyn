
#include "tensorflow/core/framework/fuzzing.h"

//#define IVYSYN_COLLECT_TYPES

namespace tffuzzing {

  const char *results_dir = "/mnt/tensorflow-ivysyn";

#if !defined(IVYSYN_VALIDATE) && !defined(IVYSYN_COLLECT_TYPES)
  bool already_fuzzing = false;
  const int TIMEOUT_SECS = 1200;
  const int RNG_SEED = 123;
  const int NMUT_UPPER_BOUND_MID = 1000000;
  const int CRASHES_BOUND = 1;
  const int TIME_THRESH_SECS = 30;
  std::string cur_fname_glob = {};

  static std::fstream mutations_file;
  static std::fstream mutations_restore;
  static std::fstream last_timestamp_file;
  static std::fstream timestamp_restore;
  static std::fstream crashes_file;
  static std::fstream num_crashes_file;
  static std::fstream unknown_type_file;
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
  static std::fstream gpu_file;
  static std::fstream cpu_file;
#endif

  void create_file(const std::string& filename, std::fstream &file, std::ios_base::openmode fflags)
  {
    std::ofstream file_stream(filename);
    if (file.is_open()) {
      file.close();
    }
    file.clear();
    file.open(filename, fflags);
    if (file.fail()) {
      std::cout << "Failed to open " << filename << std::endl;
      std::cout << "Error: " << strerror(errno) << std::endl;
    }
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

  bool was_fuzzed(const std::string& fname) {
    struct stat stat_buffer = {};
    std::string stat_filename;
    std::string unknown_filename;
    int done_status = 0;
    int unknown_status = 0;

    stat_filename = std::string(results_dir) + "/" + fname + ".done";
    unknown_filename = std::string(results_dir) + "/" + fname + ".unknown";

    done_status = stat(stat_filename.c_str(), &stat_buffer) == 0;
    unknown_status = stat(unknown_filename.c_str(), &stat_buffer) == 0;

    return done_status || unknown_status;
  }

  bool zero_muts_crashed(const std::string& fname) {
    struct stat stat_buffer = {};
    std::string zero_mut_filename;
    int done_status = 0;

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
    struct timespec ts = {};

    std::cout << "Kernel " << cur_fname_glob << " timed out, stopping fuzzing" << std::endl;
    std::string timeout_filename = std::string(results_dir) + "/" + cur_fname_glob + ".timeout";
    std::fstream timeout_file;

    create_file(timeout_filename, timeout_file, std::ios::out);

    clock_gettime(CLOCK_MONOTONIC, &ts);
    timeout_file << ts.tv_sec << std::endl;
    timeout_file.close();

    exit(-SIGALRM);
  }

#endif

#if defined(IVYSYN_COLLECT_TYPES)
  Fuzzer::Fuzzer(const std::string& fname, tensorflow::OpKernelContext* ctx, bool hasDevice, const char *device)
  {
    struct stat stat_buffer = {};
    std::string out_str;
    std::string attrs;
    std::string types_filename, gpu_filename, cpu_filename;
    tensorflow::Tensor tensor;
    tensorflow::DataType ttype;

    /* int num_args; */

    attrs = tensorflow::SummarizeAttrs(ctx->op_kernel().def()).c_str();
    num_args = ctx->num_inputs();

    types_filename = std::string(results_dir) + "/" + fname + ".types";
    cpu_filename = std::string(results_dir) + "/" + fname + ".cpu";
    gpu_filename = std::string(results_dir) + "/" + fname + ".gpu";

    std::ios_base::openmode fflags = std::ios::out | std::ios::in | std::ios::trunc;

    if (hasDevice) {
      //std::cout << fname << ": Device: " << device << std::endl;
      if (std::string(device).compare("N5Eigen16ThreadPoolDeviceE") == 0) {
        //std::cout << fname << ": Device: CPU" << std::endl;
        if (stat(cpu_filename.c_str(), &stat_buffer) != 0) {
          create_file(cpu_filename, cpu_file, fflags);
        }
      } else if (std::string(device).compare("N5Eigen9GpuDeviceE") == 0) {
        //std::cout << fname << ": Device: GPU" << std::endl;
        if (stat(gpu_filename.c_str(), &stat_buffer) != 0) {
          create_file(gpu_filename, gpu_file, fflags);
        }
      }
    }

    int exists = 0;
    exists = stat(types_filename.c_str(), &stat_buffer) == 0;

    if (exists) {
      return;
    }

    create_file(types_filename, types_file, fflags);

    out_str += std::string(attrs) + "\n";
    for (int i = 0; i < num_args; i++) {
      if (!ctx->has_input(i) || ctx->input_is_ref(i)) {
        out_str += "None/Ref\n";
      } else {
        tensor = ctx->input(i);
        ttype = tensor.dtype();
        if (tensor.dtype() == tensorflow::DataType::DT_RESOURCE) {
          out_str += "Resource\n";
        } else {
          out_str += tensor.DeviceSafeDebugString() + "\n";
        }
      }
    }
    out_str += "\n--------------------------------------\n";
    types_file << out_str << std::flush;
    types_file.close();
  }

#elif defined(IVYSYN_VALIDATE)

  Fuzzer::Fuzzer(const std::string& fname, tensorflow::OpKernelContext* ctx)
  {
    int exists_validate, exists_check, exists_true_pos, exists_false_pos;
    struct stat stat_buffer = {};

    /* std::cout << "In fuzzer for " << fname << std::endl; */

    cur_fname = fname;
    original_ctx = new tensorflow::OpKernelContext(ctx->get_params());
    original_inputs = original_ctx->get_params()->inputs;

    std::string validate_filename = std::string(results_dir) + "/" + cur_fname + ".validate";
    std::string check_filename = std::string(results_dir) + "/" + cur_fname + ".check";
    /* std::string killed_filename = std::string(results_dir) + "/" + cur_fname + ".killed"; */
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
  }

  bool Fuzzer::should_validate()
  {
    return _should_validate;
  }


  /* Parse the logged crash and recreate the context that caused the crash */
  tensorflow::OpKernelContext *Fuzzer::get_validate_context()
  {

    tensorflow::DataType ttype;
    tensorflow::OpKernelContext::Params *validate_ctx_params = original_ctx->get_params();
    std::vector<tensorflow::TensorValue> fuzz_vec;
    tensorflow::TensorValue *fuzz_tensval_ptr;
    tensorflow::OpKernelContext *validate_ctx = nullptr;

    tensorflow::TensorShape tensor_shape;

    std::vector<tensorflow::qint8> arg_qint8_vec;
    std::vector<tensorflow::qint16> arg_qint16_vec;
    std::vector<tensorflow::qint32> arg_qint32_vec;
    std::vector<tensorflow::quint8> arg_quint8_vec;
    std::vector<tensorflow::quint16> arg_quint16_vec;
    std::vector<tensorflow::int8> arg_int8_vec;
    std::vector<tensorflow::int16> arg_int16_vec;
    std::vector<tensorflow::int32> arg_int32_vec;
    std::vector<tensorflow::int64> arg_int64_vec;
    std::vector<tensorflow::uint8> arg_uint8_vec;
    std::vector<tensorflow::uint16> arg_uint16_vec;
    std::vector<tensorflow::uint32> arg_uint32_vec;
    std::vector<tensorflow::uint64> arg_uint64_vec;
    std::vector<float> arg_float_vec;
    std::vector<Eigen::half> arg_half_vec;
    std::vector<double> arg_double_vec;
    std::vector<bool> arg_bool_vec;
    std::vector<tensorflow::tstring> arg_string_vec;

    std::string contents_str;
    std::string shape_str;
    std::string type_str;

    std::string dim_str;
    std::string arg_str;

    int dim, dummy;
    long ldummy;

    std::string validate_filename = std::string(results_dir) + "/" + cur_fname + ".validate";
    std::ifstream validate_file(validate_filename);

    int inputs_read = 0;
    while (std::getline(validate_file, contents_str)) {
      inputs_read++;
      std::stringstream contents_sstream(contents_str);

      std::getline(validate_file, shape_str);
      std::stringstream shape_sstream(shape_str);
      tensor_shape = tensorflow::TensorShape();
      if (shape_str.compare("") != 0) {
        while (shape_sstream.good()) {
          std::getline(shape_sstream, dim_str, ',');
          dim = std::stoi(dim_str);
          /* std::cout << "Dim str: " << dim_str << std::endl; */
          /* std::cout << "Dim: " << dim << std::endl; */
          tensor_shape.AddDim(dim);
        }
      }

      std::getline(validate_file, type_str);
      if (type_str.compare("qint8") == 0) {
          ttype = tensorflow::DataType::DT_QINT8;
          arg_qint8_vec = {};
          if (contents_str.compare("[]") != 0) {
            while (contents_sstream.good()) {
              std::getline(contents_sstream, arg_str, ',');
              dummy = std::stoi(arg_str);
              arg_qint8_vec.push_back(tensorflow::qint8(dummy));
            }
          }
          fuzz_tensval_ptr = get_tensor_with_shape_and_multiple_values(arg_qint8_vec, ttype, tensor_shape);
          fuzz_vec.push_back(*fuzz_tensval_ptr);
      }
      else if (type_str.compare("qint16") == 0) {
          ttype = tensorflow::DataType::DT_QINT16;
          arg_qint16_vec = {};
          if (contents_str.compare("[]") != 0) {
            while (contents_sstream.good()) {
              std::getline(contents_sstream, arg_str, ',');
              dummy = std::stoi(arg_str);
              arg_qint16_vec.push_back(tensorflow::qint16(dummy));
            }
          }
          fuzz_tensval_ptr = get_tensor_with_shape_and_multiple_values(arg_qint16_vec, ttype, tensor_shape);
          fuzz_vec.push_back(*fuzz_tensval_ptr);
      }
      else if (type_str.compare("qint32") == 0) {
          ttype = tensorflow::DataType::DT_QINT32;
          arg_qint32_vec = {};
          if (contents_str.compare("[]") != 0) {
            while (contents_sstream.good()) {
              std::getline(contents_sstream, arg_str, ',');
              dummy = std::stoi(arg_str);
              arg_qint32_vec.push_back(tensorflow::qint32(dummy));
            }
          }
          fuzz_tensval_ptr = get_tensor_with_shape_and_multiple_values(arg_qint32_vec, ttype, tensor_shape);
          fuzz_vec.push_back(*fuzz_tensval_ptr);
      }
      else if (type_str.compare("quint8") == 0) {
          ttype = tensorflow::DataType::DT_QUINT8;
          arg_quint8_vec = {};
          if (contents_str.compare("[]") != 0) {
            while (contents_sstream.good()) {
              std::getline(contents_sstream, arg_str, ',');
              dummy = std::stoi(arg_str);
              arg_quint8_vec.push_back(tensorflow::quint8(dummy));
            }
          }
          fuzz_tensval_ptr = get_tensor_with_shape_and_multiple_values(arg_quint8_vec, ttype, tensor_shape);
          fuzz_vec.push_back(*fuzz_tensval_ptr);
      }
      else if (type_str.compare("quint16") == 0) {
          ttype = tensorflow::DataType::DT_QUINT16;
          arg_quint16_vec = {};
          if (contents_str.compare("[]") != 0) {
            while (contents_sstream.good()) {
              std::getline(contents_sstream, arg_str, ',');
              dummy = std::stoi(arg_str);
              arg_quint16_vec.push_back(tensorflow::quint16(dummy));
            }
          }
          fuzz_tensval_ptr = get_tensor_with_shape_and_multiple_values(arg_quint16_vec, ttype, tensor_shape);
          fuzz_vec.push_back(*fuzz_tensval_ptr);
      }
      else if (type_str.compare("int8") == 0) {
          ttype = tensorflow::DataType::DT_INT8;
          arg_int8_vec = {};
          if (contents_str.compare("[]") != 0) {
            while (contents_sstream.good()) {
              std::getline(contents_sstream, arg_str, ',');
              dummy = std::stoi(arg_str);
              arg_int8_vec.push_back(tensorflow::int8(dummy));
            }
          }
          fuzz_tensval_ptr = get_tensor_with_shape_and_multiple_values(arg_int8_vec, ttype, tensor_shape);
          fuzz_vec.push_back(*fuzz_tensval_ptr);
      }
      else if (type_str.compare("int16") == 0) {
          ttype = tensorflow::DataType::DT_INT16;
          arg_int16_vec = {};
          if (contents_str.compare("[]") != 0) {
            while (contents_sstream.good()) {
              std::getline(contents_sstream, arg_str, ',');
              dummy = std::stoi(arg_str);
              arg_int16_vec.push_back(tensorflow::int16(dummy));
            }
          }
          fuzz_tensval_ptr = get_tensor_with_shape_and_multiple_values(arg_int16_vec, ttype, tensor_shape);
          fuzz_vec.push_back(*fuzz_tensval_ptr);
      }
      else if (type_str.compare("int32") == 0) {
          ttype = tensorflow::DataType::DT_INT32;
          arg_int32_vec = {};
          if (contents_str.compare("[]") != 0) {
            while (contents_sstream.good()) {
              std::getline(contents_sstream, arg_str, ',');
              dummy = std::stoi(arg_str);
              arg_int32_vec.push_back(tensorflow::int32(dummy));
            }
          }
          fuzz_tensval_ptr = get_tensor_with_shape_and_multiple_values(arg_int32_vec, ttype, tensor_shape);
          fuzz_vec.push_back(*fuzz_tensval_ptr);
      }
      else if (type_str.compare("int64") == 0) {
          ttype = tensorflow::DataType::DT_INT64;
          arg_int64_vec = {};
          if (contents_str.compare("[]") != 0) {
            while (contents_sstream.good()) {
              std::getline(contents_sstream, arg_str, ',');
              ldummy = std::stol(arg_str);
              arg_int64_vec.push_back(tensorflow::int64(ldummy));
            }
          }
          fuzz_tensval_ptr = get_tensor_with_shape_and_multiple_values(arg_int64_vec, ttype, tensor_shape);
          fuzz_vec.push_back(*fuzz_tensval_ptr);
      }
      else if (type_str.compare("uint8") == 0) {
          ttype = tensorflow::DataType::DT_UINT8;
          arg_uint8_vec = {};
          if (contents_str.compare("[]") != 0) {
            while (contents_sstream.good()) {
              std::getline(contents_sstream, arg_str, ',');
              dummy = std::stoi(arg_str);
              arg_uint8_vec.push_back(tensorflow::uint8(dummy));
            }
          }
          fuzz_tensval_ptr = get_tensor_with_shape_and_multiple_values(arg_uint8_vec, ttype, tensor_shape);
          fuzz_vec.push_back(*fuzz_tensval_ptr);
      }
      else if (type_str.compare("uint16") == 0) {
          ttype = tensorflow::DataType::DT_UINT16;
          arg_uint16_vec = {};
          if (contents_str.compare("[]") != 0) {
            while (contents_sstream.good()) {
              std::getline(contents_sstream, arg_str, ',');
              dummy = std::stoi(arg_str);
              arg_uint16_vec.push_back(tensorflow::uint16(dummy));
            }
          }
          fuzz_tensval_ptr = get_tensor_with_shape_and_multiple_values(arg_uint16_vec, ttype, tensor_shape);
          fuzz_vec.push_back(*fuzz_tensval_ptr);
      }
      else if (type_str.compare("uint32") == 0) {
          ttype = tensorflow::DataType::DT_UINT32;
          arg_uint32_vec = {};
          if (contents_str.compare("[]") != 0) {
            while (contents_sstream.good()) {
              std::getline(contents_sstream, arg_str, ',');
              dummy = std::stoi(arg_str);
              arg_uint32_vec.push_back(tensorflow::uint32(dummy));
            }
          }
          fuzz_tensval_ptr = get_tensor_with_shape_and_multiple_values(arg_uint32_vec, ttype, tensor_shape);
          fuzz_vec.push_back(*fuzz_tensval_ptr);
      }
      else if (type_str.compare("uint64") == 0) {
          ttype = tensorflow::DataType::DT_UINT64;
          arg_uint64_vec = {};
          if (contents_str.compare("[]") != 0) {
            while (contents_sstream.good()) {
              std::getline(contents_sstream, arg_str, ',');
              ldummy = std::stol(arg_str);
              arg_uint64_vec.push_back(tensorflow::uint64(ldummy));
            }
          }
          fuzz_tensval_ptr = get_tensor_with_shape_and_multiple_values(arg_uint64_vec, ttype, tensor_shape);
          fuzz_vec.push_back(*fuzz_tensval_ptr);
      }
      else if (type_str.compare("float32") == 0) {
          ttype = tensorflow::DataType::DT_FLOAT;
          arg_float_vec = {};
          if (contents_str.compare("[]") != 0) {
            while (contents_sstream.good()) {
              std::getline(contents_sstream, arg_str, ',');
              arg_float_vec.push_back(std::stof(arg_str));
            }
          }
          fuzz_tensval_ptr = get_tensor_with_shape_and_multiple_values(arg_float_vec, ttype, tensor_shape);
          fuzz_vec.push_back(*fuzz_tensval_ptr);
      }
      else if (type_str.compare("half") == 0) {
          ttype = tensorflow::DataType::DT_HALF;
          arg_half_vec = {};
          if (contents_str.compare("[]") != 0) {
            while (contents_sstream.good()) {
              std::getline(contents_sstream, arg_str, ',');
              arg_half_vec.push_back(Eigen::half(std::stof(arg_str)));
            }
          }
          fuzz_tensval_ptr = get_tensor_with_shape_and_multiple_values(arg_half_vec, ttype, tensor_shape);
          fuzz_vec.push_back(*fuzz_tensval_ptr);
      }
      else if (type_str.compare("float64") == 0) {
          ttype = tensorflow::DataType::DT_DOUBLE;
          arg_double_vec = {};
          if (contents_str.compare("[]") != 0) {
            while (contents_sstream.good()) {
              std::getline(contents_sstream, arg_str, ',');
              arg_double_vec.push_back(std::stod(arg_str));
            }
          }
          fuzz_tensval_ptr = get_tensor_with_shape_and_multiple_values(arg_double_vec, ttype, tensor_shape);
          fuzz_vec.push_back(*fuzz_tensval_ptr);
      }
      else if (type_str.compare("bool") == 0) {
          ttype = tensorflow::DataType::DT_BOOL;
          arg_bool_vec = {};
          if (contents_str.compare("[]") != 0) {
            while (contents_sstream.good()) {
              std::getline(contents_sstream, arg_str, ',');
              if (arg_str.compare("True") == 0) {
                arg_bool_vec.push_back(true);
              } else {
                arg_bool_vec.push_back(false);
              }
            }
          }
          fuzz_tensval_ptr = get_tensor_with_shape_and_multiple_values(arg_bool_vec, ttype, tensor_shape);
          fuzz_vec.push_back(*fuzz_tensval_ptr);
      }
      else if (type_str.compare("string") == 0) {
          ttype = tensorflow::DataType::DT_STRING;
          arg_string_vec = {};
          arg_string_vec.push_back(tensorflow::tstring(contents_str));
          fuzz_tensval_ptr = get_tensor_with_shape_and_multiple_values(arg_string_vec, ttype, tensor_shape);
          fuzz_vec.push_back(*fuzz_tensval_ptr);
      }
    }

    std::cout << "Created inputs: " << std::endl;
    for (auto tensor_val : fuzz_vec) {
      std::cout << tensor_val.tensor->DebugString() << std::endl;
    }

    tensorflow::gtl::InlinedVector<tensorflow::TensorValue, 4> *fuzz_inputs = new
      tensorflow::gtl::InlinedVector<tensorflow::TensorValue, 4>(fuzz_vec.begin(), fuzz_vec.end());
    /*
     * Make sure we match the number of inputs we read. If not, remove the check
     * filename such that false_positive() won't log this kernel. The kernel
     * will eventually get reached by the driver having the expected number of
     * inputs
     */
    if (original_ctx->num_inputs() != inputs_read) {
      std::string check_filename = std::string(results_dir) + "/" + cur_fname + ".check";
      std::remove(check_filename.c_str());
    }
    validate_ctx_params->inputs = fuzz_inputs;
    validate_ctx = new tensorflow::OpKernelContext(validate_ctx_params);
    std::cout << "Returning validate context with " << validate_ctx->num_inputs() << " inputs"<< std::endl;
    return validate_ctx;
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

#else
  #if defined(IVYSYN_GPU_ONLY)
  Fuzzer::Fuzzer(const std::string& fname, tensorflow::OpKernelContext* ctx, bool hasDevice, const char *device)
  #else
  Fuzzer::Fuzzer(const std::string& fname, tensorflow::OpKernelContext* ctx)
  #endif
  {
    /* std::cout << "In fuzzer for " << fname << std::endl; */
  #if defined(IVYSYN_GPU_ONLY)
    if (hasDevice) {
        if (std::string(device).compare("N5Eigen9GpuDeviceE") != 0) {
	    std::cout << "In fuzzer for " << fname << " but not GPU implementation, skipping" << std::endl;
	    total_mutations = -1;
	    main_pool_done = true;
	    return;
        }
    }
  #endif

    main_pool_done = false;

    std::string mutfile_pattern;
    std::string mutfile_prefix;
    std::string proc_filename;
    std::string mut_filename;
    std::string last_timestamp_filename;
    std::string time_filename;
    std::string except_filename;
    std::string total_filename;
    std::string start_filename;
    std::string nofuzz_filename;

    std::fstream total_file;
    std::ios_base::openmode fflags;

    bool restore = false, do_resume = false;
    long long last_mutation = -1, last_timestamp = -1;
    struct stat stat_buffer = {};
    struct timespec ts;

    glob_t glob_result = {0};
    int glob_ret = {};
    pid_t mypid = 0;
    char *existing_pid;
    bool got_last = false;
    int tries = 0;
    bool log_crash;

    tensorflow::Tensor tensor;
    tensorflow::TensorShape tensor_shape;

    cur_fname = fname;
    cur_fname_glob.assign(cur_fname);
    /* attrs = tensorflow::SummarizeAttrs(ctx->op_kernel().def()).c_str(); */

    num_args = ctx->num_inputs();

    original_ctx = new tensorflow::OpKernelContext(ctx->get_params());
    original_inputs = original_ctx->get_params()->inputs;

    /* printf("Original num args: %d\n", original_ctx->num_inputs()); */
    /* printf("Original arguments: \n"); */
    /* for (int i = 0; i < num_args; i++) { */
    /*   std::cout << "Argument " << i << ": " << original_ctx->input(i).DebugString() << "\n"; */
    /* } */

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

    fflags = std::ios::out | std::ios::in | std::ios::trunc;

    mutations_logger_filename = mut_filename;
    timestamp_logger_filename = last_timestamp_filename;

    std::mt19937_64 rng(std::random_device{}());
    std::uniform_int_distribution<std::mt19937_64::result_type> dist(0, 2000);
    int slp = dist(rng);
    std::this_thread::sleep_for(std::chrono::milliseconds(slp));

    glob_ret = glob(mutfile_pattern.c_str(), 0, NULL, &glob_result);
    if (glob_ret != GLOB_NOMATCH && !restore) {

      // A lot of empty mutation files, probably deadlock or bug, stop
      // fuzzing this kernel
      if (glob_result.gl_pathc > 5) {
        mark_fuzzing_done();
        std::cout << cur_fname << " has a lot of empty mutation files, skipping" << std::endl;
        return;
      }

      // A mutation file for the same function exists
      for(size_t i = 0; i < glob_result.gl_pathc && !restore; ++i) {

        existing_pid = glob_result.gl_pathv[i] + mutfile_prefix.length() + 1;
        proc_filename = "/proc/" + std::string(existing_pid);

        if (stat(proc_filename.c_str(), &stat_buffer) == 0){
          // The mutations file belongs to a running process, skip
          /* printf("%s belongs to a running process, skipping\n", glob_result.gl_pathv[i]); */
          total_mutations = 0;
          is_running = true;
          globfree(&glob_result);
          return;
        } else {
          // The mutations file doesn't belong to any running process, something crashed
          mutations_restore_filename = glob_result.gl_pathv[i];
          size_t mut_str_idx = mutations_restore_filename.find("_mutations.log");
          timestamp_restore_filename = mutations_restore_filename;
          timestamp_restore_filename.replace(mut_str_idx, std::string("_mutations.log").length(), ".last_timestamp");
          /* std::cout << cur_fname << "crashed, will restore from " << mutations_restore_filename << std::endl; */
          restore = true;

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
      std::cout << mypid << ": " << cur_fname << " crashed, will restore from " << mutations_restore_filename << std::endl;
    } else {
      create_file(time_filename, time_file, fflags);
      time_file.rdbuf()->pubsetbuf(nullptr, 0);
      create_file(except_filename, except_file, fflags);
      except_file.rdbuf()->pubsetbuf(nullptr, 0);
    }

    std::cout << mypid << ": Fuzzing function " << cur_fname << std::endl;

    /* Disable buffering else program might crash before writing to logger */
    create_file(mutations_logger_filename, mutations_file, fflags);
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

    for (int i = 0; i < num_args; i++) {
      if (!ctx->has_input(i) || ctx->input_is_ref(i)) {
        mark_fuzzing_done();
        create_file(nofuzz_filename, nofuzz_file, std::ios::out);
        return;
      }
      indices.push_back(0);
      tensor = ctx->input(i);
      tensor_shape = tensor.shape();
      tensor_shapes.push_back(tensor_shape);
      tensor_dims.push_back(tensor.dims());
      tensor_types.push_back(tensor.dtype());
      tensor_types_set.insert(tensor.dtype());
      /* std::cout << "Input " << i << " dtype " << tensor.dtype() << std::endl; */
      if (tensor.dtype() == tensorflow::DataType::DT_RESOURCE) {
        mark_fuzzing_done();
        create_file(nofuzz_filename, nofuzz_file, std::ios::out);
        return;
      }
    }

    auto shuf_rng = std::default_random_engine(RNG_SEED);
    initialize_tensor_pools();
    std::shuffle(std::begin(int8_mutations), std::end(int8_mutations), shuf_rng);
    std::shuffle(std::begin(uint8_mutations), std::end(uint8_mutations), shuf_rng);
    std::shuffle(std::begin(uint32_mutations), std::end(uint32_mutations), shuf_rng);
    std::shuffle(std::begin(uint64_mutations), std::end(uint64_mutations), shuf_rng);
    std::shuffle(std::begin(int32_mutations), std::end(int32_mutations), shuf_rng);
    std::shuffle(std::begin(int64_mutations), std::end(int64_mutations), shuf_rng);
    std::shuffle(std::begin(half_mutations), std::end(half_mutations), shuf_rng);
    std::shuffle(std::begin(float_mutations), std::end(float_mutations), shuf_rng);
    std::shuffle(std::begin(double_mutations), std::end(double_mutations), shuf_rng);
    std::shuffle(std::begin(string_mutations), std::end(string_mutations), shuf_rng);
    calculate_total_mutations();

    /* std::cout << "Calculated total mutations for " << fname << std::endl; */

    // Log total number of mutations for function
    if (!restore) {
      total_file.clear();
      total_file.open(total_filename, std::ios::app);
      if (total_file.fail()) {
        std::cout << "Failed to open " << total_filename << std::endl;
        std::cout << "Error: " << strerror(errno) << std::endl;
      }
      total_file << cur_fname << ":" << all_mutations << std::endl << std::flush;
      total_file.close();
    }

    /* std::cout << "Will restore for:" << fname << ":" << restore << std::endl; */

    if (restore) {

      mutations_restore.open(mutations_restore_filename, std::ios::out | std::ios::in);
      timestamp_restore.open(timestamp_restore_filename, std::ios::out | std::ios::in);
      std::string last_line;
      while (!got_last && tries < 10) {
        getline(mutations_restore, last_line);
        if (last_line.length() > 0) {
          last_mutation = std::stoll(last_line);
          got_last = true;
        } else {
          std::cout << "Error while reading " << mutations_restore_filename << " (got " << last_line << ") ..." << std::endl;
          tries++;
        }
      }
      getline(timestamp_restore, last_line);
      if (last_line.length() <= 0) {
        std::cout << "Error while reading last timestamp" << std::endl;
      } else {
        last_timestamp = std::stoll(last_line);
      }

      if (last_mutation >= 0) {
        /* std::cout << "Restoring from mutation " << last_mutation << std::endl; */
        restore_last_mutation(last_mutation, last_timestamp, do_resume);
        /* Delete the file since we already logged the crash */
        std::remove(mutations_restore_filename.c_str());
        std::remove(timestamp_restore_filename.c_str());
      }
    } else {
      if (num_args > 0) {
        indices[0] = -1;
      }
    }

    struct sigaction timeout_sigaction = {};
    timeout_sigaction.sa_handler = handle_timeout;
    sigaction(SIGALRM, &timeout_sigaction, NULL);
    alarm(TIMEOUT_SECS);

  }
#endif

  Fuzzer::~Fuzzer()
  {
    /* Cancel current alarm */
    alarm(0);
  }

  template <class T>
    tensorflow::TensorValue *Fuzzer::get_tensor_with_shape_and_multiple_values(std::vector<T> values, tensorflow::DataType ttype,
                                                          tensorflow::TensorShape shape)
    {

      tensorflow::Tensor *tensor;
      tensorflow::TensorValue *tensor_val;
      int idx = 0;

      tensor = new tensorflow::Tensor(ttype, shape);

      /* std::cout << "Creating tensor with multiple values: " << std::endl; */
      if (values.size() == 1) {
          tensor->flat<T>().setConstant(values.at(0));
      } else {
        for (auto val: values) {
          /* std::cout << val << std::endl; */
          tensor->flat<T>()(idx++) = val;
        }
      }

      tensor_val = new tensorflow::TensorValue(tensor);

      return tensor_val;

    }


#if !defined(IVYSYN_VALIDATE) && !defined(IVYSYN_COLLECT_TYPES)
  tensorflow::TensorValue Fuzzer::get_next_mut(tensorflow::DataType ttype, int idx) {

    tensorflow::Tensor *tensor;

    switch (ttype) {
      default:
        mark_unknown_type(ttype);
      case tensorflow::DataType::DT_QINT8:
        return qint8_tensor_mutation_pool.at(indices[cur_idx++]);
      case tensorflow::DataType::DT_QINT16:
        return qint16_tensor_mutation_pool.at(indices[cur_idx++]);
      case tensorflow::DataType::DT_QINT32:
        return qint32_tensor_mutation_pool.at(indices[cur_idx++]);
      case tensorflow::DataType::DT_QUINT8:
        return quint8_tensor_mutation_pool.at(indices[cur_idx++]);
      case tensorflow::DataType::DT_QUINT16:
        return quint16_tensor_mutation_pool.at(indices[cur_idx++]);
      case tensorflow::DataType::DT_INT8:
        return int8_tensor_mutation_pool.at(indices[cur_idx++]);
      case tensorflow::DataType::DT_INT16:
        return int16_tensor_mutation_pool.at(indices[cur_idx++]);
      case tensorflow::DataType::DT_INT32:
        return int32_tensor_mutation_pool.at(indices[cur_idx++]);
      case tensorflow::DataType::DT_INT64:
        return int64_tensor_mutation_pool.at(indices[cur_idx++]);
      case tensorflow::DataType::DT_UINT8:
        return uint8_tensor_mutation_pool.at(indices[cur_idx++]);
      case tensorflow::DataType::DT_UINT16:
        return uint16_tensor_mutation_pool.at(indices[cur_idx++]);
      case tensorflow::DataType::DT_UINT32:
        return uint32_tensor_mutation_pool.at(indices[cur_idx++]);
      case tensorflow::DataType::DT_UINT64:
        return uint64_tensor_mutation_pool.at(indices[cur_idx++]);
      case tensorflow::DataType::DT_FLOAT:
        return float_tensor_mutation_pool.at(indices[cur_idx++]);
      case tensorflow::DataType::DT_HALF:
        return half_tensor_mutation_pool.at(indices[cur_idx++]);
      case tensorflow::DataType::DT_DOUBLE:
        return double_tensor_mutation_pool.at(indices[cur_idx++]);
      case tensorflow::DataType::DT_BOOL:
        return bool_tensor_mutation_pool.at(indices[cur_idx++]);
      case tensorflow::DataType::DT_STRING:
        return string_tensor_mutation_pool.at(indices[cur_idx++]);

        //  No mutations for these so just return the original tensor
      case tensorflow::DataType::DT_VARIANT:
        {
          /* std::cout << "Creating DT_VARIANT tensor\n" << std::flush; */
          tensorflow::Tensor orig = original_ctx->input(idx);
          tensor = new tensorflow::Tensor(tensorflow::tensor::DeepCopy(orig));
          return tensorflow::TensorValue(tensor);
          /* return tensorflow::TensorValue(&orig); */
        }

      case tensorflow::DataType::DT_BFLOAT16:
      case tensorflow::DataType::DT_COMPLEX64:
      case tensorflow::DataType::DT_COMPLEX128:
      case tensorflow::DataType::DT_RESOURCE:
        {
          tensor = new tensorflow::Tensor(original_ctx->input(idx));
          return tensorflow::TensorValue(tensor);
        }
    }
  }

  void Fuzzer::log_current_mutation(std::fstream &file)
  {
    tensorflow::DataType ttype;
    tensorflow::TensorValue tensor_val;
    tensorflow::Tensor tensor;
    std::string out_str;
    tensorflow::OpKernelContext *ctx;

    out_str += tensorflow::SummarizeAttrs(original_ctx->op_kernel().def()) + "\n";
    if (!main_pool_done) {
      for (int idx = 0; idx < num_args; idx++) {
        ttype = tensor_types.at(idx);
        tensor_val = get_next_mut(ttype, idx);
        ttype = tensor_val.tensor->dtype();
        switch (ttype) {
          default:
#if defined(IVYSYN_GPU_ONLY)
            out_str += tensor_val.tensor->DebugString() + "\n";
#else
            out_str += tensor_val.tensor->DebugString() + "\n";
#endif
            break;
          case tensorflow::DataType::DT_RESOURCE:
            out_str += "Resource\n";
            break;
        }
      }
    } else {
      /* std::cout << "Crash for " << cur_fname << " was from extra pool" << std::endl << std::flush; */
      ctx = get_fuzzed_context();
      for (int idx = 0; idx < num_args; idx++) {
        tensor = ctx->input(idx);
        ttype = tensor.dtype();
        switch (ttype) {
          default:
            out_str += tensor.DebugString() + "\n";
            break;
          case tensorflow::DataType::DT_RESOURCE:
            out_str += "Resource\n";
            break;
        }
      }
    }
    out_str += "\n--------------------------------------\n";
    file << out_str << std::flush;
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
      create_file(crashes_num_filename, num_crashes_file, fflags);
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

  void Fuzzer::calculate_total_mutations()
  {

    long long nmut_fuzz;
    long long pool_size;
    int overflow = 0;

    for (auto &ttype : tensor_types) {
      switch (ttype) {
        default:
          mark_unknown_type(ttype);
        case tensorflow::DataType::DT_QINT8:
          pool_size = qint8_tensor_mutation_pool.size();
          /* std::cout << "qint8 pool size: " << pool_size << std::endl; */
          overflow |= __builtin_smulll_overflow(total_mutations, pool_size, &total_mutations);
          pool_sizes.push_back(pool_size);
          break;
        case tensorflow::DataType::DT_QINT16:
          pool_size = qint16_tensor_mutation_pool.size();
          /* std::cout << "qint16 size: " << pool_size << std::endl; */
          overflow |= __builtin_smulll_overflow(total_mutations, pool_size, &total_mutations);
          pool_sizes.push_back(pool_size);
          break;
        case tensorflow::DataType::DT_QINT32:
          pool_size = qint32_tensor_mutation_pool.size();
          /* std::cout << "qint32 pool size: " << pool_size << std::endl; */
          overflow |= __builtin_smulll_overflow(total_mutations, pool_size, &total_mutations);
          pool_sizes.push_back(pool_size);
          break;
        case tensorflow::DataType::DT_QUINT8:
          pool_size = quint8_tensor_mutation_pool.size();
          /* std::cout << "quint8 pool size: " << pool_size << std::endl; */
          overflow |= __builtin_smulll_overflow(total_mutations, pool_size, &total_mutations);
          pool_sizes.push_back(pool_size);
          break;
        case tensorflow::DataType::DT_QUINT16:
          pool_size = quint16_tensor_mutation_pool.size();
          /* std::cout << "quint16 pool size: " << pool_size << std::endl; */
          overflow |= __builtin_smulll_overflow(total_mutations, pool_size, &total_mutations);
          pool_sizes.push_back(pool_size);
          break;
        case tensorflow::DataType::DT_INT8:
          pool_size = int8_tensor_mutation_pool.size();
          /* std::cout << "int8 pool size: " << pool_size << std::endl; */
          overflow |= __builtin_smulll_overflow(total_mutations, pool_size, &total_mutations);
          pool_sizes.push_back(pool_size);
          break;
        case tensorflow::DataType::DT_INT16:
          pool_size = int16_tensor_mutation_pool.size();
          /* std::cout << "int16 pool size: " << pool_size << std::endl; */
          overflow |= __builtin_smulll_overflow(total_mutations, pool_size, &total_mutations);
          pool_sizes.push_back(pool_size);
          break;
        case tensorflow::DataType::DT_INT32:
          pool_size = int32_tensor_mutation_pool.size();
          /* std::cout << "int32 pool size: " << pool_size << std::endl; */
          overflow |= __builtin_smulll_overflow(total_mutations, pool_size, &total_mutations);
          pool_sizes.push_back(pool_size);
          break;
        case tensorflow::DataType::DT_INT64:
          pool_size = int64_tensor_mutation_pool.size();
          /* std::cout << "int64 pool size: " << pool_size << std::endl; */
          overflow |= __builtin_smulll_overflow(total_mutations, pool_size, &total_mutations);
          pool_sizes.push_back(pool_size);
          break;
        case tensorflow::DataType::DT_UINT8:
          pool_size = uint8_tensor_mutation_pool.size();
          /* std::cout << "uint8 pool size: " << pool_size << std::endl; */
          overflow |= __builtin_smulll_overflow(total_mutations, pool_size, &total_mutations);
          pool_sizes.push_back(pool_size);
          break;
        case tensorflow::DataType::DT_UINT16:
          pool_size = uint16_tensor_mutation_pool.size();
          /* std::cout << "uint16 pool size: " << pool_size << std::endl; */
          overflow |= __builtin_smulll_overflow(total_mutations, pool_size, &total_mutations);
          pool_sizes.push_back(pool_size);
          break;
        case tensorflow::DataType::DT_UINT32:
          pool_size = uint32_tensor_mutation_pool.size();
          /* std::cout << "uint32 pool size: " << pool_size << std::endl; */
          overflow |= __builtin_smulll_overflow(total_mutations, pool_size, &total_mutations);
          pool_sizes.push_back(pool_size);
          break;
        case tensorflow::DataType::DT_UINT64:
          pool_size = uint64_tensor_mutation_pool.size();
          /* std::cout << "uint64 pool size: " << pool_size << std::endl; */
          overflow |= __builtin_smulll_overflow(total_mutations, pool_size, &total_mutations);
          pool_sizes.push_back(pool_size);
          break;
        case tensorflow::DataType::DT_HALF:
          pool_size = half_tensor_mutation_pool.size();
          /* std::cout << "half pool size: " << pool_size << std::endl; */
          overflow |= __builtin_smulll_overflow(total_mutations, pool_size, &total_mutations);
          pool_sizes.push_back(pool_size);
          break;
        case tensorflow::DataType::DT_FLOAT:
          pool_size = float_tensor_mutation_pool.size();
          /* std::cout << "float pool size: " << pool_size << std::endl; */
          overflow |= __builtin_smulll_overflow(total_mutations, pool_size, &total_mutations);
          pool_sizes.push_back(pool_size);
          break;
        case tensorflow::DataType::DT_DOUBLE:
          pool_size = double_tensor_mutation_pool.size();
          /* std::cout << "double pool size: " << pool_size << std::endl; */
          overflow |= __builtin_smulll_overflow(total_mutations, pool_size, &total_mutations);
          pool_sizes.push_back(pool_size);
          break;
        case tensorflow::DataType::DT_BOOL:
          pool_size = bool_tensor_mutation_pool.size();
          /* std::cout << "bool pool size: " << pool_size << std::endl; */
          overflow |= __builtin_smulll_overflow(total_mutations, pool_size, &total_mutations);
          pool_sizes.push_back(pool_size);
          break;
        case tensorflow::DataType::DT_STRING:
          pool_size = string_tensor_mutation_pool.size();
          /* std::cout << "string pool size: " << pool_size << std::endl; */
          overflow |= __builtin_smulll_overflow(total_mutations, pool_size, &total_mutations);
          pool_sizes.push_back(pool_size);
          break;
          // Just the original
        case tensorflow::DataType::DT_VARIANT:
        case tensorflow::DataType::DT_BFLOAT16:
        case tensorflow::DataType::DT_COMPLEX64:
        case tensorflow::DataType::DT_COMPLEX128:
        case tensorflow::DataType::DT_RESOURCE:
          pool_sizes.push_back(1);
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

    if (nmut_fuzz > NMUT_UPPER_BOUND_MID) {
      nmut_fuzz = NMUT_UPPER_BOUND_MID;
      num_mut_skip = total_mutations / nmut_fuzz;
    }

    all_mutations = total_mutations;
    std::cout << "Total mutations: " << total_mutations << std::endl;
    /* std::cout << "Will run with (at least): " << total_mutations << " mutations"<< std::endl; */
    std::cout << "Step size: " << num_mut_skip << std::endl;

    /* To avoid off by one on first mutation */
    total_mutations += num_mut_skip;

    for (auto &ndims : tensor_dims) {
      zero_dim_mutations += ndims;
    }

  }

  template <class T>
    tensorflow::TensorValue *Fuzzer::get_constant_tensor(T value)
    {
      tensorflow::Tensor *tensor;
      tensorflow::TensorValue *tensor_val;

      tensor = new tensorflow::Tensor(value);
      tensor_val = new tensorflow::TensorValue(tensor);
      return tensor_val;

    }

  tensorflow::TensorValue *Fuzzer::get_empty_tensor_with_shape(tensorflow::DataType ttype,
                                                         tensorflow::TensorShape shape)
  {

    tensorflow::Tensor *tensor;
    tensorflow::TensorValue *tensor_val;

    tensor = new tensorflow::Tensor(ttype, shape);
    tensor_val = new tensorflow::TensorValue(tensor);

    return tensor_val;
  }

  template <class T>
    tensorflow::TensorValue *Fuzzer::get_tensor_with_value(T value, tensorflow::Tensor *tensor)
    {

      tensorflow::TensorValue *tensor_val;

      tensor->flat<T>().setConstant(value);
      tensor_val = new tensorflow::TensorValue(tensor);
      return tensor_val;

    }

  template <class T>
    tensorflow::TensorValue *Fuzzer::get_tensor_with_shape_and_value(T value, tensorflow::DataType ttype,
                                                          tensorflow::TensorShape shape)
    {

      tensorflow::Tensor *tensor;
      tensorflow::TensorValue *tensor_val;

      tensor = new tensorflow::Tensor(ttype, shape);
      tensor->flat<T>().setConstant(value);
      tensor_val = new tensorflow::TensorValue(tensor);

      return tensor_val;

    }

  void Fuzzer::initialize_tensor_pools()
  {

    tensorflow::Tensor *tensor;
    tensorflow::TensorValue *tensor_val;
    tensorflow::DataType tensor_type;
    tensorflow::TensorShape shape, shape2, zero_shape, empty_shape;
    std::vector<int64_t> fuzz_shape_vec = {};
    int64_t rand_dim, rand_size;
    int idx, ndims;
    tensorflow::int8 rand_int8;
    tensorflow::int32 rand_int32;
    tensorflow::int64 rand_int64;
    tensorflow::uint8 rand_uint8;
    tensorflow::uint32 rand_uint32;
    tensorflow::uint64 rand_uint64;
    float rand_float;
    Eigen::half rand_half;
    double rand_double;
    tensorflow::tstring rand_string;

    /* Random generators */
    std::mt19937 rngenerator(RNG_SEED);
    std::uniform_int_distribution<> shape_distr(0, MAX_DIM_SIZE);
    std::uniform_int_distribution<> uint8_distr(0, uint8_mutations.size() - 1);
    std::uniform_int_distribution<> int8_distr(0, int8_mutations.size() - 1);
    std::uniform_int_distribution<> int32_distr(0, int32_mutations.size() - 1);
    std::uniform_int_distribution<> int64_distr(0, int64_mutations.size() - 1);
    std::uniform_int_distribution<> float_distr(0, float_mutations.size() - 1);
    std::uniform_int_distribution<> half_distr(0, half_mutations.size() - 1);
    std::uniform_int_distribution<> double_distr(0, double_mutations.size() - 1);
    std::uniform_int_distribution<> string_distr(0, string_mutations.size() - 1);
    std::uniform_int_distribution<> flip(0, 1);

    /* std::cout << "Creating mutations same as input\n" << std::flush; */

    /* Same arguments as the original we got as input */
    for (int i = 0; i < num_args; i++) {
      tensor = new tensorflow::Tensor(original_ctx->input(i));
      tensor_val = new tensorflow::TensorValue(tensor);
      tensor_type = tensor_types.at(i);
      switch (tensor_type) {
        default:
          mark_unknown_type(tensor_type);
        case tensorflow::DataType::DT_QINT8:
          qint8_tensor_mutation_pool.push_back(*tensor_val);
          break;
        case tensorflow::DataType::DT_QINT16:
          qint16_tensor_mutation_pool.push_back(*tensor_val);
          break;
        case tensorflow::DataType::DT_QINT32:
          qint32_tensor_mutation_pool.push_back(*tensor_val);
          break;
        case tensorflow::DataType::DT_QUINT8:
          quint8_tensor_mutation_pool.push_back(*tensor_val);
          break;
        case tensorflow::DataType::DT_QUINT16:
          quint16_tensor_mutation_pool.push_back(*tensor_val);
          break;
        case tensorflow::DataType::DT_INT8:
          int8_tensor_mutation_pool.push_back(*tensor_val);
          break;
        case tensorflow::DataType::DT_INT16:
          int16_tensor_mutation_pool.push_back(*tensor_val);
          break;
        case tensorflow::DataType::DT_INT32:
          int32_tensor_mutation_pool.push_back(*tensor_val);
          break;
        case tensorflow::DataType::DT_INT64:
          int64_tensor_mutation_pool.push_back(*tensor_val);
          break;
        case tensorflow::DataType::DT_UINT8:
          uint8_tensor_mutation_pool.push_back(*tensor_val);
          break;
        case tensorflow::DataType::DT_UINT16:
          uint16_tensor_mutation_pool.push_back(*tensor_val);
          break;
        case tensorflow::DataType::DT_UINT32:
          uint32_tensor_mutation_pool.push_back(*tensor_val);
          break;
        case tensorflow::DataType::DT_UINT64:
          uint64_tensor_mutation_pool.push_back(*tensor_val);
          break;
        case tensorflow::DataType::DT_HALF:
          half_tensor_mutation_pool.push_back(*tensor_val);
          break;
        case tensorflow::DataType::DT_FLOAT:
          float_tensor_mutation_pool.push_back(*tensor_val);
          break;
        case tensorflow::DataType::DT_DOUBLE:
          double_tensor_mutation_pool.push_back(*tensor_val);
          break;
        case tensorflow::DataType::DT_STRING:
          string_tensor_mutation_pool.push_back(*tensor_val);
          break;
        /* Will just create both later */
        case tensorflow::DataType::DT_BOOL:
        /* We don't create mutations for these */
        case tensorflow::DataType::DT_VARIANT:
        case tensorflow::DataType::DT_BFLOAT16:
        case tensorflow::DataType::DT_COMPLEX64:
        case tensorflow::DataType::DT_COMPLEX128:
        case tensorflow::DataType::DT_RESOURCE:
          break;
      }
    }

  /*
   * Create tensors with the same shapes as the original
   * but change values by picking a random value from the
   * mutations of the corresponding data type,
   * plus a tensor with zeroes for the integral types,
   * plus an empty tensor of same number of dimensions but 0-sized dimensions
   */
    idx = 0;
    for (auto &tshape : tensor_shapes) {

      ndims = tensor_dims.at(idx);
      zero_shape = tensorflow::TensorShape();
      for (int d = 0; d < ndims; d++) {
        zero_shape.AddDim(0);
      }

      tensor_type = tensor_types.at(idx++);

      tensor = new tensorflow::Tensor(tensor_type, tshape);
      switch (tensor_type) {
        default:
          mark_unknown_type(tensor_type);
        case tensorflow::DataType::DT_INT8:
          /* std::cout << "Creating int8 tensors" << std::endl << std::flush; */
          rand_int8 = int8_mutations.at(int8_distr(rngenerator));
          tensor_val = get_tensor_with_value<tensorflow::int8>(rand_int8, tensor);
          int8_tensor_mutation_pool.push_back(*tensor_val);

          tensor = new tensorflow::Tensor(tensor_type, tshape);
          tensor_val = get_tensor_with_value<tensorflow::int8>(0, tensor);
          int8_tensor_mutation_pool.push_back(*tensor_val);

          tensor_val = get_empty_tensor_with_shape(tensorflow::DataType::DT_INT8, zero_shape);
          int8_tensor_mutation_pool.push_back(*tensor_val);
          break;
        case tensorflow::DataType::DT_INT32:
          /* std::cout << "Creating int32 tensors" << std::endl << std::flush; */
          rand_int32 = int32_mutations.at(int32_distr(rngenerator));
          tensor_val = get_tensor_with_value<tensorflow::int32>(rand_int32, tensor);
          int32_tensor_mutation_pool.push_back(*tensor_val);

          tensor = new tensorflow::Tensor(tensor_type, tshape);
          tensor_val = get_tensor_with_value<tensorflow::int32>(0, tensor);
          int32_tensor_mutation_pool.push_back(*tensor_val);

          tensor_val = get_empty_tensor_with_shape(tensorflow::DataType::DT_INT32, zero_shape);
          int32_tensor_mutation_pool.push_back(*tensor_val);
          break;
        case tensorflow::DataType::DT_INT64:
          /* std::cout << "Creating int64 tensors" << std::endl << std::flush; */
          rand_int64 = int64_mutations.at(int64_distr(rngenerator));
          tensor_val = get_tensor_with_value<tensorflow::int64>(rand_int64, tensor);
          int64_tensor_mutation_pool.push_back(*tensor_val);

          tensor = new tensorflow::Tensor(tensor_type, tshape);
          tensor_val = get_tensor_with_value<tensorflow::int64>(0, tensor);
          int64_tensor_mutation_pool.push_back(*tensor_val);

          tensor_val = get_empty_tensor_with_shape(tensorflow::DataType::DT_INT64, zero_shape);
          int64_tensor_mutation_pool.push_back(*tensor_val);
          break;
        case tensorflow::DataType::DT_UINT8:
          /* std::cout << "Creating uint8 tensors" << std::endl << std::flush; */
          rand_uint8 = uint8_mutations.at(uint8_distr(rngenerator));
          tensor_val = get_tensor_with_value<tensorflow::uint8>(rand_uint8, tensor);
          uint8_tensor_mutation_pool.push_back(*tensor_val);

          tensor = new tensorflow::Tensor(tensor_type, tshape);
          tensor_val = get_tensor_with_value<tensorflow::uint8>(0, tensor);
          uint8_tensor_mutation_pool.push_back(*tensor_val);

          tensor_val = get_empty_tensor_with_shape(tensorflow::DataType::DT_UINT8, zero_shape);
          uint8_tensor_mutation_pool.push_back(*tensor_val);
          break;
        case tensorflow::DataType::DT_UINT32:
          /* std::cout << "Creating uint32 tensors" << std::endl << std::flush; */
          rand_uint32 = (tensorflow::uint32) int32_mutations.at(int32_distr(rngenerator));
          tensor_val = get_tensor_with_value<tensorflow::uint32>(rand_uint32, tensor);
          uint32_tensor_mutation_pool.push_back(*tensor_val);

          tensor = new tensorflow::Tensor(tensor_type, tshape);
          tensor_val = get_tensor_with_value<tensorflow::uint32>(0, tensor);
          uint32_tensor_mutation_pool.push_back(*tensor_val);

          tensor_val = get_empty_tensor_with_shape(tensorflow::DataType::DT_UINT32, zero_shape);
          uint32_tensor_mutation_pool.push_back(*tensor_val);
          break;
        case tensorflow::DataType::DT_UINT64:
          /* std::cout << "Creating uint64 tensors" << std::endl << std::flush; */
          rand_uint64 = (tensorflow::uint64) int64_mutations.at(int64_distr(rngenerator));
          tensor_val = get_tensor_with_value<tensorflow::uint64>(rand_int64, tensor);
          uint64_tensor_mutation_pool.push_back(*tensor_val);

          tensor = new tensorflow::Tensor(tensor_type, tshape);
          tensor_val = get_tensor_with_value<tensorflow::uint64>(0, tensor);
          uint64_tensor_mutation_pool.push_back(*tensor_val);

          tensor_val = get_empty_tensor_with_shape(tensorflow::DataType::DT_UINT64, zero_shape);
          uint64_tensor_mutation_pool.push_back(*tensor_val);
          break;
        case tensorflow::DataType::DT_HALF:
          /* std::cout << "Creating half tensors" << std::endl << std::flush; */
          rand_half = Eigen::half(half_mutations.at(half_distr(rngenerator)));
          tensor_val = get_tensor_with_value<Eigen::half>(rand_half, tensor);
          half_tensor_mutation_pool.push_back(*tensor_val);

          tensor = new tensorflow::Tensor(tensor_type, tshape);
          tensor_val = get_tensor_with_value<Eigen::half>(Eigen::half(0.0), tensor);
          half_tensor_mutation_pool.push_back(*tensor_val);

          tensor_val = get_empty_tensor_with_shape(tensorflow::DataType::DT_HALF, zero_shape);
          half_tensor_mutation_pool.push_back(*tensor_val);
          break;
        case tensorflow::DataType::DT_FLOAT:
          /* std::cout << "Creating float tensors" << std::endl << std::flush; */
          rand_float = float_mutations.at(float_distr(rngenerator));
          tensor_val = get_tensor_with_value<float>(rand_float, tensor);
          float_tensor_mutation_pool.push_back(*tensor_val);

          tensor = new tensorflow::Tensor(tensor_type, tshape);
          tensor_val = get_tensor_with_value<float>(0, tensor);
          float_tensor_mutation_pool.push_back(*tensor_val);

          tensor_val = get_empty_tensor_with_shape(tensorflow::DataType::DT_FLOAT, zero_shape);
          float_tensor_mutation_pool.push_back(*tensor_val);
          break;
        case tensorflow::DataType::DT_DOUBLE:
          /* std::cout << "Creating double tensors" << std::endl << std::flush; */
          rand_double = double_mutations.at(double_distr(rngenerator));
          tensor_val = get_tensor_with_value<double>(rand_double, tensor);
          double_tensor_mutation_pool.push_back(*tensor_val);

          tensor = new tensorflow::Tensor(tensor_type, tshape);
          tensor_val = get_tensor_with_value<double>(0, tensor);
          double_tensor_mutation_pool.push_back(*tensor_val);

          tensor_val = get_empty_tensor_with_shape(tensorflow::DataType::DT_DOUBLE, zero_shape);
          double_tensor_mutation_pool.push_back(*tensor_val);
          break;
        /* Will just create both later */
        case tensorflow::DataType::DT_BOOL:
        /* Don't create other mutations for these */
        case tensorflow::DataType::DT_VARIANT:
        case tensorflow::DataType::DT_STRING:
        case tensorflow::DataType::DT_BFLOAT16:
        case tensorflow::DataType::DT_COMPLEX64:
        case tensorflow::DataType::DT_COMPLEX128:
        case tensorflow::DataType::DT_RESOURCE:
        case tensorflow::DataType::DT_QINT8:
        case tensorflow::DataType::DT_QINT16:
        case tensorflow::DataType::DT_QINT32:
        case tensorflow::DataType::DT_QUINT8:
        case tensorflow::DataType::DT_QUINT16:
        case tensorflow::DataType::DT_INT16:
        case tensorflow::DataType::DT_UINT16:
          break;
      }

      // std::cout  << tensor_val->tensor->DebugString() << std::endl;
    }

    /*
     * Create tensors with single value picked from the value
     * pool for the corresponding type plus tensors with no
     * value (single, 0-sized dimension) plus completely empty
     * tensor
     */

    empty_shape = tensorflow::TensorShape();
    shape = tensorflow::TensorShape();
    shape.AddDim(0);

    for (auto &ttype : tensor_types_set) {
      switch (ttype) {
        case tensorflow::DataType::DT_UINT8:
          for (auto &fuzzval : uint8_mutations) {
            tensor_val = get_constant_tensor(fuzzval);
            uint8_tensor_mutation_pool.push_back(*tensor_val);
          }
          tensor_val = get_empty_tensor_with_shape(tensorflow::DataType::DT_UINT8, shape);
          uint8_tensor_mutation_pool.push_back(*tensor_val);

          tensor_val = get_empty_tensor_with_shape(tensorflow::DataType::DT_UINT8, empty_shape);
          uint8_tensor_mutation_pool.push_back(*tensor_val);
          break;
        case tensorflow::DataType::DT_UINT32:
          for (auto &fuzzval : uint32_mutations) {
            tensor_val = get_constant_tensor(fuzzval);
            uint32_tensor_mutation_pool.push_back(*tensor_val);
          }

          tensor_val = get_empty_tensor_with_shape(tensorflow::DataType::DT_UINT32, shape);
          uint32_tensor_mutation_pool.push_back(*tensor_val);

          tensor_val = get_empty_tensor_with_shape(tensorflow::DataType::DT_UINT32, empty_shape);
          uint32_tensor_mutation_pool.push_back(*tensor_val);
          break;
        case tensorflow::DataType::DT_UINT64:
          for (auto &fuzzval : uint64_mutations) {
            tensor_val = get_constant_tensor(fuzzval);
            uint64_tensor_mutation_pool.push_back(*tensor_val);
          }

          tensor_val = get_empty_tensor_with_shape(tensorflow::DataType::DT_UINT64, shape);
          uint64_tensor_mutation_pool.push_back(*tensor_val);

          tensor_val = get_empty_tensor_with_shape(tensorflow::DataType::DT_UINT64, empty_shape);
          uint64_tensor_mutation_pool.push_back(*tensor_val);
          break;
        case tensorflow::DataType::DT_INT8:
          for (auto &fuzzval : int8_mutations) {
            tensor_val = get_constant_tensor(fuzzval);
            int8_tensor_mutation_pool.push_back(*tensor_val);
          }

          tensor_val = get_empty_tensor_with_shape(tensorflow::DataType::DT_INT8, shape);
          int8_tensor_mutation_pool.push_back(*tensor_val);

          tensor_val = get_empty_tensor_with_shape(tensorflow::DataType::DT_INT8, empty_shape);
          int8_tensor_mutation_pool.push_back(*tensor_val);
          break;
        case tensorflow::DataType::DT_INT32:
          for (auto &fuzzval : int32_mutations) {
            tensor_val = get_constant_tensor(fuzzval);
            int32_tensor_mutation_pool.push_back(*tensor_val);
          }

          tensor_val = get_empty_tensor_with_shape(tensorflow::DataType::DT_INT32, shape);
          int32_tensor_mutation_pool.push_back(*tensor_val);

          tensor_val = get_empty_tensor_with_shape(tensorflow::DataType::DT_INT32, empty_shape);
          int32_tensor_mutation_pool.push_back(*tensor_val);
          break;
        case tensorflow::DataType::DT_INT64:
          for (auto &fuzzval : int64_mutations) {
            tensor_val = get_constant_tensor(fuzzval);
            int64_tensor_mutation_pool.push_back(*tensor_val);
          }

          tensor_val = get_empty_tensor_with_shape(tensorflow::DataType::DT_INT64, shape);
          int64_tensor_mutation_pool.push_back(*tensor_val);

          tensor_val = get_empty_tensor_with_shape(tensorflow::DataType::DT_INT64, empty_shape);
          int64_tensor_mutation_pool.push_back(*tensor_val);
          break;
        case tensorflow::DataType::DT_FLOAT:
          for (auto &fuzzval : float_mutations) {
            tensor_val = get_constant_tensor(fuzzval);
            float_tensor_mutation_pool.push_back(*tensor_val);
          }

          tensor_val = get_empty_tensor_with_shape(tensorflow::DataType::DT_FLOAT, shape);
          float_tensor_mutation_pool.push_back(*tensor_val);

          tensor_val = get_empty_tensor_with_shape(tensorflow::DataType::DT_FLOAT, empty_shape);
          float_tensor_mutation_pool.push_back(*tensor_val);
          break;
        case tensorflow::DataType::DT_HALF:
          for (auto &fuzzval : half_mutations) {
            tensor_val = get_constant_tensor(fuzzval);
            half_tensor_mutation_pool.push_back(*tensor_val);
          }

          tensor_val = get_empty_tensor_with_shape(tensorflow::DataType::DT_HALF, shape);
          half_tensor_mutation_pool.push_back(*tensor_val);

          tensor_val = get_empty_tensor_with_shape(tensorflow::DataType::DT_HALF, empty_shape);
          half_tensor_mutation_pool.push_back(*tensor_val);
          break;
        case tensorflow::DataType::DT_DOUBLE:
          for (auto &fuzzval : double_mutations) {
            tensor_val = get_constant_tensor(fuzzval);
            double_tensor_mutation_pool.push_back(*tensor_val);
          }

          tensor_val = get_empty_tensor_with_shape(tensorflow::DataType::DT_DOUBLE, shape);
          double_tensor_mutation_pool.push_back(*tensor_val);

          tensor_val = get_empty_tensor_with_shape(tensorflow::DataType::DT_DOUBLE, empty_shape);
          double_tensor_mutation_pool.push_back(*tensor_val);
          break;
        case tensorflow::DataType::DT_BOOL:
          tensor_val = get_constant_tensor(true);
          bool_tensor_mutation_pool.push_back(*tensor_val);
          tensor_val = get_constant_tensor(false);
          bool_tensor_mutation_pool.push_back(*tensor_val);
          break;
        case tensorflow::DataType::DT_STRING:
          for (auto &fuzzval : string_mutations) {
            tensor_val = get_constant_tensor(fuzzval);
            string_tensor_mutation_pool.push_back(*tensor_val);
          }

          tensor_val = get_empty_tensor_with_shape(tensorflow::DataType::DT_STRING, shape);
          string_tensor_mutation_pool.push_back(*tensor_val);

          tensor_val = get_empty_tensor_with_shape(tensorflow::DataType::DT_STRING, empty_shape);
          string_tensor_mutation_pool.push_back(*tensor_val);
          break;
        case tensorflow::DataType::DT_QINT8:
        case tensorflow::DataType::DT_QINT16:
        case tensorflow::DataType::DT_QINT32:
        case tensorflow::DataType::DT_QUINT8:
        case tensorflow::DataType::DT_QUINT16:
        case tensorflow::DataType::DT_INT16:
        case tensorflow::DataType::DT_UINT16:
        case tensorflow::DataType::DT_VARIANT:
        case tensorflow::DataType::DT_BFLOAT16:
        case tensorflow::DataType::DT_COMPLEX64:
        case tensorflow::DataType::DT_COMPLEX128:
        case tensorflow::DataType::DT_RESOURCE:
        default:
          break;
      }
    }
    /*
     * Create tensors with increasing number of dimensions, having
     * random dimension sizes and containing values picked at
     * random from the corresponding value pool for that type. Sometimes
     * insert a 0-sized dim instead.
     */
    for (int cur_ndims = 1; cur_ndims <= TENSOR_MAX_NUM_DIMS_FUZZ; cur_ndims+=TENSOR_DIM_STEP_FUZZ) {
      shape = tensorflow::TensorShape();
      shape2 = tensorflow::TensorShape();
      rand_dim = shape_distr(rngenerator);

      /* Some 0, rest another (same) random size */
      for (int i = 0; i < cur_ndims; i++) {
        if (flip(rngenerator)) {
          shape.AddDim(0);
        } else {
          shape.AddDim(rand_dim);
        }
      }

      /* Some 0, rest other (different) random sizes */
      for (int i = 0; i < cur_ndims; i++) {
        if (flip(rngenerator)) {
          shape2.AddDim(0);
        } else {
          rand_dim = shape_distr(rngenerator);
          shape2.AddDim(rand_dim);
        }
      }

      rand_int8 = int8_mutations.at(int8_distr(rngenerator));
      tensor_val = get_tensor_with_shape_and_value(rand_int8, tensorflow::DataType::DT_INT8, shape);
      int8_tensor_mutation_pool.push_back(*tensor_val);

      rand_int8 = int8_mutations.at(int8_distr(rngenerator));
      tensor_val = get_tensor_with_shape_and_value(rand_int8, tensorflow::DataType::DT_INT8, shape2);
      int8_tensor_mutation_pool.push_back(*tensor_val);

      rand_int32 = int32_mutations.at(int32_distr(rngenerator));
      tensor_val = get_tensor_with_shape_and_value(rand_int32, tensorflow::DataType::DT_INT32, shape);
      int32_tensor_mutation_pool.push_back(*tensor_val);

      rand_int32 = int32_mutations.at(int32_distr(rngenerator));
      tensor_val = get_tensor_with_shape_and_value(rand_int32, tensorflow::DataType::DT_INT32, shape2);
      int32_tensor_mutation_pool.push_back(*tensor_val);

      rand_int64 = int64_mutations.at(int64_distr(rngenerator));
      tensor_val = get_tensor_with_shape_and_value(rand_int64, tensorflow::DataType::DT_INT64, shape);
      int64_tensor_mutation_pool.push_back(*tensor_val);

      rand_int64 = int64_mutations.at(int64_distr(rngenerator));
      tensor_val = get_tensor_with_shape_and_value(rand_int64, tensorflow::DataType::DT_INT64, shape2);
      int64_tensor_mutation_pool.push_back(*tensor_val);

      rand_uint8 = (tensorflow::uint8) int32_mutations.at(int32_distr(rngenerator));
      tensor_val = get_tensor_with_shape_and_value(rand_uint8, tensorflow::DataType::DT_UINT8, shape);
      uint8_tensor_mutation_pool.push_back(*tensor_val);

      rand_uint8 = (tensorflow::uint8) int32_mutations.at(int32_distr(rngenerator));
      tensor_val = get_tensor_with_shape_and_value(rand_uint8, tensorflow::DataType::DT_UINT8, shape2);
      uint8_tensor_mutation_pool.push_back(*tensor_val);

      rand_uint32 = (tensorflow::uint32) int32_mutations.at(int32_distr(rngenerator));
      tensor_val = get_tensor_with_shape_and_value(rand_uint32, tensorflow::DataType::DT_UINT32, shape);
      uint32_tensor_mutation_pool.push_back(*tensor_val);

      rand_uint32 = (tensorflow::uint32) int32_mutations.at(int32_distr(rngenerator));
      tensor_val = get_tensor_with_shape_and_value(rand_uint32, tensorflow::DataType::DT_UINT32, shape2);
      uint32_tensor_mutation_pool.push_back(*tensor_val);

      rand_uint64 = (tensorflow::uint64) int64_mutations.at(int64_distr(rngenerator));
      tensor_val = get_tensor_with_shape_and_value(rand_uint64, tensorflow::DataType::DT_UINT64, shape);
      uint64_tensor_mutation_pool.push_back(*tensor_val);

      rand_uint64 = (tensorflow::uint64) int64_mutations.at(int64_distr(rngenerator));
      tensor_val = get_tensor_with_shape_and_value(rand_uint64, tensorflow::DataType::DT_UINT64, shape2);
      uint64_tensor_mutation_pool.push_back(*tensor_val);

      rand_float = float_mutations.at(float_distr(rngenerator));
      tensor_val = get_tensor_with_shape_and_value(rand_float, tensorflow::DataType::DT_FLOAT, shape);
      float_tensor_mutation_pool.push_back(*tensor_val);

      rand_float = float_mutations.at(float_distr(rngenerator));
      tensor_val = get_tensor_with_shape_and_value(rand_float, tensorflow::DataType::DT_FLOAT, shape2);
      float_tensor_mutation_pool.push_back(*tensor_val);

      rand_half = Eigen::half(half_mutations.at(half_distr(rngenerator)));
      tensor_val = get_tensor_with_shape_and_value(rand_half, tensorflow::DataType::DT_HALF, shape);
      half_tensor_mutation_pool.push_back(*tensor_val);

      rand_half = Eigen::half(half_mutations.at(half_distr(rngenerator)));
      tensor_val = get_tensor_with_shape_and_value(rand_half, tensorflow::DataType::DT_HALF, shape2);
      half_tensor_mutation_pool.push_back(*tensor_val);

      rand_double = double_mutations.at(double_distr(rngenerator));
      tensor_val = get_tensor_with_shape_and_value(rand_double, tensorflow::DataType::DT_DOUBLE, shape);
      double_tensor_mutation_pool.push_back(*tensor_val);

      rand_double = double_mutations.at(double_distr(rngenerator));
      tensor_val = get_tensor_with_shape_and_value(rand_double, tensorflow::DataType::DT_DOUBLE, shape2);
      double_tensor_mutation_pool.push_back(*tensor_val);

    }

  }

  void Fuzzer::mark_unknown_type(tensorflow::DataType ttype)
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
        std::ios_base::openmode fflags = std::ios::out | std::ios::in | std::ios::trunc;
        std::string zero_muts_filename = std::string(results_dir) + "/" + cur_fname + ".zero_muts";
        create_file(zero_muts_filename, zero_muts_file, fflags);
        has_more = true;
        total_mutations = zero_dim_mutations;
        main_pool_done = true;
      } else {
        original_ctx->get_params()->inputs = original_inputs;
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
      for (int i = 0; i < num_args; i++) {
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

    /* std::cout << "next_mutations_indices end\n" << std::flush; */
  }


  tensorflow::OpKernelContext *Fuzzer::get_fuzzed_context() {

    /* std::cout << "get_fuzzed_context()\n" << std::flush; */

    tensorflow::DataType ttype;
    tensorflow::OpKernelContext::Params *fuzz_ctx_params = original_ctx->get_params();
    std::vector<tensorflow::TensorValue> fuzz_vec;
    tensorflow::TensorValue fuzz_tensval;
    tensorflow::TensorValue *fuzz_tensval_ptr;
    tensorflow::OpKernelContext *fuzz_ctx = nullptr;

    tensorflow::Tensor tensor;
    tensorflow::TensorShape shape, orig_shape;
    long set_zero, cur_dim;

    if (!main_pool_done) {
      for (int idx = 0; idx < num_args; idx++) {
        ttype = tensor_types.at(idx);
        fuzz_tensval = get_next_mut(ttype, idx);
        fuzz_vec.push_back(fuzz_tensval);
      }
    } else {
      cur_dim = 1;
      for (int idx = 0; idx < num_args; idx++) {
        if (tensor_types.at(idx) == tensorflow::DataType::DT_VARIANT) {
          tensorflow::Tensor orig = original_ctx->input(idx);
          tensorflow::Tensor *tensor = new tensorflow::Tensor(tensorflow::tensor::DeepCopy(orig));
          fuzz_tensval = tensorflow::TensorValue(tensor);
        } else {
          shape = tensorflow::TensorShape();
          orig_shape = tensor_shapes.at(idx);
          for (int d = 0; d < tensor_dims.at(idx); d++) {
            if (cur_dim++ == total_mutations) {
              shape.AddDim(0);
            } else {
              shape.AddDim(orig_shape.dim_size(d));
            }
          }
          fuzz_tensval_ptr = get_empty_tensor_with_shape(tensor_types.at(idx), shape);
          if (fuzz_tensval_ptr == nullptr) {
            fuzz_tensval_ptr = new tensorflow::TensorValue(new tensorflow::Tensor(original_ctx->input(idx)));
          }
          fuzz_tensval = *fuzz_tensval_ptr;
        }
        fuzz_vec.push_back(fuzz_tensval);
      }
    }

    tensorflow::gtl::InlinedVector<tensorflow::TensorValue, 4> *fuzz_inputs = new
      tensorflow::gtl::InlinedVector<tensorflow::TensorValue, 4>(fuzz_vec.begin(), fuzz_vec.end());
    fuzz_ctx_params->inputs = fuzz_inputs;
    fuzz_ctx = new tensorflow::OpKernelContext(fuzz_ctx_params);

    return fuzz_ctx;

  }

  void Fuzzer::mut_start_time()
  {

    if (main_pool_done) {
      return;
    }

    clock_gettime(CLOCK_MONOTONIC, &start_time);
  }

  void Fuzzer::mut_end_time(tensorflow::OpKernelContext *fuzz_ctx)
  {

    /* if (main_pool_done) { */
    /*   return; */
    /* } */

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

    /* sprintf(logbuf, "%llu:%lu", total_mutations, duration); */
    if (fuzz_ctx->status() == tensorflow::Status::OK()) {
      /* time_file.write(logbuf, LOGBUFSZ); */
      time_file << total_mutations << ":" << duration << std::endl << std::flush;
    } else {
      /* except_file.write(logbuf, LOGBUFSZ); */
      except_file << total_mutations << ":" << duration << std::endl << std::flush;
    }


    // Log mutations that took more than THRESH seconds to finish
    /* if (duration_secs > TIME_THRESH_SECS) { */
      /* We want to log the mutation that just run, so this will achieve it:
       * 1. Move to the mutation BEFORE the mutation that just run
       * 2. Call next_mutations_indices to set up the indices array correctly
       * for the mutation after that, which is the mutation that just run
       * 3. Call log_current_mutation to log it
       * 4. When the fuzzed function calls has_more_mutations,
       * next_mutations_indices will be called again and will move to the next
       * mutation
       */
      /* total_mutations += (num_mut_skip * 2); */
      /* next_mutations_indices(false); */

      /* duration_filename = std::string(results_dir) + "/" + cur_fname + ".duration"; */
      /* if (stat(duration_filename.c_str(), &stat_buffer) != 0) { */
      /*   fflags |= std::ios::trunc; */
      /*   create_file(duration_filename, duration_file, fflags); */
      /* } else { */
      /*   duration_file.open(duration_filename, std::ios::app); */
      /*   if (duration_file.fail()) { */
      /*     std::cout << "Failed to open " << duration_filename << std::endl; */
      /*     std::cout << "Error: " << strerror(errno) << std::endl; */
      /*   } */
      /* } */

      /* duration_file << "Duration (secs): " << duration_secs << std::endl; */
      /* log_current_mutation(duration_file); */
      /* duration_file << std::flush; */
      /* duration_file.close(); */
    /* } */
  }

#endif

}
