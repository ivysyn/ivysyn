
#pragma once

#define IVYSYN_VALIDATE

#include <algorithm>
#include <array>
#include <chrono>         // std::chrono::seconds
#include <cstdarg>
#include <cstdio>
#include <cstdio>
#include <execinfo.h>
#include <mutex>
#include <fstream>
#include <glob.h>
#include <initializer_list>
#include <iostream>
#include <mutex>
#include <random>
#include <set>
#include <signal.h>
#include <string>
#include <sys/stat.h>
#include <sys/time.h>
#include <thread>         // std::this_thread::sleep_for
#include <unistd.h>
#include <unordered_map>
#include <vector>
#include <sys/stat.h>
#include <fcntl.h>

#include "c10/util/ArrayRef.h"
#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/TensorFactories.h>
#include <c10/core/TensorOptions.h>

#define NS_PER_SEC (1000 * 1000 * 1000)

#define MEDIUM_INT_FUZZ 0x20000000
#define MEDIUM_INT_NEG_FUZZ -0x20000000
#define LARGE_INT_FUZZ 0x70000000
#define LARGE_INT_NEG_FUZZ -0x70000000
#define LARGE_LONG_FUZZ 0x12345678abc
#define LARGE_LONG_NEG_FUZZ -0x12345678abc
#define HUGE_LONG_FUZZ 0x7777777777777777
#define HUGE_LONG_NEG_FUZZ -0x7777777777777777
#define LARGE_FLOAT_FUZZ 3.5e+035
#define LARGE_FLOAT_NEG_FUZZ -3.5e+035
#define LARGE_DOUBLE_FUZZ 1.5e+300
#define LARGE_DOUBLE_NEG_FUZZ -1.5e+300
#define LARGE_STRING "aaaabaaacaaadaaaeaaafaaagaaahaaaiaaajaaakaaalaaamaaanaaaoaaapaaaqaaaraaasaaataaauaaavaaawaaaxaaayaaazaabbaabcaabdaabeaabfaabgaabhaabiaabjaabkaablaabmaabnaaboaabpaabqaabraabsaabtaabuaabvaabwaabxaabyaabzaacbaaccaacdaaceaacfaacgaachaaciaacjaackaaclaacmaacnaacoaacpaacqaacraacsaactaacuaacvaacwaacxaacyaac"
#define ZERO_FUZZ 0

#define ARRAYREF_LEN 10
#define TENSOR_NUM_DIMS_FUZZ 5
#define TENSOR_DIM_SIZE_FUZZ 5
#define MAX_TENSOR_DIMS_FUZZ 15
#define MEDIUM_TENSOR_DIMS_FUZZ 10
#define SMALL_INT_FUZZ 0xfffe
#define SMALL_INT_NEG_FUZZ -0xfffe
#define LOGBUFSZ 0x20
#define BUFSZ 0x100
#define FILENAME_SZ 100

namespace fuzzing {

    extern bool already_fuzzing;
    extern bool do_quantize;
    extern const char* results_dir;

    enum TorchType {
        FUZZ_INT = 0,
        FUZZ_LONG,
        FUZZ_FLOAT,
        FUZZ_DOUBLE,
        FUZZ_BOOLEAN,
        FUZZ_SCALAR,
        FUZZ_TENSOR,
        FUZZ_SPARSE_TENSOR,
        FUZZ_TENSOR_LIST,
        FUZZ_INTARRAY_REF,
        FUZZ_DOUBLEARRAYREF,
        FUZZ_DIMNAME,
        FUZZ_DIMNAME_LIST,
        FUZZ_SCALARTYPE,
        FUZZ_BOOLARRAY,
        FUZZ_TENSOR_OPTIONS,
        FUZZ_LAYOUT,
        FUZZ_DEVICE,
        FUZZ_C10OPTIONAL_TENSOR,
        FUZZ_C10OPTIONAL_INTARRAYREF,
        FUZZ_C10OPTIONAL_DOUBLEARRAYREF,
        FUZZ_C10OPTIONAL_LONG,
        FUZZ_C10OPTIONAL_INT,
        FUZZ_C10OPTIONAL_DOUBLE,
        FUZZ_C10OPTIONAL_BOOL,
        FUZZ_C10OPTIONAL_SCALAR,
        FUZZ_C10OPTIONAL_SCALARTYPE,
        FUZZ_C10OPTIONAL_STRING,
        FUZZ_STRING,

        FUZZ_NUM_TYPES,
    };

    std::unordered_map<std::string, fuzzing::TorchType> const map_str_enum = {
        {"int", fuzzing::FUZZ_INT},
        {"int64_t", fuzzing::FUZZ_LONG},
        {"double", fuzzing::FUZZ_DOUBLE},
        {"bool", fuzzing::FUZZ_BOOLEAN},
        {"float", fuzzing::FUZZ_FLOAT},
        {"Layout", fuzzing::FUZZ_LAYOUT},
        {"Device", fuzzing::FUZZ_DEVICE},
        {"std::string", fuzzing::FUZZ_STRING},
        {"TensorList", fuzzing::FUZZ_TENSOR_LIST},
        {"Dimname", fuzzing::FUZZ_DIMNAME}, // For named tensors
        {"DimnameList", fuzzing::FUZZ_DIMNAME_LIST},

        {"Tensor", fuzzing::FUZZ_TENSOR},
        {"at::Tensor", fuzzing::FUZZ_TENSOR},

        {"SparseTensor", fuzzing::FUZZ_SPARSE_TENSOR},
        {"at::sparse::SparseTensor", fuzzing::FUZZ_SPARSE_TENSOR},

        {"Scalar", fuzzing::FUZZ_SCALAR},
        {"c10::Scalar", fuzzing::FUZZ_SCALAR},
        {"at::Scalar", fuzzing::FUZZ_SCALAR},

        {"IntArrayRef", fuzzing::FUZZ_INTARRAY_REF},
        {"c10::IntArrayRef", fuzzing::FUZZ_INTARRAY_REF},
        {"at::IntArrayRef", fuzzing::FUZZ_INTARRAY_REF},

        {"ArrayRef<double>", fuzzing::FUZZ_DOUBLEARRAYREF},
        {"c10::ArrayRef<double>", fuzzing::FUZZ_DOUBLEARRAYREF},
        {"at::ArrayRef<double>", fuzzing::FUZZ_DOUBLEARRAYREF},

        {"std::array<bool,3>", fuzzing::FUZZ_BOOLARRAY},
        {"std::array<bool,3ul>", fuzzing::FUZZ_BOOLARRAY},

        {"ScalarType", fuzzing::FUZZ_SCALARTYPE},
        {"at::ScalarType", fuzzing::FUZZ_SCALARTYPE},
        {"c10::ScalarType", fuzzing::FUZZ_SCALARTYPE},

        {"TensorOptions", fuzzing::FUZZ_TENSOR_OPTIONS},
        {"c10::TensorOptions", fuzzing::FUZZ_TENSOR_OPTIONS},
        {"at::TensorOptions", fuzzing::FUZZ_TENSOR_OPTIONS},

        {"optional<Tensor>", fuzzing::FUZZ_C10OPTIONAL_TENSOR},
        {"optional<at::Tensor>", fuzzing::FUZZ_C10OPTIONAL_TENSOR},
        {"c10::optional<Tensor>", fuzzing::FUZZ_C10OPTIONAL_TENSOR},
        {"c10::optional<at::Tensor>", fuzzing::FUZZ_C10OPTIONAL_TENSOR},

        {"optional<IntArrayRef>", fuzzing::FUZZ_C10OPTIONAL_INTARRAYREF},
        {"optional<c10::IntArrayRef>", fuzzing::FUZZ_C10OPTIONAL_INTARRAYREF},
        {"optional<at::IntArrayRef>", fuzzing::FUZZ_C10OPTIONAL_INTARRAYREF},
        {"c10::optional<IntArrayRef>", fuzzing::FUZZ_C10OPTIONAL_INTARRAYREF},
        {"c10::optional<c10::IntArrayRef>", fuzzing::FUZZ_C10OPTIONAL_INTARRAYREF},
        {"c10::optional<at::IntArrayRef>", fuzzing::FUZZ_C10OPTIONAL_INTARRAYREF},

        {"optional<Scalar>", fuzzing::FUZZ_C10OPTIONAL_SCALAR},
        {"optional<at::Scalar>", fuzzing::FUZZ_C10OPTIONAL_SCALAR},
        {"optional<c10::Scalar>", fuzzing::FUZZ_C10OPTIONAL_SCALAR},
        {"c10::optional<Scalar>", fuzzing::FUZZ_C10OPTIONAL_SCALAR},
        {"c10::optional<c10::Scalar>", fuzzing::FUZZ_C10OPTIONAL_SCALAR},
        {"c10::optional<at::Scalar>", fuzzing::FUZZ_C10OPTIONAL_SCALAR},

        {"optional<ScalarType>", fuzzing::FUZZ_C10OPTIONAL_SCALARTYPE},
        {"c10::optional<ScalarType>", fuzzing::FUZZ_C10OPTIONAL_SCALARTYPE},
        {"optional<c10::ScalarType>", fuzzing::FUZZ_C10OPTIONAL_SCALARTYPE},

        {"optional<ArrayRef<double>>", fuzzing::FUZZ_C10OPTIONAL_DOUBLEARRAYREF},
        {"optional<c10::ArrayRef<double>>", fuzzing::FUZZ_C10OPTIONAL_DOUBLEARRAYREF},
        {"c10::optional<ArrayRef<double>>", fuzzing::FUZZ_C10OPTIONAL_DOUBLEARRAYREF},
        {"c10::optional<c10::ArrayRef<double>>", fuzzing::FUZZ_C10OPTIONAL_DOUBLEARRAYREF},

        {"c10::optional<int>", fuzzing::FUZZ_C10OPTIONAL_INT},
        {"optional<int>", fuzzing::FUZZ_C10OPTIONAL_INT},

        {"c10::optional<int64_t>", fuzzing::FUZZ_C10OPTIONAL_LONG},
        {"optional<int64_t>", fuzzing::FUZZ_C10OPTIONAL_LONG},

        {"c10::optional<double>", fuzzing::FUZZ_C10OPTIONAL_DOUBLE},
        {"optional<double>", fuzzing::FUZZ_C10OPTIONAL_DOUBLE},

        {"c10::optional<bool>", fuzzing::FUZZ_C10OPTIONAL_BOOL},
        {"optional<bool>", fuzzing::FUZZ_C10OPTIONAL_BOOL},

        {"c10::optional<std::string>", fuzzing::FUZZ_C10OPTIONAL_STRING},
        {"optional<std::string>", fuzzing::FUZZ_C10OPTIONAL_STRING},
    };

    bool was_fuzzed(const std::string& fname);
    bool was_killed(const std::string& fname);
    void create_file(const std::string& filename, std::fstream &file, std::ios_base::openmode fflags);

    class Fuzzer {
    private:

#if defined(IVYSYN_VALIDATE)
        std::vector<int> int_mutations = {};
        std::vector<int64_t> long_mutations = {};
        std::vector<float> float_mutations = {};
        std::vector<double> double_mutations = {};
        std::vector<std::string> string_mutations = {};
        std::vector<bool> bool_mutations = {};
        bool _should_validate = false;
#endif

        void mark_unknown_type(std::string &ttype);
        std::string cur_fname;
        bool have_mkldnn_tensors = false;

#if !defined(IVYSYN_VALIDATE) && !defined(IVYSYN_COLLECT_TYPES)
        bool main_pool_done = false;
        long long num_mut_skip;
        bool is_running = false;
        long long total_mutations = 1;
        long zero_dim_mutations = 0;
        long zero_dim_intarray_mutations = 0;
        /* int extra_intarray_idx = 0, extra_tensor_idx = 0; */
        long long all_mutations;
        int rnd_idx = 0;
        std::string mutations_logger_filename;
        std::string timestamp_logger_filename;
        std::string mutations_restore_filename;
        std::string timestamp_restore_filename;
        std::string crashes_logger_filename;
        std::vector<int> pool_sizes;
        std::vector<at::IntArrayRef> tensor_dims;
        std::vector<at::IntArrayRef> sparse_tensor_dims;
        std::vector<int> intarrayref_sizes_vec;
        std::set<int> intarrayref_sizes;
        std::set<int> doublearrayref_sizes;
        std::vector<TorchType> func_types;

        std::vector<int> int_mutations = {ZERO_FUZZ, LARGE_INT_FUZZ, LARGE_INT_NEG_FUZZ,
            MEDIUM_INT_FUZZ, MEDIUM_INT_NEG_FUZZ, SMALL_INT_FUZZ, SMALL_INT_NEG_FUZZ};
        std::vector<int64_t> long_mutations = {ZERO_FUZZ, LARGE_LONG_FUZZ, LARGE_LONG_NEG_FUZZ,
            LARGE_INT_FUZZ, LARGE_INT_NEG_FUZZ, MEDIUM_INT_FUZZ, MEDIUM_INT_NEG_FUZZ,
            HUGE_LONG_FUZZ, HUGE_LONG_NEG_FUZZ, SMALL_INT_FUZZ, SMALL_INT_NEG_FUZZ, 1, 2, -1};
        std::vector<float> float_mutations = {ZERO_FUZZ, LARGE_FLOAT_FUZZ, LARGE_FLOAT_NEG_FUZZ};
        std::vector<double> double_mutations = {ZERO_FUZZ, LARGE_DOUBLE_FUZZ, LARGE_DOUBLE_NEG_FUZZ};
        std::vector<std::string> string_mutations = {std::string(""), std::string(LARGE_STRING)};
        std::vector<bool> bool_mutations = {true, false};

        std::vector<at::ScalarType> original_tensor_types;
        std::vector<void *> original_args;

        void initialize_intarrayref_pool();
        void initialize_doublearrayref_pool();
        void initialize_tensor_pool();
        void initialize_sparse_tensor_pool();
        void initialize_tensor_options_pool();
        void initialize_scalar_pool();
        void initialize_boolarrays();
        void calculate_total_mutations();
        void next_mutations_indices(bool log);
        inline void inc_mutations_indices(bool log);
        void restore_last_mutation(long long last_mutation, long long last_timestamp, bool resume);
        void log_current_mutation(std::fstream &file);
        void increase_num_crashes();
        void mark_fuzzing_done();

#endif

    public:

        Fuzzer(char *fname, std::vector<std::string> types_vec, std::vector<void *> args);
        ~Fuzzer();

#if defined(IVYSYN_VALIDATE)
        bool should_validate();
        void false_positive();
#endif

#if !defined(IVYSYN_COLLECT_TYPES)

        std::vector<at::ScalarType> scalar_types = {at::ScalarType::ComplexDouble, at::ScalarType::Double,
        at::ScalarType::Long, at::ScalarType::Bool};

        int cur_idx = 0;
        std::vector<int> indices;
        std::vector<int> nullopt_indices = {};

        std::vector<at::IntArrayRef> intarrayref_mutations;
        std::vector<at::ArrayRef<double>> doublearrayref_mutations;
        std::vector<at::Tensor> tensor_mutations;
        std::vector<double> tensor_contents;
        std::vector<at::Tensor> sparse_tensor_mutations;
        std::vector<at::TensorOptions> tensor_options_mutations;
        std::vector<at::Scalar> scalar_mutations;
        std::vector<std::array<bool,3>> bool_arrays;

        std::vector<at::Tensor> extra_tensor_mutations;
        std::vector<at::IntArrayRef> extra_intarrayref_mutations;
        std::vector<std::string> extra_func_names = {
          "_grid_sampler_2d_cpu_fallback",
          "thnn_conv2d",
          "_saturate_weight_to_fp16",
          "mkldnn_adaptive_avg_pool2d",
          "ormqr",
          "mkldnn_linear",
          "_pad_packed_sequence",
        };

        int get_next_mut_int();
        int64_t get_next_mut_long();
        bool get_next_mut_bool();
        double get_next_mut_double();
        std::string get_next_mut_string();
        at::Scalar get_next_mut_scalar();
        at::ScalarType get_next_mut_scalartype();
        at::IntArrayRef get_next_mut_intarrayref();
        at::ArrayRef<double> get_next_mut_doublearrayref();
        at::Tensor get_next_mut_tensor();
        at::Tensor get_next_mut_sparse_tensor();
        at::TensorOptions get_next_mut_tensor_options();
        std::array<bool,3> get_next_mut_boolarray();
        c10::optional<at::Tensor> get_next_mut_c10opt_tensor();
        c10::optional<at::Scalar> get_next_mut_c10opt_scalar();
        c10::optional<at::ScalarType> get_next_mut_c10opt_scalartype();
        c10::optional<at::IntArrayRef> get_next_mut_c10opt_intarrayref();
        c10::optional<at::ArrayRef<double>> get_next_mut_c10opt_doublearrayref();
        c10::optional<int> get_next_mut_c10opt_int();
        c10::optional<int64_t> get_next_mut_c10opt_long();
        c10::optional<double> get_next_mut_c10opt_double();
        c10::optional<bool> get_next_mut_c10opt_bool();
        c10::optional<std::string> get_next_mut_c10opt_string();

#endif

#if !defined(IVYSYN_VALIDATE) && !defined(IVYSYN_COLLECT_TYPES)
        bool has_more_mutations(bool reset);
        double get_tensor_contents();
        void mut_start_time();
        void mut_end_time(bool failed);
#endif

    };

}
