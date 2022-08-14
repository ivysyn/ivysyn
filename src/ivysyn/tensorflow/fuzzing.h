
/* #ifndef TENSORFLOW_CORE_FUZZING_H_ */
/* #define TENSORFLOW_CORE_FUZZING_H_ */

#pragma once

//#define IVYSYN_VALIDATE

#include <array>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <execinfo.h>
#include <filesystem>
#include <fstream>
#include <glob.h>
#include <initializer_list>
#include <iostream>
#include <random>
#include <set>
#include <signal.h>
#include <stdio.h>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"

#define NS_PER_SEC (1000 * 1000 * 1000)

#define MEDIUM_INT_FUZZ 0x20000000
#define MEDIUM_INT_NEG_FUZZ -0x20000000
#define LARGE_INT_FUZZ 0x70000000
#define LARGE_INT_NEG_FUZZ -0x70000000
#define LARGE_LONG_FUZZ 0x12345678abc
#define LARGE_LONG_NEG_FUZZ -0x12345678abc
#define LARGE_FLOAT_FUZZ 3.5e+035
#define LARGE_FLOAT_NEG_FUZZ -3.5e+035
#define LARGE_DOUBLE_FUZZ 1.5e+300
#define LARGE_DOUBLE_NEG_FUZZ -1.5e+300
#define LARGE_HALF_FUZZ 65000
#define LARGE_HALF_NEG_FUZZ -65000
#define LARGE_STRING "aaaabaaacaaadaaaeaaafaaagaaahaaaiaaajaaakaaalaaamaaanaaaoaaapaaaqaaaraaasaaataaauaaavaaawaaaxaaayaaazaabbaabcaabdaabeaabfaabgaabhaabiaabjaabkaablaabmaabnaaboaabpaabqaabraabsaabtaabuaabvaabwaabxaabyaabzaacbaaccaacdaaceaacfaacgaachaaciaacjaackaaclaacmaacnaacoaacpaacqaacraacsaactaacuaacvaacwaacxaacyaac"
#define ZERO_FUZZ 0
#define SMALL_INT_FUZZ 0xfffe
#define SMALL_INT_NEG_FUZZ -0xfffe

#define TENSOR_MAX_NUM_DIMS_FUZZ 10
#define TENSOR_DIM_STEP_FUZZ 1
#define MAX_DIM_SIZE 10

#define FILENAME_SZ 0x100
#define LOGBUFSZ 0x20
#define BUFSZ 0x100


namespace tffuzzing {

    extern bool already_fuzzing;
    extern const char *results_dir;

    bool was_fuzzed(const std::string& fname);
    bool was_killed(const std::string& fname);
    void create_file(const std::string& filename, std::fstream &file, std::ios_base::openmode fflags);
    struct timespec time_diff(struct timespec start, struct timespec end);
    void handle_timeout(int);

    class Fuzzer {
    private:

#if defined(IVYSYN_VALIDATE)
        bool _should_validate = false;
#endif
        tensorflow::OpKernelContext *original_ctx;
        const tensorflow::gtl::InlinedVector<tensorflow::TensorValue, 4>* original_inputs;
        std::string cur_fname;
        int num_args;
        template <class T> tensorflow::TensorValue *get_tensor_with_shape_and_multiple_values(std::vector<T> values, tensorflow::DataType ttype, tensorflow::TensorShape shape);

#if !defined(IVYSYN_VALIDATE) && !defined(IVYSYN_COLLECT_TYPES)
        bool main_pool_done = false;
        unsigned long num_mut_skip;
        bool is_running = false;
        long long total_mutations = 1;
        long zero_dim_mutations = 0;
        long long all_mutations;
        int cur_idx = 0;
        int cur_idx_zero_dims = 0;
        std::string mutations_logger_filename;
        std::string timestamp_logger_filename;
        std::string mutations_restore_filename;
        std::string timestamp_restore_filename;
        std::string crashes_logger_filename;
        std::vector<int> indices;
        std::vector<tensorflow::TensorShape> tensor_shapes;
        std::vector<int> tensor_dims;
        std::vector<tensorflow::DataType> tensor_types;
        std::set<tensorflow::DataType> tensor_types_set;

        std::vector<tensorflow::int8> int8_mutations{-127, ZERO_FUZZ, 127};
        std::vector<tensorflow::uint8> uint8_mutations{ZERO_FUZZ, 255};
        std::vector<tensorflow::uint32> uint32_mutations{ZERO_FUZZ, LARGE_INT_FUZZ, MEDIUM_INT_FUZZ, SMALL_INT_FUZZ};
        std::vector<tensorflow::uint64> uint64_mutations{ZERO_FUZZ, LARGE_LONG_FUZZ, LARGE_INT_FUZZ};
        std::vector<tensorflow::int32> int32_mutations{ZERO_FUZZ, LARGE_INT_FUZZ, LARGE_INT_NEG_FUZZ,
            MEDIUM_INT_FUZZ, MEDIUM_INT_NEG_FUZZ, SMALL_INT_FUZZ, SMALL_INT_NEG_FUZZ};
        std::vector<tensorflow::int64> int64_mutations{ZERO_FUZZ, LARGE_LONG_FUZZ, LARGE_LONG_NEG_FUZZ,
            LARGE_INT_FUZZ, LARGE_INT_NEG_FUZZ,
        };
        // Float here, converted to Eigen::half when used
        std::vector<float> half_mutations{ZERO_FUZZ, LARGE_HALF_FUZZ, LARGE_HALF_NEG_FUZZ};
        std::vector<float> float_mutations{ZERO_FUZZ, LARGE_FLOAT_FUZZ, LARGE_FLOAT_NEG_FUZZ};
        std::vector<double> double_mutations{ZERO_FUZZ, LARGE_DOUBLE_FUZZ, LARGE_DOUBLE_NEG_FUZZ};
        std::vector<tensorflow::tstring> string_mutations{tensorflow::tstring(""), tensorflow::tstring(LARGE_STRING)};

        std::vector<tensorflow::TensorValue> qint8_tensor_mutation_pool = {};
        std::vector<tensorflow::TensorValue> qint16_tensor_mutation_pool = {};
        std::vector<tensorflow::TensorValue> qint32_tensor_mutation_pool = {};
        std::vector<tensorflow::TensorValue> quint8_tensor_mutation_pool = {};
        std::vector<tensorflow::TensorValue> quint16_tensor_mutation_pool = {};
        std::vector<tensorflow::TensorValue> int8_tensor_mutation_pool = {};
        std::vector<tensorflow::TensorValue> uint8_tensor_mutation_pool = {};
        std::vector<tensorflow::TensorValue> int16_tensor_mutation_pool = {};
        std::vector<tensorflow::TensorValue> uint16_tensor_mutation_pool = {};
        std::vector<tensorflow::TensorValue> int32_tensor_mutation_pool = {};
        std::vector<tensorflow::TensorValue> uint32_tensor_mutation_pool = {};
        std::vector<tensorflow::TensorValue> int64_tensor_mutation_pool = {};
        std::vector<tensorflow::TensorValue> uint64_tensor_mutation_pool = {};
        std::vector<tensorflow::TensorValue> half_tensor_mutation_pool = {};
        std::vector<tensorflow::TensorValue> float_tensor_mutation_pool = {};
        std::vector<tensorflow::TensorValue> double_tensor_mutation_pool = {};
        std::vector<tensorflow::TensorValue> bool_tensor_mutation_pool = {};
        std::vector<tensorflow::TensorValue> string_tensor_mutation_pool = {};

        std::vector<int> pool_sizes = {};

        void initialize_tensor_pools();
        void calculate_total_mutations();
        void next_mutations_indices(bool log);
        void increase_num_crashes();
        inline void inc_mutations_indices(bool log);
        void restore_last_mutation(long long last_mutation, long long last_timestamp, bool do_resume);
        void log_current_mutation(std::fstream &file);
        void mark_fuzzing_done();
        void mark_unknown_type(tensorflow::DataType ttype);
        tensorflow::TensorValue *get_empty_tensor_with_shape(tensorflow::DataType ttype, tensorflow::TensorShape shape);
        template <class T> tensorflow::TensorValue *get_constant_tensor(T value);
        template <class T> tensorflow::TensorValue *get_tensor_with_value(T value, tensorflow::Tensor *tensor);
        template <class T> tensorflow::TensorValue *get_tensor_with_shape_and_value(T value, tensorflow::DataType ttype, tensorflow::TensorShape shape);
#endif

    public:

#if defined(IVYSYN_COLLECT_TYPES) || defined(IVYSYN_GPU_ONLY)
        Fuzzer(const std::string& fname, tensorflow::OpKernelContext* ctx, bool hasDevice, const char *device);
#else
        Fuzzer(const std::string& fname, tensorflow::OpKernelContext* ctx);
#endif

#if defined(IVYSYN_VALIDATE)
        bool should_validate();
        void false_positive();
        tensorflow::OpKernelContext *get_validate_context();
#endif

        ~Fuzzer();

#if !defined(IVYSYN_VALIDATE) && !defined(IVYSYN_COLLECT_TYPES)
        bool has_more_mutations(bool reset);
        tensorflow::TensorValue get_next_mut(tensorflow::DataType ttype, int idx);
        tensorflow::OpKernelContext *get_fuzzed_context();

        void mut_start_time();
        void mut_end_time(tensorflow::OpKernelContext *fuzz_ctx);
#endif

    };

}

/* #endif  // TENSORFLOW_CORE_FUZZING_H_ */
