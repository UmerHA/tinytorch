#ifndef TINYTORCH_H
#define TINYTORCH_H

#include <vector>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <sstream>
#include <cuda_bf16.h>

using bf16 = __nv_bfloat16;

template <class... Ts>
std::string joint_str(Ts const&... args) {
    std::ostringstream oss;
    ((oss << args), ...);
    return oss.str();
}

#define TINYTORCH_CHECK(condition, ...) \
    do { \
        if (!(condition)) { throw std::runtime_error(joint_str("Assertion failed: ", __VA_ARGS__, "\nFile: ", __FILE__, "\nLine: ", __LINE__)); } \
    } while (0)

#define TINYTORCH_DISPATCH_CASE(enum_type, type, ...)   \
    case enum_type: {                                   \
        using scalar_t = type;                          \
        __VA_ARGS__                                     \
        break;                                          \
    }

#define TINYTORCH_DISPATCH_BY_TYPE(TYPE, ...)                       \
    switch (TYPE) {                                                 \
        TINYTORCH_DISPATCH_CASE(DataType::FP32, float, __VA_ARGS__) \
        TINYTORCH_DISPATCH_CASE(DataType::BF16, bf16,  __VA_ARGS__) \
        default: throw std::runtime_error("Type not supported in dispatcher, use fp32 or bf16."); \
    }

class CudaDevice {
public:
    bool is_cuda() const { return true; }
};

enum class DataType {
    BF16,
    FP32,
    INT32,
};

static size_t dtype_size(DataType dtype);

class TinyTensor {
public:
    TinyTensor();
    TinyTensor(void* data, const std::vector<int64_t>& sizes, DataType dtype);
    TinyTensor(float* data, const std::vector<int64_t>& sizes);
    TinyTensor(int* data, const std::vector<int64_t>& sizes);
    ~TinyTensor();

    TinyTensor(const TinyTensor&);
    TinyTensor(TinyTensor&& other);
    TinyTensor& operator=(const TinyTensor&);
    TinyTensor& operator=(TinyTensor&&);

    int64_t dim() const;
    int64_t size(int64_t dim) const;
    const std::vector<int64_t>& sizes() const;
    const std::string sizes_as_str() const;
    template <typename T>       T* data_ptr(); 
    template <typename T> const T* data_ptr() const;
    int64_t numel() const;
    void print(int limit) const;
    void print() const;

    CudaDevice device() const { return CudaDevice(); }
    bool is_contiguous() const;

    DataType options() const;
    DataType dtype() const;

private:
    std::vector<int64_t> m_sizes;
    void* m_data;
    DataType m_dtype;
};

TinyTensor constants(const std::vector<int64_t>& sizes, float value, DataType dtype);
TinyTensor zeros(const std::vector<int64_t>& sizes, DataType dtype);
TinyTensor ones (const std::vector<int64_t>& sizes, DataType dtype);

#endif // TINYTORCH_H
