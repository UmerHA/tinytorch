#include "tinytorch.h"
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

using bf16 = __nv_bfloat16;

static size_t dtype_size(DataType dtype) {
    switch (dtype) {
        case DataType::BF16: return 2;  // bf16 is 2 bytes
        case DataType::FP32: return 4;  // fp32 is 4 bytes
        case DataType::INT32: return 4; // int32 is 4 bytes
        default: throw std::runtime_error("Dtype currently not supported. Use FP32 or INT32 instead.");
    }
}

TinyTensor::TinyTensor() : m_data(nullptr) {}

TinyTensor::TinyTensor(void* data, const std::vector<int64_t>& sizes, DataType dtype) : m_sizes(sizes), m_dtype(dtype) {
    size_t element_size = dtype_size(dtype);
    size_t total_size = numel() * element_size;
    cudaMalloc(&m_data, total_size);
    cudaMemcpy(m_data, data, total_size, cudaMemcpyHostToDevice);
}
TinyTensor::TinyTensor(float* data, const std::vector<int64_t>& sizes) : TinyTensor((void*)data, sizes, DataType::FP32) {}

TinyTensor::~TinyTensor() { if (m_data) cudaFree(m_data); }


TinyTensor::TinyTensor(TinyTensor&& other) {           throw std::runtime_error("Move constructor isn't implemented for TinyTensor. Sorry bro.\n"); std::exit(EXIT_FAILURE); }
TinyTensor& TinyTensor::operator=(TinyTensor&&) {      throw std::runtime_error("Move assignment isn't implemented for TinyTensor. Sorry bro.\n");  std::exit(EXIT_FAILURE); }

TinyTensor::TinyTensor(const TinyTensor& other): m_sizes(other.m_sizes), m_dtype(other.m_dtype) {
    size_t total_size = other.numel() * dtype_size(m_dtype);
    cudaMalloc(&m_data, total_size);
    cudaMemcpy(m_data, other.m_data, total_size, cudaMemcpyDeviceToDevice);
}

TinyTensor& TinyTensor::operator=(const TinyTensor& other) {
    if (this != &other) {
        if (m_data) { cudaFree(m_data); }
        m_sizes = other.m_sizes;
        m_dtype = other.m_dtype;
        size_t total_size = other.numel() * dtype_size(m_dtype);
        cudaMalloc(&m_data, total_size);
        cudaMemcpy(m_data, other.m_data, total_size, cudaMemcpyDeviceToDevice);
    }
    return *this;
}


int64_t TinyTensor::dim() const { return m_sizes.size(); }
int64_t TinyTensor::size(int64_t dim) const { return m_sizes.at(dim); }
const std::vector<int64_t>& TinyTensor::sizes() const { return m_sizes; }

const std::string TinyTensor::sizes_as_str() const {
    std::ostringstream oss;
    oss << "(";
    for (size_t i = 0; i < m_sizes.size(); ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << m_sizes[i];
    }
    oss << ")";
    return oss.str();
}

template <typename T>       T* TinyTensor::data_ptr()       { return static_cast<T*>(m_data); }
template <typename T> const T* TinyTensor::data_ptr() const { return static_cast<T*>(m_data); }

template       bf16*  TinyTensor::data_ptr<bf16>();
template const bf16*  TinyTensor::data_ptr<bf16>() const;
template       float* TinyTensor::data_ptr<float>();
template const float* TinyTensor::data_ptr<float>() const;
template       int*   TinyTensor::data_ptr<int>();
template const int*   TinyTensor::data_ptr<int>() const;

int64_t TinyTensor::numel() const {
    int64_t size = 1;
    for (auto s : m_sizes) size *= s;
    return size;
}

void TinyTensor::print(int limit) const {
    // Create a host copy of the data
    size_t element_size = dtype_size(m_dtype);
    size_t total_size = numel() * element_size;
    void* host_data = std::malloc(total_size);
    if (!host_data) throw std::runtime_error("Failed to allocate host memory for printing.");
    
    cudaError_t cuda_status = cudaMemcpy(host_data, m_data, total_size, cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
        std::free(host_data);
        throw std::runtime_error("Failed to copy data from device to host for printing.");
    }

    size_t total_elements = numel();
    std::string limit_note = "";
    if (limit != -1 && limit < total_elements) {
        total_elements = limit;
        limit_note = " (first " + std::to_string(limit)+ ")";
    }
    std::cout << "TinyTensor of shape " << sizes_as_str() << " and flattened data" << limit_note << ": " << std::endl;
    for (size_t i = 0; i < total_elements; ++i) {
        switch (m_dtype) {
            case DataType::BF16:  std::cout << std::fixed << std::setprecision(2) << __bfloat162float(static_cast<const bf16* >(host_data)[i]) << " "; break;
            case DataType::FP32:  std::cout << std::fixed << std::setprecision(2) << static_cast<const float*>(host_data)[i] << " "; break;
            case DataType::INT32: std::cout << std::fixed << std::setprecision(2) << static_cast<const int*  >(host_data)[i] << " "; break;
        }
    }
    std::cout << std::endl << std::defaultfloat << std::setprecision(6); // reset precision
    std::free(host_data);
}
void TinyTensor::print() const {TinyTensor::print(-1);}

bool TinyTensor::is_contiguous() const { return true; }
DataType TinyTensor::options() const { return m_dtype ;}
DataType TinyTensor::dtype()   const { return m_dtype ;}

TinyTensor constants(const std::vector<int64_t>& sizes, float value, DataType dtype) {
    size_t element_size = dtype_size(dtype);
    size_t total_size = 1;
    for (auto s : sizes) total_size *= s;
    
    size_t total_bytes = total_size * element_size;
    void* host_data = std::malloc(total_bytes);
    
    if (!host_data) throw std::runtime_error("Failed to allocate memory while creating TinyTesnor of constants.");

    for (size_t i = 0; i < total_size; ++i) {
        if      (dtype == DataType::BF16)  static_cast<bf16*>(host_data)[i]  = static_cast<bf16>(value);
        else if (dtype == DataType::FP32)  static_cast<float*>(host_data)[i] = static_cast<float>(value);
        else if (dtype == DataType::INT32) static_cast<int*>  (host_data)[i] = static_cast<int>  (value);
    }

    TinyTensor result(host_data, sizes, dtype);
    std::free(host_data);
    return result;
}
TinyTensor zeros(const std::vector<int64_t>& sizes, DataType dtype) { return constants(sizes, 0, dtype); }
TinyTensor ones (const std::vector<int64_t>& sizes, DataType dtype) { return constants(sizes, 1, dtype); }
