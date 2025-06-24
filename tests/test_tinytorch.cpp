
#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"

#include "../tinytorch.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <vector>
#include <sstream>
#include <iostream>

// ===== Helpers =============================================================

// Copy TinyTensor (fp32) device → host
std::vector<float> to_host_fp32(const TinyTensor& t) {
    REQUIRE(t.dtype() == DataType::FP32);
    std::vector<float> h(t.numel());
    cudaMemcpy(h.data(), t.data_ptr<float>(),
               h.size() * sizeof(float), cudaMemcpyDeviceToHost);
    return h;
}

// Copy TinyTensor (int32) device → host
std::vector<int32_t> to_host_int32(const TinyTensor& t) {
    REQUIRE(t.dtype() == DataType::INT32);
    std::vector<int32_t> h(t.numel());
    cudaMemcpy(h.data(), t.data_ptr<int32_t>(),
               h.size() * sizeof(int32_t), cudaMemcpyDeviceToHost);
    return h;
}

// Copy TinyTensor (bf16) device → host and convert to float
std::vector<float> to_host_bf16(const TinyTensor& t) {
    REQUIRE(t.dtype() == DataType::BF16);
    std::vector<bf16> h_raw(t.numel());
    cudaMemcpy(h_raw.data(), t.data_ptr<bf16>(),
               h_raw.size() * sizeof(bf16), cudaMemcpyDeviceToHost);
    std::vector<float> h_fp32(h_raw.size());
    for (size_t i = 0; i < h_raw.size(); ++i) h_fp32[i] = __bfloat162float(h_raw[i]);
    return h_fp32;
}

// ===== Device kernels ======================================================

__global__ void scale_kernel(float* x, int64_t n, float s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= s;
}

__global__ void idx_kernel_int(int32_t* x, int64_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = i;
}

// ===== Functions under test ===============================================

void scale_tensor(TinyTensor& t, float s) {
    TINYTORCH_CHECK(t.dtype() == DataType::FP32,
                    "scale expects fp32 tensor, got something else.");
    int64_t n = t.numel();
    int bs = 256;
    scale_kernel<<<(n + bs - 1)/bs, bs>>>(t.data_ptr<float>(), n, s);
    cudaDeviceSynchronize();
}

// ===== Test cases ==========================================================

TEST_CASE("zeros/ones/constants produce correct data for FP32") {
    std::vector<int64_t> shape = {2, 3};
    auto z = zeros(shape, DataType::FP32);
    auto o = ones (shape, DataType::FP32);
    auto c = constants(shape, 42.f, DataType::FP32);

    for (float v : to_host_fp32(z)) REQUIRE(v == 0.f);
    for (float v : to_host_fp32(o)) REQUIRE(v == 1.f);
    for (float v : to_host_fp32(c)) REQUIRE(v == 42.f);
}

TEST_CASE("factory helpers work for all dtypes") {
    std::vector<int64_t> shape = {1};
    struct Helper {
        TinyTensor (*fn)(const std::vector<int64_t>&, DataType);
        float expected;
    };
    std::vector<Helper> helpers = {
        {zeros, 0.f},
        {ones , 1.f},
    };

    for (auto dtype : {DataType::FP32, DataType::BF16, DataType::INT32}) {
        for (const auto& h : helpers) {
            auto t = h.fn(shape, dtype);
            if (dtype == DataType::FP32) {
                REQUIRE(to_host_fp32(t)[0] == h.expected);
            } else if (dtype == DataType::BF16) {
                REQUIRE(to_host_bf16(t)[0] == h.expected);
            } else {
                REQUIRE(to_host_int32(t)[0] == static_cast<int32_t>(h.expected));
            }
        }
    }
}

TEST_CASE("copy constructor performs deep copy") {
    auto a = ones({4}, DataType::FP32);
    auto b(a);  // copy ctor
    REQUIRE(a.data_ptr<float>() != b.data_ptr<float>());

    float two = 2.f;
    cudaMemcpy(a.data_ptr<float>(), &two, sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    REQUIRE(to_host_fp32(a)[0] == 2.f);
    REQUIRE(to_host_fp32(b)[0] == 1.f);
}

TEST_CASE("copy assignment performs deep copy") {
    auto a = ones({4}, DataType::FP32);
    auto b = zeros({4}, DataType::FP32);
    b = a;  // assignment
    REQUIRE(a.data_ptr<float>() != b.data_ptr<float>());

    float three = 3.f;
    cudaMemcpy(a.data_ptr<float>(), &three, sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    REQUIRE(to_host_fp32(a)[0] == 3.f);
    REQUIRE(to_host_fp32(b)[0] == 1.f);
}

TEST_CASE("scale_tensor scales and errors correctly") {
    auto t = ones({10}, DataType::FP32);
    scale_tensor(t, 2.f);
    for (float v: to_host_fp32(t)) REQUIRE(v == 2.f);

    auto bad = ones({10}, DataType::BF16);
    REQUIRE_THROWS_AS(scale_tensor(bad, 2.f), std::runtime_error);
}

TEST_CASE("kernel round‑trip INT32 index fill") {
    int64_t n = 128;
    auto t = zeros({n}, DataType::INT32);
    int bs = 128;
    idx_kernel_int<<<(n + bs - 1)/bs, bs>>>(t.data_ptr<int32_t>(), n);
    cudaDeviceSynchronize();

    auto h = to_host_int32(t);
    for (int i = 0; i < n; ++i) REQUIRE(h[i] == i);
}

TEST_CASE("dtype_size returns correct sizes") {
    REQUIRE(dtype_size(DataType::FP32) == 4);
    REQUIRE(dtype_size(DataType::BF16) == 2);
    REQUIRE(dtype_size(DataType::INT32) == 4);
}

TEST_CASE("stress loop create/destroy") {
    for (int iter = 0; iter < 1000; ++iter) {
        auto t = zeros({16, 16}, DataType::FP32);
        (void)t;
    }
    cudaDeviceSynchronize();
}

TEST_CASE("print utility runs without crash") {
    auto t = constants({2}, 7.f, DataType::FP32);
    t.print();
}
