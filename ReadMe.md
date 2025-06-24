# TinyTorch 🔥🤏

TinyTorch is a super lightweight tensor helper that lets you write and test CUDA
kernels *without pulling the full PyTorch build chain*.  

It covers the 90% use‑case for fast iteration:
* **Device storage** – `cudaMalloc`/`cudaFree` handled for you.  
* **Types** – `float32`, `bfloat16`, `int32`.  
* **Copy helpers** – host↔device and tensor↔tensor.  
* **Simple API** – `zeros`, `ones`, `constants`, `sizes()`, `numel()`, `print()`.

## Quick start

```bash
# Clone
git clone https://github.com/your‑org/tinytorch.git
cd tinytorch

# Build the sample (needs CUDA ≥11.0, C++17)
nvcc -std=c++17 tinyutil.cpp tinytorch.cpp -o tinyutil
./tinyutil     # runs a small demo and prints shapes / data
```

## Writing a kernel

```cpp
#include "tinytorch.h"

__global__ void scale(float* x, int64_t n, float s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= s;
}

void scale_tensor(TinyTensor& t, float s) {
    TINYTORCH_CHECK(t.dtype() == DataType::FP32, "scale expects fp32");
    int64_t n = t.numel();
    int bs = 256;
    scale<<<(n + bs - 1)/bs, bs>>>(t.data_ptr<float>(), n, s);
}
```

Compile just like the demo, link nothing else.

## Limitations
* Only contiguous tensors on a single CUDA device.
* Move semantics are disabled (copy is fine).
* No CPU fallback, no autograd, no broadcasting.


## Roadmap
I'll only be adding more features to TinyTorch as I need them for my work. But contributions are welcome!
