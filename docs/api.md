
# TinyTorch API

TinyTorch is two files:

* `tinytorch.h` – public interface.  
* `tinytorch.cpp` – small implementation.

Everything lives in the `TinyTensor` class.


## 1. Core types
| Type                     | Purpose                       |
|--------------------------|-------------------------------|
| `TinyTensor`             | Device tensor wrapper.        |
| `DataType`               | `BF16`, `FP32`, `INT32`.      |
| `CudaDevice`             | Dummy device descriptor.      |

### `dtype_size(DataType) → size_t`

Returns bytes per element.

---

## 2. `TinyTensor` interface

| Method | Note |
|--------|------|
| **ctors** |
| `TinyTensor()` | default, empty. |
| `TinyTensor(void* host_data, sizes, dtype)` | copies host→device. |
| `TinyTensor(float* host, sizes)` | fp32 helper. |
| `TinyTensor(int* host, sizes)` | int32 helper. |
| **meta** |
| `int64_t dim()` | # of dimensions. |
| `const vector<int64_t>& sizes()` | shape. |
| `int64_t numel()` | total elements. |
| `DataType dtype()` | storage type. |
| `bool is_contiguous()` | always `true` in v0. |
| **data access** |
| `T* data_ptr<T>()` | raw pointer on device. |
| **utils** |
| `void print(int limit=-1)` | dumps flattened data to stdout. |

Copy ctor and copy assignment are implemented.  
Move ctor/assignment intentionally `throw`; this prevents silent use‑after‑free
bugs when experimenting.

---

## 3. Factory helpers

```cpp
TinyTensor zeros(sizes, dtype);
TinyTensor ones (sizes, dtype);
TinyTensor constants(sizes, float value, dtype);
```

All three allocate on device and fill via a pinned host buffer.

---

## 4. Dispatch macros

* `TINYTORCH_CHECK(cond, msg...)` – lightweight assert.  
* `TINYTORCH_DISPATCH_BY_TYPE(dtype, lambda_body)` – cheap type switch.

Example:

```cpp
TINYTORCH_DISPATCH_BY_TYPE(t.dtype(), {
    // scalar_t is float or bf16, depending on t.dtype()
    my_kernel<<<...>>>(t.data_ptr<scalar_t>(), ...);
});
```

---

## 5. Writing tests

1. Use Catch2 (single header, already vendored in `tests/`).  
2. Always launch a CUDA kernel under test, then copy to host and compare.  
3. Keep GPU memory small; CI runners may not have large cards.

---

## 6. FAQ

**Q: Can I use it from Python?**  
A: Not now. The first goal is *zero‑dependency* kernel work. Bindings may come
later.

**Q: Does it work on Windows?**  
A: Yes—`nvcc` + `MSVC` is fine. Paths in the sample assume POSIX; adjust.
