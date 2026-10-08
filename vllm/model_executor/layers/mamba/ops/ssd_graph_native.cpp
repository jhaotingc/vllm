// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <cuda.h>
#include <torch/csrc/autograd/python_variable.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

static void check_cuda(CUresult result) {
  if (result != CUDA_SUCCESS) {
    const char* message = nullptr;
    cuGetErrorString(result, &message);
    throw std::runtime_error(message ? message : "CUDA driver error");
  }
}

static const at::Tensor& unpack(py::handle object) {
  if (!THPVariable_Check(object.ptr())) {
    throw std::runtime_error("SSD graph arguments must be tensors or None");
  }
  return THPVariable_Unpack(object.ptr());
}

static std::vector<int64_t> signature(const py::tuple& arguments) {
  if (arguments.size() != 14) throw std::runtime_error("Expected 14 tensors");
  std::vector<int64_t> key;
  std::array<uintptr_t, 14> addresses{};
  int64_t rows = unpack(arguments[0]).size(0);
  for (size_t i = 0; i < arguments.size(); ++i) {
    if (arguments[i].is_none()) {
      key.push_back(-1);
      continue;
    }
    const auto& tensor = unpack(arguments[i]);
    if (!tensor.is_cuda()) throw std::runtime_error("Expected CUDA tensor");
    uintptr_t address = reinterpret_cast<uintptr_t>(tensor.data_ptr());
    addresses[i] = address;
    key.push_back(static_cast<int64_t>(tensor.scalar_type()));
    key.push_back(tensor.get_device());
    key.push_back(tensor.dim());
    key.push_back(address % 16);  // Preserve Triton's pointer alignment class.
    size_t alias = i;
    for (size_t j = 0; j < i; ++j) {
      if (!arguments[j].is_none() && addresses[j] == address) {
        alias = j;
        break;
      }
    }
    key.push_back(alias);
    bool token_rows = i == 0 || i == 1 || i == 3 || i == 4 || i == 5 || i == 7;
    if (token_rows && tensor.size(0) != rows) {
      throw std::runtime_error("SSD token dimensions disagree");
    }
    // These five Triton kernels never take total token count as a launch
    // parameter. Chunk boundaries are read from live cu_chunk_seqlens.
    for (int64_t d = 0; d < tensor.dim(); ++d) {
      key.push_back(token_rows && d == 0 ? -2 : tensor.size(d));
      key.push_back(tensor.stride(d));
    }
    if ((i == 3 || i == 4 || i == 6) && tensor.stride(-1) != 1) {
      throw std::runtime_error("Contiguous-copy SSD path is not graphed");
    }
    if ((i == 0 || i == 7) && tensor.stride(-1) != 1 && tensor.stride(0) != 1) {
      throw std::runtime_error("Contiguous-copy SSD path is not graphed");
    }
  }
  return key;
}

struct Relocation {
  size_t argument;
  size_t offset;
  size_t tensor;
};

struct KernelNode {
  CUgraphNode node{};
  CUDA_KERNEL_NODE_PARAMS parameters{};
  std::vector<std::vector<unsigned char>> values;
  std::vector<void*> pointers;
  std::vector<Relocation> relocations;
};

class GraphRebinder {
 public:
  GraphRebinder(uintptr_t graph_address, uintptr_t exec_address,
                const py::tuple& arguments)
      : executable_(reinterpret_cast<CUgraphExec>(exec_address)),
        signature_(signature(arguments)) {
    std::array<uintptr_t, 14> addresses{};
    std::array<size_t, 14> uses{};
    for (size_t i = 0; i < arguments.size(); ++i) {
      if (!arguments[i].is_none()) {
        addresses[i] =
            reinterpret_cast<uintptr_t>(unpack(arguments[i]).data_ptr());
      }
    }
    CUgraph graph = reinterpret_cast<CUgraph>(graph_address);
    size_t count = 0;
    check_cuda(cuGraphGetNodes(graph, nullptr, &count));
    std::vector<CUgraphNode> nodes(count);
    check_cuda(cuGraphGetNodes(graph, nodes.data(), &count));
    kernels_.reserve(count);
    for (auto node : nodes) {
      CUgraphNodeType type;
      check_cuda(cuGraphNodeGetType(node, &type));
      if (type == CU_GRAPH_NODE_TYPE_EMPTY) continue;
      if (type != CU_GRAPH_NODE_TYPE_KERNEL) {
        throw std::runtime_error("SSD graph contains a non-kernel node");
      }
      kernels_.emplace_back();
      auto& kernel = kernels_.back();
      kernel.node = node;
      check_cuda(cuGraphKernelNodeGetParams(node, &kernel.parameters));
      if (!kernel.parameters.func || !kernel.parameters.kernelParams ||
          kernel.parameters.extra) {
        throw std::runtime_error(
            "Unsupported graph kernel parameter representation");
      }
      size_t parameters = 0;
      check_cuda(cuFuncGetParamCount(kernel.parameters.func, &parameters));
      if (parameters > 128)
        throw std::runtime_error("Unexpected parameter count");
      kernel.values.resize(parameters);
      kernel.pointers.resize(parameters);
      for (size_t p = 0; p < parameters; ++p) {
        size_t offset = 0;
        size_t size = 0;
        check_cuda(
            cuFuncGetParamInfo(kernel.parameters.func, p, &offset, &size));
        if (size > 1024) throw std::runtime_error("Unexpected argument size");
        kernel.values[p].resize(size);
        std::memcpy(kernel.values[p].data(), kernel.parameters.kernelParams[p],
                    size);
        kernel.pointers[p] = kernel.values[p].data();
        for (size_t b = 0; b + sizeof(uintptr_t) <= size;
             b += sizeof(uintptr_t)) {
          uintptr_t value = 0;
          std::memcpy(&value, kernel.values[p].data() + b, sizeof(value));
          for (size_t i = 0; i < addresses.size(); ++i) {
            if (addresses[i] && value == addresses[i]) {
              kernel.relocations.push_back({p, b, i});
              ++uses[i];
              break;
            }
          }
        }
      }
      kernel.parameters.kernelParams = kernel.pointers.data();
    }
    if (kernels_.size() != 5)
      throw std::runtime_error("Expected exactly five SSD kernels");
    for (size_t i = 0; i < uses.size(); ++i) {
      // seq_idx may be optimized away; cu_seqlens supplies a host-side size
      // only.
      if (i != 10 && i != 11 && addresses[i] && !uses[i]) {
        bool earlier_alias = false;
        for (size_t j = 0; j < i; ++j)
          earlier_alias |= addresses[j] == addresses[i];
        if (!earlier_alias)
          throw std::runtime_error("A live SSD tensor has no relocation");
      }
    }
    uses_ = uses;
  }

  void replay(const py::tuple& arguments, uintptr_t stream_address) {
    if (signature(arguments) != signature_) {
      throw std::runtime_error("SSD graph metadata or alias pattern changed");
    }
    std::array<uintptr_t, 14> addresses{};
    for (size_t i = 0; i < arguments.size(); ++i) {
      if (!arguments[i].is_none()) {
        addresses[i] =
            reinterpret_cast<uintptr_t>(unpack(arguments[i]).data_ptr());
      }
    }
    for (auto& kernel : kernels_) {
      if (kernel.relocations.empty()) continue;
      for (const auto& relocation : kernel.relocations) {
        auto address = addresses[relocation.tensor];
        std::memcpy(
            kernel.values[relocation.argument].data() + relocation.offset,
            &address, sizeof(address));
      }
      check_cuda(cuGraphExecKernelNodeSetParams(executable_, kernel.node,
                                                &kernel.parameters));
    }
    check_cuda(
        cuGraphLaunch(executable_, reinterpret_cast<CUstream>(stream_address)));
  }

  std::vector<size_t> relocation_counts() const {
    return std::vector<size_t>(uses_.begin(), uses_.end());
  }

 private:
  CUgraphExec executable_;
  std::vector<int64_t> signature_;
  std::vector<KernelNode> kernels_;
  std::array<size_t, 14> uses_{};
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("signature", [](const py::tuple& arguments) {
    auto key = signature(arguments);
    return py::bytes(reinterpret_cast<const char*>(key.data()),
                     key.size() * sizeof(int64_t));
  });
  py::class_<GraphRebinder>(module, "GraphRebinder")
      .def(py::init<uintptr_t, uintptr_t, const py::tuple&>())
      .def("replay", &GraphRebinder::replay)
      .def("relocation_counts", &GraphRebinder::relocation_counts);
}
