// Copyright © 2023 Apple Inc.
//
#include <cerrno>
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <json.hpp>
#include <memory>
#include <mutex>
#include <optional>
#include <regex>
#include <stack>
#include <vector>

#include "mlx/allocator.h"
#include "mlx/backend/cuda/cuda.h"
#include "mlx/io.h"
#include "mlx/io/load.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/transforms.h"

#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

using json = nlohmann::json;

#define ST_F16 "F16"
#define ST_BF16 "BF16"
#define ST_F32 "F32"

#define ST_BOOL "BOOL"
#define ST_I8 "I8"
#define ST_I16 "I16"
#define ST_I32 "I32"
#define ST_I64 "I64"
#define ST_U8 "U8"
#define ST_U16 "U16"
#define ST_U32 "U32"
#define ST_U64 "U64"
#define ST_F8_E4M3 "F8_E4M3"

// Note: Complex numbers aren't in the spec yet so this could change -
// https://github.com/huggingface/safetensors/issues/389
#define ST_C64 "C64"

namespace mlx::core {

std::string dtype_to_safetensor_str(Dtype t) {
  switch (t) {
    case float32:
      return ST_F32;
    case bfloat16:
      return ST_BF16;
    case float16:
      return ST_F16;
    case int64:
      return ST_I64;
    case int32:
      return ST_I32;
    case int16:
      return ST_I16;
    case int8:
      return ST_I8;
    case uint64:
      return ST_U64;
    case uint32:
      return ST_U32;
    case uint16:
      return ST_U16;
    case uint8:
      return ST_U8;
    case bool_:
      return ST_BOOL;
    case complex64:
      return ST_C64;
    default:
      throw std::runtime_error("[save_safetensors] received invalid dtype.");
  }
}

Dtype dtype_from_safetensor_str(std::string_view str) {
  if (str == ST_F32) {
    return float32;
  } else if (str == ST_F16) {
    return float16;
  } else if (str == ST_BF16) {
    return bfloat16;
  } else if (str == ST_I64) {
    return int64;
  } else if (str == ST_I32) {
    return int32;
  } else if (str == ST_I16) {
    return int16;
  } else if (str == ST_I8) {
    return int8;
  } else if (str == ST_U64) {
    return uint64;
  } else if (str == ST_U32) {
    return uint32;
  } else if (str == ST_U16) {
    return uint16;
  } else if (str == ST_U8) {
    return uint8;
  } else if (str == ST_BOOL) {
    return bool_;
  } else if (str == ST_C64) {
    return complex64;
  } else if (str == ST_F8_E4M3) {
    return uint8;
  } else {
    throw std::runtime_error(
        "[safetensor] unsupported dtype " + std::string(str));
  }
}

#ifndef _WIN32
namespace {

bool vmlx_mmap_safetensors_enabled() {
  const char* env = std::getenv("VMLX_MMAP_SAFETENSORS");
  return env && std::string_view(env) == "1";
}

bool vmlx_mmap_safetensors_debug() {
  const char* env = std::getenv("VMLX_MMAP_SAFETENSORS_DEBUG");
  return env && std::string_view(env) == "1";
}

struct VmlxMmapTensorRecord {
  void* mapped;
  size_t map_len;
  size_t page_delta;
  size_t tensor_nbytes;
  std::string name;
  Shape shape;
  bool active;
};

std::mutex& vmlx_mmap_registry_mutex() {
  static auto* m = new std::mutex();
  return *m;
}

std::vector<VmlxMmapTensorRecord>& vmlx_mmap_registry() {
  static auto* r = new std::vector<VmlxMmapTensorRecord>();
  return *r;
}

void vmlx_register_mmap_tensor(
    void* mapped,
    size_t map_len,
    size_t page_delta,
    size_t tensor_nbytes,
    const std::string& name,
    const Shape& shape) {
  std::lock_guard<std::mutex> guard(vmlx_mmap_registry_mutex());
  vmlx_mmap_registry().push_back(
      {mapped, map_len, page_delta, tensor_nbytes, name, shape, true});
}

void vmlx_unregister_mmap_tensor(void* mapped, size_t map_len) {
  std::lock_guard<std::mutex> guard(vmlx_mmap_registry_mutex());
  auto& registry = vmlx_mmap_registry();
  registry.erase(
      std::remove_if(
          registry.begin(),
          registry.end(),
          [&](const auto& r) {
            return r.mapped == mapped && r.map_len == map_len;
          }),
      registry.end());
}

int vmlx_jangpress_madvise_value(int32_t advice) {
  if (advice == 1) {
    return MADV_WILLNEED;
  }
  if (advice == 0) {
    return MADV_DONTNEED;
  }
  return static_cast<int>(advice);
}

int64_t vmlx_advise_page_range(void* addr, size_t nbytes, int32_t advice) {
  if (addr == nullptr || nbytes == 0) {
    return 0;
  }
  const long page_size_l = sysconf(_SC_PAGESIZE);
  const uintptr_t page_size = page_size_l > 0
      ? static_cast<uintptr_t>(page_size_l)
      : static_cast<uintptr_t>(4096);
  const uintptr_t start = reinterpret_cast<uintptr_t>(addr);
  const uintptr_t aligned_start = start - (start % page_size);
  const uintptr_t end = start + nbytes;
  const uintptr_t aligned_end =
      ((end + page_size - 1) / page_size) * page_size;
  const size_t len = static_cast<size_t>(aligned_end - aligned_start);
  if (madvise(reinterpret_cast<void*>(aligned_start), len,
              vmlx_jangpress_madvise_value(advice)) != 0) {
    return 0;
  }
  return static_cast<int64_t>(nbytes);
}

struct VmlxParsedRoutedTensor {
  bool matched = false;
  bool stacked = false;
  int layer = -1;
  int expert = -1;
};

std::optional<std::pair<int, int>> vmlx_match_two_ints(
    const std::string& name,
    const std::regex& re) {
  std::smatch m;
  try {
    if (!std::regex_match(name, m, re)) {
      return std::nullopt;
    }
  } catch (const std::regex_error&) {
    return std::nullopt;
  }
  std::vector<int> nums;
  for (size_t i = 1; i < m.size(); ++i) {
    const auto s = m[i].str();
    if (!s.empty() &&
        std::all_of(s.begin(), s.end(), [](unsigned char c) {
          return std::isdigit(c) != 0;
        })) {
      nums.push_back(std::stoi(s));
    }
  }
  if (nums.size() < 2) {
    return std::nullopt;
  }
  return std::make_pair(nums[0], nums[1]);
}

std::optional<int> vmlx_match_one_int(
    const std::string& name,
    const std::regex& re) {
  std::smatch m;
  try {
    if (!std::regex_match(name, m, re)) {
      return std::nullopt;
    }
  } catch (const std::regex_error&) {
    return std::nullopt;
  }
  for (size_t i = 1; i < m.size(); ++i) {
    const auto s = m[i].str();
    if (!s.empty() &&
        std::all_of(s.begin(), s.end(), [](unsigned char c) {
          return std::isdigit(c) != 0;
        })) {
      return std::stoi(s);
    }
  }
  return std::nullopt;
}

VmlxParsedRoutedTensor vmlx_parse_routed_tensor_name(const std::string& name) {
  // The canonical mmap registry contains every tensor, not just routed MoE
  // weights. Avoid feeding attention, norms, and bias tensors through the
  // heavier regex table; libc++ std::regex can throw error_complexity at match
  // time on long non-matching names.
  const bool maybe_routed =
      name.find(".mlp.experts.") != std::string::npos ||
      name.find(".ffn.experts.") != std::string::npos ||
      name.find(".ffn.switch_mlp.") != std::string::npos ||
      name.find(".block_sparse_moe.experts.") != std::string::npos ||
      name.find(".block_sparse_moe.switch_mlp.") != std::string::npos ||
      name.find(".mlp.switch_mlp.") != std::string::npos ||
      name.find(".switch_mlp.") != std::string::npos ||
      name.find(".mixer.experts.") != std::string::npos ||
      name.find(".mixer.switch_mlp.") != std::string::npos;
  if (!maybe_routed) {
    return {};
  }

  // std::regex defaults to ECMAScript, which does not support PCRE
  // non-capturing groups. Keep patterns ECMAScript-valid and let
  // vmlx_match_* extract the numeric captures so prefix/alternation captures do
  // not shift the layer/expert fields.
  static const std::string vl_prefix = R"(((model|language_model)\.)*)";
  // Leak these regex tables for process lifetime. The router advisor runs on a
  // background queue and may still drain observations while the executable is
  // exiting; destructing function-local static regexes at exit can race that
  // queue and produce use-after-free inside libc++ regex.
  static const auto& per_expert = *new std::vector<std::regex>{
      std::regex("^" + vl_prefix +
                 R"(layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.(weight|tq_packed|tq_norms)$)"),
      std::regex(R"(^layers\.(\d+)\.ffn\.experts\.(\d+)\.w[123]\.(weight|tq_packed|tq_norms)$)"),
      std::regex(R"(^layers\.(\d+)\.ffn\.experts\.(\d+)\.(gate|up|down)_proj\.(weight|tq_packed|tq_norms)$)"),
      std::regex("^" + vl_prefix +
                 R"(layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.w[123]\.(weight|tq_packed|tq_norms)$)"),
      std::regex(R"(^backbone\.layers\.(\d+)\.mixer\.experts\.(\d+)\.(gate|up|down)_proj\.(weight|tq_packed|tq_norms)$)")};
  for (const auto& re : per_expert) {
    if (auto v = vmlx_match_two_ints(name, re)) {
      return {true, false, v->first, v->second};
    }
  }

  static const auto& stacked = *new std::vector<std::regex>{
      std::regex("^" + vl_prefix +
                 R"(layers\.(\d+)\.mlp\.switch_mlp\.(gate|up|down)_proj\.(weight|scales|biases|tq_packed|tq_norms)$)"),
      std::regex(R"(^layers\.(\d+)\.ffn\.switch_mlp\.(gate|up|down)_proj\.(weight|scales|biases|tq_packed|tq_norms)$)"),
      std::regex("^" + vl_prefix +
                 R"(layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj|gate_proj|up_proj)\.(weight|scales|biases|tq_packed|tq_norms)$)"),
      std::regex("^" + vl_prefix +
                 R"(layers\.(\d+)\.block_sparse_moe\.switch_mlp\.(gate|up|down)_proj\.(weight|scales|biases|tq_packed|tq_norms)$)"),
      std::regex("^" + vl_prefix +
                 R"(layers\.(\d+)\.switch_mlp\.(gate|up|down)_proj\.(weight|scales|biases|tq_packed|tq_norms)$)"),
      std::regex(R"(^backbone\.layers\.(\d+)\.mixer\.switch_mlp\.(fc[12]\.(weight|tq_packed|tq_norms)|(gate|up|down)_proj\.weight)$)")};
  for (const auto& re : stacked) {
    if (auto layer = vmlx_match_one_int(name, re)) {
      return {true, true, *layer, 0};
    }
  }
  return {};
}

int64_t vmlx_advise_record_expert(
    const VmlxMmapTensorRecord& r,
    int expert,
    int32_t advice) {
  auto parsed = vmlx_parse_routed_tensor_name(r.name);
  if (!parsed.matched) {
    return 0;
  }
  size_t byte_begin = 0;
  size_t byte_len = r.tensor_nbytes;
  if (parsed.stacked) {
    if (r.shape.empty() || r.shape[0] <= 0) {
      return 0;
    }
    const size_t num_experts = static_cast<size_t>(r.shape[0]);
    if (expert < 0 || static_cast<size_t>(expert) >= num_experts) {
      return 0;
    }
    const size_t per_expert = r.tensor_nbytes / num_experts;
    if (per_expert == 0 || per_expert * num_experts != r.tensor_nbytes) {
      return 0;
    }
    byte_begin = per_expert * static_cast<size_t>(expert);
    byte_len = per_expert;
  } else if (parsed.expert != expert) {
    return 0;
  }
  auto* addr = static_cast<char*>(r.mapped) + r.page_delta + byte_begin;
  return vmlx_advise_page_range(addr, byte_len, advice);
}

std::optional<array> make_mmap_backed_array(
    const std::string& file,
    const array& holder,
    void* mapped,
    size_t map_len,
    const std::string& tensor_name,
    const Shape& shape,
    Dtype type,
    size_t absolute_offset) {
  size_t element_count = 1;
  for (auto dim : shape) {
    if (dim < 0) {
      return std::nullopt;
    }
    element_count *= static_cast<size_t>(dim);
  }
  const size_t item_size = size_of(type);
  const size_t tensor_nbytes = element_count * item_size;
  if (tensor_nbytes == 0 || item_size == 0) {
    return std::nullopt;
  }

  // array::copy_shared_buffer takes element offsets, not byte offsets.
  // Safetensors payloads should be dtype-aligned; if a shard violates
  // that, fall back to the original copy loader rather than shift bytes.
  if ((absolute_offset % item_size) != 0) {
    if (vmlx_mmap_safetensors_debug()) {
      std::fprintf(
          stderr,
          "[VMLX_MMAP_SAFETENSORS] fallback %s offset=%zu not aligned to item_size=%zu\n",
          file.c_str(),
          absolute_offset,
          item_size);
    }
    return std::nullopt;
  }
  if (absolute_offset > map_len || tensor_nbytes > map_len - absolute_offset) {
    if (vmlx_mmap_safetensors_debug()) {
      std::fprintf(
          stderr,
          "[VMLX_MMAP_SAFETENSORS] fallback %s tensor=%s offset=%zu len=%zu beyond mapped shard len=%zu\n",
          file.c_str(),
          tensor_name.c_str(),
          absolute_offset,
          tensor_nbytes,
          map_len);
    }
    return std::nullopt;
  }

  array out(allocator::Buffer(nullptr), shape, type, [](allocator::Buffer) {});
  out.copy_shared_buffer(
      holder,
      out.strides(),
      out.flags(),
      element_count,
      static_cast<int64_t>(absolute_offset / item_size));
  vmlx_register_mmap_tensor(
      mapped, map_len, absolute_offset, tensor_nbytes, tensor_name, shape);
  return out;
}

SafetensorsLoad load_safetensors_mmap(const std::string& file, StreamOrDevice s) {
  int fd = open(file.c_str(), O_RDONLY | O_BINARY);
  if (fd < 0) {
    throw std::runtime_error(
        "[load_safetensors_mmap] failed to open " + file + ": " +
        std::strerror(errno));
  }

  struct stat st {};
  if (fstat(fd, &st) != 0 || st.st_size <= 0) {
    int err = errno;
    close(fd);
    throw std::runtime_error(
        "[load_safetensors_mmap] failed to stat " + file + ": " +
        std::strerror(err));
  }
  const size_t file_size = static_cast<size_t>(st.st_size);

  void* mapped = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  array shard_holder(
      allocator::Buffer(nullptr), Shape{1}, uint8, [](allocator::Buffer) {});
  bool whole_shard_mmap = false;
  if (mapped != MAP_FAILED) {
    auto buffer = allocator::make_buffer(mapped, file_size);
    if (buffer.ptr() == nullptr) {
      munmap(mapped, file_size);
      mapped = MAP_FAILED;
    } else {
      array::Flags flags{true, true, true};
      Strides one_dim_stride{1};
      shard_holder.set_data(
          buffer,
          file_size,
          one_dim_stride,
          flags,
          [mapped, file_size](allocator::Buffer b) {
            vmlx_unregister_mmap_tensor(mapped, file_size);
            allocator::release(b);
            munmap(mapped, file_size);
          });
      whole_shard_mmap = true;
    }
  }
  if (!whole_shard_mmap && vmlx_mmap_safetensors_debug()) {
    std::fprintf(
        stderr,
        "[VMLX_MMAP_SAFETENSORS] whole-shard mmap unavailable %s len=%zu errno=%d %s; falling back to copy loader for tensors\n",
        file.c_str(),
        file_size,
        errno,
        std::strerror(errno));
  }

  auto close_fd = [&fd]() {
    if (fd >= 0) {
      close(fd);
      fd = -1;
    }
  };

  auto read_exact = [&](void* data, size_t n) {
    char* p = static_cast<char*>(data);
    size_t remaining = n;
    while (remaining > 0) {
      ssize_t got = read(fd, p, remaining);
      if (got <= 0) {
        int err = errno;
        close_fd();
        throw std::runtime_error(
            "[load_safetensors_mmap] short read " + file + ": " +
            std::strerror(err));
      }
      p += got;
      remaining -= static_cast<size_t>(got);
    }
  };

  uint64_t jsonHeaderLength = 0;
  constexpr uint64_t kMaxJsonHeaderLength = 100000000;
  read_exact(&jsonHeaderLength, 8);
  if (jsonHeaderLength <= 0 || jsonHeaderLength >= kMaxJsonHeaderLength) {
    close_fd();
    throw std::runtime_error(
        "[load_safetensors_mmap] invalid json header length " + file);
  }

  std::string rawJson(jsonHeaderLength, '\0');
  read_exact(rawJson.data(), jsonHeaderLength);
  auto metadata = json::parse(rawJson.begin(), rawJson.end());
  if (!metadata.is_object()) {
    close_fd();
    throw std::runtime_error("[load_safetensors_mmap] invalid json metadata " + file);
  }

  auto stream = cu::is_available() ? to_stream(s) : to_stream(s, Device::cpu);
  const size_t payload_start = jsonHeaderLength + 8;
  if (payload_start > file_size) {
    close_fd();
    throw std::runtime_error(
        "[load_safetensors_mmap] payload starts beyond end of file " + file);
  }
  std::unordered_map<std::string, array> res;
  std::unordered_map<std::string, std::string> metadata_map;
  auto fallback_reader = std::make_shared<io::ParallelFileReader>(file);
  size_t mmap_count = 0;
  size_t fallback_count = 0;
  size_t mmap_bytes = 0;

  for (const auto& item : metadata.items()) {
    if (item.key() == "__metadata__") {
      for (const auto& meta_item : item.value().items()) {
        metadata_map.insert({meta_item.key(), meta_item.value()});
      }
      continue;
    }
    const std::string& dtype = item.value().at("dtype");
    const Shape& shape = item.value().at("shape");
    const std::vector<size_t>& data_offsets = item.value().at("data_offsets");
    Dtype type = dtype_from_safetensor_str(dtype);
    const size_t tensor_offset = payload_start + data_offsets.at(0);

    std::optional<array> mapped_array;
    if (whole_shard_mmap && data_offsets.size() >= 2 &&
        data_offsets.at(1) <= file_size - payload_start) {
      mapped_array = make_mmap_backed_array(
          file, shard_holder, mapped, file_size,
          item.key(), shape, type, tensor_offset);
    }

    if (mapped_array) {
      size_t element_count = 1;
      for (auto dim : shape) {
        element_count *= static_cast<size_t>(dim);
      }
      mmap_bytes += element_count * size_of(type);
      mmap_count++;
      res.insert({item.key(), std::move(*mapped_array)});
    } else {
      fallback_count++;
      res.insert(
          {item.key(),
           array(
               shape,
               type,
               std::make_shared<Load>(
                   stream, fallback_reader, tensor_offset, false),
               std::vector<array>{})});
    }
  }

  close_fd();
  if (vmlx_mmap_safetensors_debug()) {
    std::fprintf(
        stderr,
        "[VMLX_MMAP_SAFETENSORS] %s whole_shard=%d mmap_tensors=%zu fallback_tensors=%zu mmap_bytes=%zu\n",
        file.c_str(),
        whole_shard_mmap ? 1 : 0,
        mmap_count,
        fallback_count,
        mmap_bytes);
  }
  return {res, metadata_map};
}

} // namespace

extern "C" int64_t mlx_safetensors_mmap_advise_routed(
    int32_t advice,
    int32_t pct) {
  std::vector<VmlxMmapTensorRecord> records;
  {
    std::lock_guard<std::mutex> guard(vmlx_mmap_registry_mutex());
    for (const auto& r : vmlx_mmap_registry()) {
      if (r.active && vmlx_parse_routed_tensor_name(r.name).matched) {
        records.push_back(r);
      }
    }
  }
  if (records.empty()) {
    return 0;
  }
  std::sort(records.begin(), records.end(), [](const auto& a, const auto& b) {
    return a.name < b.name;
  });
  const int clamped = std::max(0, std::min(100, static_cast<int>(pct)));
  const size_t count = clamped == 100
      ? records.size()
      : (records.size() * static_cast<size_t>(clamped) + 99) / 100;
  int64_t total = 0;
  for (size_t i = 0; i < count && i < records.size(); ++i) {
    auto* addr = static_cast<char*>(records[i].mapped) + records[i].page_delta;
    total += vmlx_advise_page_range(addr, records[i].tensor_nbytes, advice);
  }
  return total;
}

extern "C" int64_t mlx_safetensors_mmap_advise_experts(
    int32_t advice,
    const int32_t* layers,
    const int32_t* experts,
    int64_t count) {
  if (layers == nullptr || experts == nullptr || count <= 0) {
    return 0;
  }
  std::vector<VmlxMmapTensorRecord> records;
  {
    std::lock_guard<std::mutex> guard(vmlx_mmap_registry_mutex());
    for (const auto& r : vmlx_mmap_registry()) {
      if (r.active) {
        records.push_back(r);
      }
    }
  }

  int64_t total = 0;
  for (int64_t i = 0; i < count; ++i) {
    const int layer = static_cast<int>(layers[i]);
    const int expert = static_cast<int>(experts[i]);
    for (const auto& r : records) {
      auto parsed = vmlx_parse_routed_tensor_name(r.name);
      if (!parsed.matched || parsed.layer != layer) {
        continue;
      }
      total += vmlx_advise_record_expert(r, expert, advice);
    }
  }
  return total;
}
#endif

/** Load array from reader in safetensor format */
SafetensorsLoad load_safetensors(
    std::shared_ptr<io::Reader> in_stream,
    StreamOrDevice s) {
  ////////////////////////////////////////////////////////
  // Open and check file
  if (!in_stream->good() || !in_stream->is_open()) {
    throw std::runtime_error(
        "[load_safetensors] Failed to open " + in_stream->label());
  }

  auto stream = cu::is_available() ? to_stream(s) : to_stream(s, Device::cpu);

  uint64_t jsonHeaderLength = 0;
  // This is the same limit as in the original Rust Safetensors code.
  constexpr uint64_t kMaxJsonHeaderLength = 100000000;
  in_stream->read(reinterpret_cast<char*>(&jsonHeaderLength), 8);
  if (jsonHeaderLength <= 0 || jsonHeaderLength >= kMaxJsonHeaderLength) {
    throw std::runtime_error(
        "[load_safetensors] Invalid json header length " + in_stream->label());
  }
  // Load the json metadata
  auto rawJson = std::make_unique<char[]>(jsonHeaderLength);
  in_stream->read(rawJson.get(), jsonHeaderLength);
  auto metadata = json::parse(rawJson.get(), rawJson.get() + jsonHeaderLength);
  // Should always be an object on the top-level
  if (!metadata.is_object()) {
    throw std::runtime_error(
        "[load_safetensors] Invalid json metadata " + in_stream->label());
  }
  size_t offset = jsonHeaderLength + 8;
  // Load the arrays using metadata
  std::unordered_map<std::string, array> res;
  std::unordered_map<std::string, std::string> metadata_map;
  for (const auto& item : metadata.items()) {
    if (item.key() == "__metadata__") {
      for (const auto& meta_item : item.value().items()) {
        metadata_map.insert({meta_item.key(), meta_item.value()});
      }
      continue;
    }
    const std::string& dtype = item.value().at("dtype");
    const Shape& shape = item.value().at("shape");
    const std::vector<size_t>& data_offsets = item.value().at("data_offsets");
    Dtype type = dtype_from_safetensor_str(dtype);
    res.insert(
        {item.key(),
         array(
             shape,
             type,
             std::make_shared<Load>(
                 stream, in_stream, offset + data_offsets.at(0), false),
             std::vector<array>{})});
  }
  return {res, metadata_map};
}

SafetensorsLoad load_safetensors(const std::string& file, StreamOrDevice s) {
#ifndef _WIN32
  if (vmlx_mmap_safetensors_enabled()) {
    return load_safetensors_mmap(file, s);
  }
#endif
  return load_safetensors(std::make_shared<io::ParallelFileReader>(file), s);
}

void save_safetensors(
    std::shared_ptr<io::Writer> out_stream,
    std::unordered_map<std::string, array> a,
    std::unordered_map<std::string, std::string> metadata /* = {} */) {
  ////////////////////////////////////////////////////////
  // Check file
  if (!out_stream->good() || !out_stream->is_open()) {
    throw std::runtime_error(
        "[save_safetensors] Failed to open " + out_stream->label());
  }

  ////////////////////////////////////////////////////////
  // Check array map
  json parent;
  json _metadata;
  for (auto& [key, value] : metadata) {
    _metadata[key] = value;
  }
  parent["__metadata__"] = _metadata;

  {
    std::vector<array> to_eval;
    to_eval.reserve(a.size());
    for (auto& p : a) {
      p.second = contiguous(p.second);
      to_eval.push_back(p.second);
    }
    eval(std::move(to_eval));
  }

  size_t offset = 0;
  for (auto& [key, arr] : a) {
    if (arr.nbytes() == 0) {
      throw std::invalid_argument(
          "[save_safetensors] cannot serialize an empty array key: " + key);
    }

    json child;
    child["dtype"] = dtype_to_safetensor_str(arr.dtype());
    child["shape"] = arr.shape();
    child["data_offsets"] = std::vector<size_t>{offset, offset + arr.nbytes()};
    parent[key] = child;
    offset += arr.nbytes();
  }

  auto header = parent.dump();
  uint64_t header_len = header.length();
  out_stream->write(reinterpret_cast<char*>(&header_len), 8);
  out_stream->write(header.c_str(), header_len);
  for (auto& [key, arr] : a) {
    out_stream->write(arr.data<char>(), arr.nbytes());
  }
}

void save_safetensors(
    std::string file,
    std::unordered_map<std::string, array> a,
    std::unordered_map<std::string, std::string> metadata /* = {} */) {
  // Add .safetensors to file name if it is not there
  if (file.length() < 12 ||
      file.substr(file.length() - 12, 12) != ".safetensors")
    file += ".safetensors";

  // Serialize array
  save_safetensors(
      std::make_shared<io::FileWriter>(std::move(file)), a, metadata);
}

} // namespace mlx::core
