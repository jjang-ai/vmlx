// Copyright © 2023-2024 Apple Inc.

#define MLX_MTL_CONST static constant constexpr const
#define MLX_MTL_LOOP_UNROLL _Pragma("clang loop unroll(full)")

using namespace metal;

// Based on GPU merge sort algorithm at
// https://github.com/NVIDIA/cccl/tree/main/cub/cub

///////////////////////////////////////////////////////////////////////////////
// Thread-level sort
///////////////////////////////////////////////////////////////////////////////

template <typename T>
METAL_FUNC void thread_swap(thread T& a, thread T& b) {
  T w = a;
  a = b;
  b = w;
}

template <typename T, typename = void>
struct Init {
  static constexpr constant T v = Limits<T>::max;
};

template <typename T>
struct Init<T, metal::enable_if_t<metal::is_floating_point_v<T>>> {
  static constexpr constant T v = metal::numeric_limits<T>::quiet_NaN();
};

template <>
struct Init<complex64_t> {
  static constexpr constant complex64_t v = complex64_t(
      metal::numeric_limits<float>::quiet_NaN(),
      metal::numeric_limits<float>::quiet_NaN());
};

template <typename T>
struct LessThan {
  static constexpr constant T init = Init<T>::v;
  METAL_FUNC bool operator()(T a, T b) const thread {
    if constexpr (metal::is_floating_point_v<T>) {
      bool an = metal::isnan(a);
      bool bn = metal::isnan(b);
      if (an | bn) {
        return (!an) & bn;
      }
    } else if constexpr (metal::is_same_v<T, complex64_t>) {
      bool an = metal::isnan(a.real) || metal::isnan(a.imag);
      bool bn = metal::isnan(b.real) || metal::isnan(b.imag);
      if (an | bn) {
        return (!an) & bn;
      }
    }
    return a < b;
  }
};

template <
    typename ValT,
    typename IdxT,
    bool ARG_SORT,
    short N_PER_THREAD,
    typename CompareOp>
struct ThreadSort {
  static METAL_FUNC void sort(
      thread ValT (&vals)[N_PER_THREAD],
      thread IdxT (&idxs)[N_PER_THREAD]) {
    CompareOp op;
    MLX_MTL_LOOP_UNROLL
    for (short i = 0; i < N_PER_THREAD; ++i) {
      MLX_MTL_LOOP_UNROLL
      for (short j = i & 1; j < N_PER_THREAD - 1; j += 2) {
        if (op(vals[j + 1], vals[j])) {
          thread_swap(vals[j + 1], vals[j]);
          if (ARG_SORT) {
            thread_swap(idxs[j + 1], idxs[j]);
          }
        }
      }
    }
  }
};

///////////////////////////////////////////////////////////////////////////////
// Threadgroup-level sort
///////////////////////////////////////////////////////////////////////////////

template <
    typename ValT,
    typename IdxT,
    bool ARG_SORT,
    short BLOCK_THREADS,
    short N_PER_THREAD,
    typename CompareOp>
struct BlockMergeSort {
  using thread_sort_t =
      ThreadSort<ValT, IdxT, ARG_SORT, N_PER_THREAD, CompareOp>;
  static METAL_FUNC int merge_partition(
      const threadgroup ValT* As,
      const threadgroup ValT* Bs,
      short A_sz,
      short B_sz,
      short sort_md) {
    CompareOp op;

    short A_st = max(0, sort_md - B_sz);
    short A_ed = min(sort_md, A_sz);

    while (A_st < A_ed) {
      short md = A_st + (A_ed - A_st) / 2;
      auto a = As[md];
      auto b = Bs[sort_md - 1 - md];

      if (op(b, a)) {
        A_ed = md;
      } else {
        A_st = md + 1;
      }
    }

    return A_ed;
  }

  static METAL_FUNC void merge_step(
      const threadgroup ValT* As,
      const threadgroup ValT* Bs,
      const threadgroup IdxT* As_idx,
      const threadgroup IdxT* Bs_idx,
      short A_sz,
      short B_sz,
      thread ValT (&vals)[N_PER_THREAD],
      thread IdxT (&idxs)[N_PER_THREAD]) {
    CompareOp op;
    short a_idx = 0;
    short b_idx = 0;

    for (int i = 0; i < N_PER_THREAD; ++i) {
      auto a = (a_idx < A_sz) ? As[a_idx] : ValT(CompareOp::init);
      auto b = (b_idx < B_sz) ? Bs[b_idx] : ValT(CompareOp::init);
      bool pred = (b_idx < B_sz) && (a_idx >= A_sz || op(b, a));

      vals[i] = pred ? b : a;
      if (ARG_SORT) {
        if (pred) {
          idxs[i] = Bs_idx[b_idx];
        } else {
          idxs[i] = (a_idx < A_sz) ? As_idx[a_idx] : IdxT(0);
        }
      }

      b_idx += short(pred);
      a_idx += short(!pred);
    }
  }

  static METAL_FUNC void sort(
      threadgroup ValT* tgp_vals [[threadgroup(0)]],
      threadgroup IdxT* tgp_idxs [[threadgroup(1)]],
      int size_sorted_axis,
      uint3 lid [[thread_position_in_threadgroup]]) {
    // Get thread location
    int idx = lid.x * N_PER_THREAD;

    // Load from shared memory
    thread ValT thread_vals[N_PER_THREAD];
    thread IdxT thread_idxs[N_PER_THREAD];
    for (int i = 0; i < N_PER_THREAD; ++i) {
      thread_vals[i] = tgp_vals[idx + i];
      if (ARG_SORT) {
        thread_idxs[i] = tgp_idxs[idx + i];
      }
    }

    // Per thread sort
    if (idx < size_sorted_axis) {
      thread_sort_t::sort(thread_vals, thread_idxs);
    }

    // Do merges using threadgroup memory
    for (int merge_threads = 2; merge_threads <= BLOCK_THREADS;
         merge_threads *= 2) {
      // Update threadgroup memory
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int i = 0; i < N_PER_THREAD; ++i) {
        tgp_vals[idx + i] = thread_vals[i];
        if (ARG_SORT) {
          tgp_idxs[idx + i] = thread_idxs[i];
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      // Find location in merge step
      int merge_group = lid.x / merge_threads;
      int merge_lane = lid.x % merge_threads;

      int sort_sz = N_PER_THREAD * merge_threads;
      int sort_st = N_PER_THREAD * merge_threads * merge_group;

      // As = tgp_vals[A_st:A_ed] is sorted
      // Bs = tgp_vals[B_st:B_ed] is sorted
      int A_st = sort_st;
      int A_ed = sort_st + sort_sz / 2;
      int B_st = sort_st + sort_sz / 2;
      int B_ed = sort_st + sort_sz;

      const threadgroup ValT* As = tgp_vals + A_st;
      const threadgroup ValT* Bs = tgp_vals + B_st;
      int A_sz = A_ed - A_st;
      int B_sz = B_ed - B_st;

      // Find a partition of merge elements
      //  Ci = merge(As[partition:], Bs[sort_md - partition:])
      //       of size N_PER_THREAD for each merge lane i
      //  C = [Ci] is sorted
      int sort_md = N_PER_THREAD * merge_lane;
      int partition = merge_partition(As, Bs, A_sz, B_sz, sort_md);

      As += partition;
      Bs += sort_md - partition;

      A_sz -= partition;
      B_sz -= sort_md - partition;

      const threadgroup IdxT* As_idx =
          ARG_SORT ? tgp_idxs + A_st + partition : nullptr;
      const threadgroup IdxT* Bs_idx =
          ARG_SORT ? tgp_idxs + B_st + sort_md - partition : nullptr;

      // Merge starting at the partition and store results in thread registers
      merge_step(As, Bs, As_idx, Bs_idx, A_sz, B_sz, thread_vals, thread_idxs);
    }

    // Write out to shared memory
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < N_PER_THREAD; ++i) {
      tgp_vals[idx + i] = thread_vals[i];
      if (ARG_SORT) {
        tgp_idxs[idx + i] = thread_idxs[i];
      }
    }
  }
};

///////////////////////////////////////////////////////////////////////////////
